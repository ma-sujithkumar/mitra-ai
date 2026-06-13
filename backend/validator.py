from __future__ import annotations

import csv
import re
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class ValidationCheckResult:
    key: str
    label: str
    status: str
    detail: str
    warn_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        result_dict = asdict(self)
        if self.warn_message is None:
            result_dict.pop("warn_message")
        return result_dict


@dataclass(frozen=True)
class DatasetValidationSummary:
    row_count: int
    column_count: int
    column_names: list[str]
    null_counts: dict[str, int]
    numeric_unique_values: dict[str, set[object]]
    target_unique_count: int | None = None


@dataclass(frozen=True)
class ValidationReport:
    session_id: str
    passed: bool
    blocker_count: int
    warn_count: int
    checks: list[ValidationCheckResult]

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "passed": self.passed,
            "blocker_count": self.blocker_count,
            "warn_count": self.warn_count,
            "checks": [check_result.to_dict() for check_result in self.checks],
        }


class DataValidator:
    check_order = ["format", "rows", "nulls", "variance", "pii", "target"]
    check_labels = {
        "format": "File format & encoding",
        "rows": "Row count",
        "nulls": "Null density",
        "variance": "Zero-variance scan",
        "pii": "PII heuristic",
        "target": "Target separability",
    }
    blocker_statuses = {
        "format": {"fail"},
        "rows": {"fail"},
        "nulls": {"fail"},
        "variance": {"fail"},
        "target": {"fail"},
    }

    def __init__(
        self,
        min_rows: int,
        null_threshold: float,
        pii_patterns: list[str],
        chunk_size_rows: int = 50000,
    ) -> None:
        self.min_rows = min_rows
        self.null_threshold = null_threshold
        self.pii_patterns = pii_patterns
        self.chunk_size_rows = chunk_size_rows

    def validate(
        self,
        data_file: Path,
        session_id: str,
        target_col: str | None,
    ) -> Iterator[ValidationCheckResult]:
        summary = self._summarize_csv(
            data_file=data_file,
            target_col=target_col,
        )

        checks_by_key = {
            "format": self._check_format(data_file=data_file, summary=summary),
            "rows": self._check_rows(summary=summary),
            "nulls": self._check_nulls(summary=summary),
            "variance": self._check_variance(summary=summary),
            "pii": self._check_pii(column_names=summary.column_names),
            "target": self._check_target(
                summary=summary,
                target_col=target_col,
            ),
        }
        for check_key in self.check_order:
            yield checks_by_key[check_key]

    def build_report(
        self,
        session_id: str,
        checks: list[ValidationCheckResult],
    ) -> ValidationReport:
        blocker_count = sum(
            1
            for check_result in checks
            if check_result.status in self.blocker_statuses.get(check_result.key, set())
        )
        warn_count = sum(1 for check_result in checks if check_result.status == "warn")
        return ValidationReport(
            session_id=session_id,
            passed=blocker_count == 0,
            blocker_count=blocker_count,
            warn_count=warn_count,
            checks=checks,
        )

    def _check_format(
        self,
        data_file: Path,
        summary: DatasetValidationSummary,
    ) -> ValidationCheckResult:
        try:
            with data_file.open("r", encoding="utf-8", newline="") as csv_file:
                sample_text = csv_file.read(4096)
                csv.Sniffer().sniff(sample_text)
        except (UnicodeDecodeError, csv.Error) as error:
            return ValidationCheckResult(
                key="format",
                label=self.check_labels["format"],
                status="fail",
                detail=f"Could not parse CSV format: {error}",
            )

        return ValidationCheckResult(
            key="format",
            label=self.check_labels["format"],
            status="pass",
            detail=f"utf-8, delimiter detected, {summary.column_count} columns",
        )

    def _check_rows(
        self,
        summary: DatasetValidationSummary,
    ) -> ValidationCheckResult:
        if summary.row_count < self.min_rows:
            return ValidationCheckResult(
                key="rows",
                label=self.check_labels["rows"],
                status="fail",
                detail=f"{summary.row_count} rows, below minimum ({self.min_rows})",
            )

        return ValidationCheckResult(
            key="rows",
            label=self.check_labels["rows"],
            status="pass",
            detail=f"{summary.row_count} rows, above minimum ({self.min_rows})",
        )

    def _check_nulls(
        self,
        summary: DatasetValidationSummary,
    ) -> ValidationCheckResult:
        if summary.row_count == 0:
            return ValidationCheckResult(
                key="nulls",
                label=self.check_labels["nulls"],
                status="pass",
                detail="0 columns exceed null threshold",
            )

        null_heavy_columns = [
            column_name
            for column_name, null_count in summary.null_counts.items()
            if (null_count / summary.row_count) > self.null_threshold
        ]
        if null_heavy_columns:
            return ValidationCheckResult(
                key="nulls",
                label=self.check_labels["nulls"],
                status="fail",
                detail=f"{len(null_heavy_columns)} columns exceed null threshold",
            )

        return ValidationCheckResult(
            key="nulls",
            label=self.check_labels["nulls"],
            status="pass",
            detail="0 columns exceed null threshold",
        )

    def _check_variance(
        self,
        summary: DatasetValidationSummary,
    ) -> ValidationCheckResult:
        zero_variance_columns = [
            column_name
            for column_name, unique_values in summary.numeric_unique_values.items()
            if len(unique_values) <= 1
        ]
        if zero_variance_columns:
            return ValidationCheckResult(
                key="variance",
                label=self.check_labels["variance"],
                status="fail",
                detail=f"Constant columns detected: {', '.join(zero_variance_columns)}",
            )

        return ValidationCheckResult(
            key="variance",
            label=self.check_labels["variance"],
            status="pass",
            detail="No constant numeric columns detected",
        )

    def _check_pii(self, column_names: list[str]) -> ValidationCheckResult:
        pii_columns = [
            column_name
            for column_name in column_names
            if self._matches_pii_pattern(column_name=column_name)
        ]
        if pii_columns:
            matched_columns = ", ".join(pii_columns)
            return ValidationCheckResult(
                key="pii",
                label=self.check_labels["pii"],
                status="warn",
                detail=f"PII-suspect columns: {matched_columns}",
                warn_message=f"Column names match PII patterns: {matched_columns}",
            )

        return ValidationCheckResult(
            key="pii",
            label=self.check_labels["pii"],
            status="pass",
            detail="No PII-suspect column names",
        )

    def _check_target(
        self,
        summary: DatasetValidationSummary,
        target_col: str | None,
    ) -> ValidationCheckResult:
        normalized_target_col = (target_col or "").strip()
        if not normalized_target_col:
            return ValidationCheckResult(
                key="target",
                label=self.check_labels["target"],
                status="pass",
                detail="No target column supplied; unsupervised flow allowed",
            )
        if normalized_target_col not in summary.column_names:
            return ValidationCheckResult(
                key="target",
                label=self.check_labels["target"],
                status="fail",
                detail=f"Target column missing: {normalized_target_col}",
            )

        unique_target_count = summary.target_unique_count or 0
        return ValidationCheckResult(
            key="target",
            label=self.check_labels["target"],
            status="pass",
            detail=f"{normalized_target_col}, {unique_target_count} unique values",
        )

    def _matches_pii_pattern(self, column_name: str) -> bool:
        return any(
            re.search(pattern, column_name) is not None
            for pattern in self.pii_patterns
        )

    def _summarize_csv(
        self,
        data_file: Path,
        target_col: str | None,
    ) -> DatasetValidationSummary:
        row_count = 0
        column_names: list[str] = []
        null_counts: dict[str, int] = {}
        numeric_unique_values: dict[str, set[object]] = {}
        target_unique_values: set[object] = set()
        normalized_target_col = (target_col or "").strip()

        for data_chunk in pd.read_csv(data_file, chunksize=self.chunk_size_rows):
            if not column_names:
                column_names = list(data_chunk.columns)
                null_counts = {column_name: 0 for column_name in column_names}
            row_count += len(data_chunk)

            chunk_null_counts = data_chunk.isna().sum()
            for column_name, null_count in chunk_null_counts.items():
                null_counts[column_name] = null_counts.get(column_name, 0) + int(null_count)

            numeric_chunk = data_chunk.select_dtypes(include="number")
            for column_name in numeric_chunk.columns:
                unique_values = numeric_unique_values.setdefault(column_name, set())
                if len(unique_values) <= 1:
                    unique_values.update(numeric_chunk[column_name].dropna().unique().tolist())

            if normalized_target_col in data_chunk.columns:
                target_unique_values.update(
                    data_chunk[normalized_target_col].dropna().unique().tolist()
                )

        target_unique_count = (
            len(target_unique_values)
            if normalized_target_col and normalized_target_col in column_names
            else None
        )
        return DatasetValidationSummary(
            row_count=row_count,
            column_count=len(column_names),
            column_names=column_names,
            null_counts=null_counts,
            numeric_unique_values=numeric_unique_values,
            target_unique_count=target_unique_count,
        )
