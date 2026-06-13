import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import pandas

from config_loader import ConfigLoader
from session import SessionManager

logger = logging.getLogger(__name__)

CHECK_ORDER = ["format", "rows", "nulls", "variance", "pii", "target"]

CHECK_LABELS = {
    "format":   "File format & encoding",
    "rows":     "Row count",
    "nulls":    "Null density",
    "variance": "Zero-variance scan",
    "pii":      "PII heuristic",
    "target":   "Target separability",
}


@dataclass
class ValidationCheckResult:
    key: str
    label: str
    status: str  # "pass" | "warn" | "fail"
    detail: str
    warn_message: Optional[str] = None


class DataValidator:

    def validate(
        self,
        session_id: str,
        target_col: Optional[str] = None,
    ) -> Generator[ValidationCheckResult, None, None]:
        data_csv = SessionManager.get_session_path(session_id, "data/data.csv")

        results = []
        blocker_count = 0
        warn_count = 0

        for check_key in CHECK_ORDER:
            logger.info(f"=> Running check: {check_key}")
            result = self._run_check(check_key, data_csv, target_col)
            results.append(result)
            if result.status == "fail":
                blocker_count += 1
            elif result.status == "warn":
                warn_count += 1
            yield result

        passed = blocker_count == 0
        self._write_report(session_id, results, passed, blocker_count, warn_count)
        logger.info(f"=> Validation done: passed={passed}, blockers={blocker_count}, warns={warn_count}")

    def _run_check(
        self,
        check_key: str,
        data_csv: Path,
        target_col: Optional[str],
    ) -> ValidationCheckResult:
        label = CHECK_LABELS[check_key]
        check_dispatch = {
            "format":   self._check_format,
            "rows":     self._check_rows,
            "nulls":    self._check_nulls,
            "variance": self._check_variance,
            "pii":      self._check_pii,
            "target":   self._check_target,
        }
        return check_dispatch[check_key](label, data_csv, target_col)

    def _check_format(self, label: str, data_csv: Path, _target_col: Optional[str]) -> ValidationCheckResult:
        try:
            with open(data_csv, "rb") as file_handle:
                raw = file_handle.read(4096)

            try:
                raw.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                return ValidationCheckResult(
                    key="format", label=label, status="fail",
                    detail="File is not UTF-8 encoded or is binary",
                )

            sample = pandas.read_csv(data_csv, nrows=5)
            num_cols = len(sample.columns)
            return ValidationCheckResult(
                key="format", label=label, status="pass",
                detail=f"utf-8, comma-delimited, {num_cols} columns",
            )
        except Exception as exc:
            return ValidationCheckResult(
                key="format", label=label, status="fail",
                detail=f"Cannot parse file: {exc}",
            )

    def _check_rows(self, label: str, data_csv: Path, _target_col: Optional[str]) -> ValidationCheckResult:
        min_rows = ConfigLoader.get_int("upload", "MIN_ROWS")
        chunk_size = ConfigLoader.get_int("upload", "CHUNK_SIZE_ROWS")

        total_rows = 0
        for chunk in pandas.read_csv(data_csv, chunksize=chunk_size):
            total_rows += len(chunk)

        if total_rows < min_rows:
            return ValidationCheckResult(
                key="rows", label=label, status="fail",
                detail=f"{total_rows} rows — below minimum ({min_rows})",
            )
        return ValidationCheckResult(
            key="rows", label=label, status="pass",
            detail=f"{total_rows} rows, above minimum ({min_rows})",
        )

    def _check_nulls(self, label: str, data_csv: Path, _target_col: Optional[str]) -> ValidationCheckResult:
        null_threshold = ConfigLoader.get_float("upload", "NULL_THRESHOLD")
        chunk_size = ConfigLoader.get_int("upload", "CHUNK_SIZE_ROWS")

        null_counts: dict = {}
        total_rows = 0

        for chunk in pandas.read_csv(data_csv, chunksize=chunk_size):
            total_rows += len(chunk)
            for col_name in chunk.columns:
                null_counts[col_name] = null_counts.get(col_name, 0) + chunk[col_name].isna().sum()

        offending_cols = [
            col_name for col_name, count in null_counts.items()
            if total_rows > 0 and (count / total_rows) > null_threshold
        ]

        if offending_cols:
            return ValidationCheckResult(
                key="nulls", label=label, status="fail",
                detail=f"{len(offending_cols)} columns exceed {int(null_threshold*100)}% null threshold: {', '.join(offending_cols[:3])}",
            )
        return ValidationCheckResult(
            key="nulls", label=label, status="pass",
            detail=f"0 columns exceed {int(null_threshold*100)}% threshold",
        )

    def _check_variance(self, label: str, data_csv: Path, _target_col: Optional[str]) -> ValidationCheckResult:
        chunk_size = ConfigLoader.get_int("upload", "CHUNK_SIZE_ROWS")

        sum_sq: dict = {}
        sums: dict = {}
        counts: dict = {}

        for chunk in pandas.read_csv(data_csv, chunksize=chunk_size):
            numeric_chunk = chunk.select_dtypes(include=["number"])
            for col_name in numeric_chunk.columns:
                col_values = numeric_chunk[col_name].dropna()
                counts[col_name] = counts.get(col_name, 0) + len(col_values)
                sums[col_name] = sums.get(col_name, 0.0) + float(col_values.sum())
                sum_sq[col_name] = sum_sq.get(col_name, 0.0) + float((col_values ** 2).sum())

        zero_variance_cols = []
        for col_name, count in counts.items():
            if count < 2:
                continue
            mean_val = sums[col_name] / count
            variance = (sum_sq[col_name] / count) - (mean_val ** 2)
            if abs(variance) < 1e-10:
                zero_variance_cols.append(col_name)

        if zero_variance_cols:
            return ValidationCheckResult(
                key="variance", label=label, status="fail",
                detail=f"Constant columns detected: {', '.join(zero_variance_cols[:3])}",
            )
        return ValidationCheckResult(
            key="variance", label=label, status="pass",
            detail="No constant columns detected",
        )

    def _check_pii(self, label: str, data_csv: Path, _target_col: Optional[str]) -> ValidationCheckResult:
        pii_patterns = ConfigLoader.get_json_list("upload", "PII_PATTERNS")
        sample = pandas.read_csv(data_csv, nrows=1)
        col_names = list(sample.columns)

        matched_cols = []
        for col_name in col_names:
            for pattern in pii_patterns:
                if re.search(pattern, col_name):
                    matched_cols.append(col_name)
                    break

        if matched_cols:
            return ValidationCheckResult(
                key="pii", label=label, status="warn",
                detail=f"PII-suspect column names: {', '.join(matched_cols)}",
                warn_message=f"Columns matching PII patterns: {', '.join(matched_cols)}",
            )
        return ValidationCheckResult(
            key="pii", label=label, status="pass",
            detail="No PII-suspect column names",
        )

    def _check_target(self, label: str, data_csv: Path, target_col: Optional[str]) -> ValidationCheckResult:
        if not target_col:
            return ValidationCheckResult(
                key="target", label=label, status="pass",
                detail="No target column specified (unsupervised mode)",
            )

        sample = pandas.read_csv(data_csv, nrows=5)
        if target_col not in sample.columns:
            return ValidationCheckResult(
                key="target", label=label, status="fail",
                detail=f"Target column '{target_col}' not found in dataset",
            )

        chunk_size = ConfigLoader.get_int("upload", "CHUNK_SIZE_ROWS")
        class_counts: dict = {}
        total_rows = 0

        for chunk in pandas.read_csv(data_csv, chunksize=chunk_size):
            total_rows += len(chunk)
            value_counts = chunk[target_col].value_counts()
            for value, count in value_counts.items():
                class_counts[value] = class_counts.get(value, 0) + count

        num_classes = len(class_counts)
        class_ratios = sorted([count / total_rows for count in class_counts.values()])

        # Imbalance check: smallest class < 20% of largest
        imbalanced = (
            len(class_ratios) > 1
            and class_ratios[0] < 0.2 * class_ratios[-1]
        )

        detail = f"{target_col} - {num_classes} classes"
        if imbalanced:
            return ValidationCheckResult(
                key="target", label=label, status="warn",
                detail=detail,
                warn_message="Mild class imbalance detected",
            )
        return ValidationCheckResult(
            key="target", label=label, status="pass",
            detail=f"{detail}, balanced",
        )

    def _write_report(
        self,
        session_id: str,
        results: list[ValidationCheckResult],
        passed: bool,
        blocker_count: int,
        warn_count: int,
    ) -> None:
        report = {
            "session_id": session_id,
            "passed": passed,
            "blocker_count": blocker_count,
            "warn_count": warn_count,
            "checks": [
                {
                    "key": result.key,
                    "label": result.label,
                    "status": result.status,
                    "detail": result.detail,
                    **({"warn_message": result.warn_message} if result.warn_message else {}),
                }
                for result in results
            ],
        }
        report_path = SessionManager.get_session_path(session_id, "reports/validation_report.json")
        with open(report_path, "w") as report_file:
            json.dump(report, report_file, indent=2)
