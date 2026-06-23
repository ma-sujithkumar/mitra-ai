"""EvidencePacket dataclasses — sole contract for what each tool sends to the model.

Every model call receives a typed EvidencePacket constructed by the calling tool.
The dataclass is the contract; adding a field is a code change here, never a
prompt edit.

The `render(packet)` serializer returns (prompt_text, sent_field_names: set[str]).
`sent_field_names` is the whitelist the response validator uses to membership-check
`evidence_cited`. Nested dataclass fields are flattened with dotted paths
(e.g. `columns.null_run_lengths`).
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, is_dataclass
from typing import Any


# ---------- per-tool evidence packets ----------


@dataclass
class ColumnTypeEvidence:
    name: str
    dtype: str
    null_rate: float
    nunique: int
    top_values: list[str]            # up to 5
    random_samples: list[str]        # 5 string-cast values
    regex_signature: dict[str, int]  # {"uuid": n, "email": n, "iso_date": n, "phone": n, "numeric_string": n}


@dataclass
class SemanticTypeInferEvidence:
    columns: list[ColumnTypeEvidence]


@dataclass
class NullColumnEvidence:
    name: str
    null_rate: float
    null_run_lengths: list[int]
    null_mask_corr_top5: dict[str, float]
    target_rate_when_null: float | None
    target_rate_when_present: float | None
    random_present_values: list[str]
    dtype: str
    semantic_type: str


@dataclass
class MissingValueEvidence:
    columns: list[NullColumnEvidence]


@dataclass
class OutlierColumnEvidence:
    name: str
    histogram_10bin: list[int]
    extreme_top5: list[tuple[float, Any]]
    extreme_bottom5: list[tuple[float, Any]]
    mi_with_target: float
    target_corr: float


@dataclass
class OutlierEvidence:
    columns: list[OutlierColumnEvidence]
    downstream_model_hint: str  # "linear" | "tree"


@dataclass
class ScalerColumnEvidence:
    name: str
    histogram_20bin: list[int]
    skewness: float
    kurtosis: float
    outlier_rate: float
    bounded: bool
    bounds: tuple[float, float] | None
    monotonic_with_target: float


@dataclass
class ScalerEvidence:
    columns: list[ScalerColumnEvidence]


@dataclass
class CreatorColumnEvidence:
    name: str
    semantic_type: str
    mi_with_target: float
    nunique: int
    correlated_with_top3: dict[str, float]
    decomposed_from: str | None


@dataclass
class FeatureCreatorEvidence:
    columns: list[CreatorColumnEvidence]
    co_occurring_pairs: list[tuple[str, str, float]]  # (col_a, col_b, joint_mi)


@dataclass
class ClusterEvidence:
    cluster_id: int
    members: list[str]
    mean_mi: float
    max_mi: float
    intra_cluster_corr: float


@dataclass
class FeatureSelectorEvidence:
    n_rows: int
    n_features: int
    task: str
    linear_baseline_score: float
    clusters: list[ClusterEvidence]


# ---------- serializer ----------


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def _collect_field_names(value: Any, prefix: str = "") -> set[str]:
    """Flatten field names with dotted paths.

    For a dataclass containing `columns: list[ColumnTypeEvidence]`, the sent
    field names include `columns`, plus `columns.name`, `columns.dtype`, ...
    """
    names: set[str] = set()
    if is_dataclass(value):
        for f in dataclasses.fields(value):
            child_path = f"{prefix}{f.name}" if not prefix else f"{prefix}.{f.name}"
            names.add(child_path)
            names |= _collect_field_names(getattr(value, f.name), child_path)
    elif isinstance(value, list) and value and is_dataclass(value[0]):
        # Same prefix — `columns.null_rate` rather than `columns.0.null_rate`.
        names |= _collect_field_names(value[0], prefix)
    return names


def render(packet: Any, truncate_after_chars: int | None = None) -> tuple[str, set[str]]:
    """Render an EvidencePacket to (prompt_block, sent_field_names).

    The block is a fenced JSON snippet so models can locate it precisely. If
    `truncate_after_chars` is given, verbose fields (histograms, random samples,
    extremes) are dropped in that order until the block fits. A trailing
    `## TRUNCATED` note records what was dropped.
    """
    if not is_dataclass(packet):
        raise TypeError(f"render() requires a dataclass instance, got {type(packet)}")

    sent_fields = _collect_field_names(packet)
    data = _to_jsonable(packet)
    body = json.dumps(data, indent=2, default=str)

    truncated_note = ""
    if truncate_after_chars is not None and len(body) > truncate_after_chars:
        verbose_keys = ["histogram_10bin", "histogram_20bin", "random_samples", "extreme_top5", "extreme_bottom5"]
        dropped: list[str] = []
        for key in verbose_keys:
            if len(body) <= truncate_after_chars:
                break
            data = _strip_key(data, key)
            dropped.append(key)
            body = json.dumps(data, indent=2, default=str)
        if dropped:
            truncated_note = f"\n## TRUNCATED\nDropped fields to fit budget: {', '.join(dropped)}\n"

    block = f"## EVIDENCE\n```json\n{body}\n```{truncated_note}"
    return block, sent_fields


def _strip_key(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return {k: ([] if k == key else _strip_key(v, key)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_key(v, key) for v in obj]
    return obj


def render_delta(packet: Any, prior_response: str, contrast_pairs: list[tuple[str, str, dict]]) -> str:
    """Render a delta-evidence pack for the revision retry.

    `contrast_pairs` is a list of (col_a, col_b, differing_fields_dict) tuples
    produced by the lazy-batch checker. Appended under `## REVISION`.
    """
    if not contrast_pairs:
        return ""
    blocks = []
    for col_a, col_b, diffs in contrast_pairs:
        blocks.append(
            json.dumps(
                {
                    "columns": [col_a, col_b],
                    "differing_fields": diffs,
                    "your_previous_answer": prior_response,
                },
                indent=2,
                default=str,
            )
        )
    return "\n## REVISION\nThese columns received the SAME answer despite different evidence. Reconsider.\n```json\n" + "\n".join(blocks) + "\n```\n"
