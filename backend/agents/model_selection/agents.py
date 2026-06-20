"""Agents used by the MITRA model-selection stage.

The component is intentionally bounded:

1. ``ModelLibraryCatalogAgent`` discovers allowed names from MLKit.
2. ``DatasetProfileAgent`` converts upstream artifacts into a compact profile.
3. ``LLMModelRankingAgent`` may make one LLM call, but can only suggest names
   present in the discovered catalog.
4. ``DeterministicRankingAgent`` provides a no-network fallback.
5. ``ModelSelectionValidationAgent`` is the final hard guard.
6. ``ModelSelectionOrchestratorAgent`` writes the trusted result atomically.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from .catalog import ModelLibraryCatalogAgent
from .errors import UnsupportedProblemTypeError
from .schemas import (
    DatasetProfile,
    FeatureSelectionInput,
    MetadataInput,
    ModelCandidate,
    ModelDescriptor,
    RankedSuggestion,
    SelectionReport,
)

logger = logging.getLogger(__name__)

# Upper bound on how many models a single run may select. Matches the
# config.ini [pipeline] MAX_ML_MODELS ceiling so the pipeline can request the
# full shortlist (the deterministic agent returns fewer if the catalog is
# smaller for the task).
MAX_SELECTABLE_MODELS = 10


class LLMClient(Protocol):
    """Small adapter expected from the project's LiteLLM factory."""

    def complete(self, prompt: str) -> str:
        """Return a text completion for one prompt."""


class DatasetProfileAgent:
    """Build a compact, deterministic profile without reading the raw dataset."""

    def run(
        self,
        metadata: MetadataInput,
        feature_selection: FeatureSelectionInput,
        mini_data_path: str | Path | None = None,
    ) -> DatasetProfile:
        if metadata.problem_type == "unsupervised":
            raise UnsupportedProblemTypeError(
                "The current MLKit registry contains classifiers and regressors only; "
                "it exposes no unsupervised model. Model selection stopped instead of "
                "inventing KMeans/DBSCAN entries outside model_library."
            )

        selected_names = list(feature_selection.keep)
        selected_names.extend(item.name for item in feature_selection.engineered)
        if not selected_names:
            selected_names = [
                column
                for column in metadata.input_cols
                if column not in set(metadata.drop_cols)
            ]
        selected_names = list(dict.fromkeys(selected_names))

        numeric = 0
        categorical = 0
        for column in selected_names:
            column_type = metadata.col_types.get(column, "")
            if column_type == "numeric":
                numeric += 1
            elif column_type in {"categorical", "text", "datetime"}:
                categorical += 1

        # mini_data is only a statistics artifact.  Reading its header and row
        # count is safe and helps when upstream omitted col_count.
        mini_rows = 0
        if mini_data_path:
            path = Path(mini_data_path)
            if path.is_file():
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle)
                    next(reader, None)
                    mini_rows = sum(1 for _ in reader)

        feature_count = len(selected_names) or metadata.col_count or mini_rows
        class_count, imbalance_ratio = self._class_balance(metadata.class_balance)

        return DatasetProfile(
            task_type=metadata.problem_type,
            data_format=metadata.data_format,
            row_count=metadata.row_count,
            original_col_count=metadata.col_count or mini_rows,
            selected_feature_count=feature_count,
            numeric_feature_count=numeric,
            categorical_feature_count=categorical,
            class_count=class_count,
            imbalance_ratio=imbalance_ratio,
            user_description=metadata.user_description,
        )

    @staticmethod
    def _class_balance(balance: dict[str, float | int]) -> tuple[int | None, float | None]:
        positive = [float(value) for value in balance.values() if float(value) > 0]
        if not positive:
            return None, None
        return len(positive), max(positive) / min(positive)


class DeterministicRankingAgent:
    """Rank catalog models using transparent dataset-profile heuristics."""

    def run(
        self,
        profile: DatasetProfile,
        descriptors: list[ModelDescriptor],
        max_models: int,
    ) -> list[RankedSuggestion]:
        ranked: list[RankedSuggestion] = []
        for descriptor in descriptors:
            score, reasons = self._score(descriptor, profile)
            if math.isinf(score) and score < 0:
                continue
            ranked.append(
                RankedSuggestion(
                    model_name=descriptor.model_name,
                    rationale="; ".join(reasons[:3]),
                    score=round(score, 3),
                )
            )

        ranked.sort(key=lambda item: (-item.score, item.model_name))
        return ranked[:max_models]

    def _score(
        self, descriptor: ModelDescriptor, profile: DatasetProfile
    ) -> tuple[float, list[str]]:
        name = descriptor.model_name
        normalized = name.lower()
        score = 10.0
        reasons: list[str] = []

        is_image_cnn = normalized.startswith("pytorchcnn")
        if profile.data_format == "image":
            if not is_image_cnn:
                return -math.inf, ["not an image-native model"]
            score += 100
            reasons.append("CNN wrapper is the image-native option in MLKit")
        else:
            if is_image_cnn:
                score -= 25
                reasons.append("CNN is deprioritized for tabular input")

        is_large = profile.row_count >= 50_000
        is_small = 0 < profile.row_count <= 2_000
        high_dimensional = profile.selected_feature_count >= max(
            100, int(max(profile.row_count, 1) * 0.25)
        )
        mostly_numeric = (
            profile.numeric_feature_count > profile.categorical_feature_count
        )

        # General-purpose nonlinear ensembles.
        if "xgb" in normalized:
            score += 34
            reasons.append("strong nonlinear tabular baseline")
        elif "histgradientboosting" in normalized:
            score += 31
            reasons.append("efficient boosting baseline")
        elif "randomforest" in normalized:
            score += 29
            reasons.append("robust ensemble baseline")
        elif "extratrees" in normalized:
            score += 27
            reasons.append("diverse tree ensemble")
        elif "gradientboosting" in normalized:
            score += 25
            reasons.append("strong boosting baseline")

        # Linear/high-dimensional candidates.
        if any(token in normalized for token in ("logistic", "linearsvc", "ridge", "elasticnet", "lasso")):
            score += 18
            reasons.append("stable linear baseline")
            if high_dimensional:
                score += 15
                reasons.append("well suited to high-dimensional features")

        # Kernel methods are attractive on smaller datasets but expensive at scale.
        if normalized in {"svc", "nusvc", "svr", "nusvr"}:
            score += 17
            reasons.append("captures nonlinear boundaries")
            if is_small:
                score += 12
                reasons.append("dataset size is suitable for kernel methods")
            if is_large:
                score -= 25
                reasons.append("kernel scaling cost on large datasets")

        if "mlp" in normalized or "pytorchfcnn" in normalized:
            score += 15
            reasons.append("neural nonlinear baseline")
            if profile.row_count >= 10_000:
                score += 8
                reasons.append("enough rows to support neural training")
            if is_small:
                score -= 8

        if any(token in normalized for token in ("dummy", "radiusneighbors", "theilsen")):
            score -= 35
            reasons.append("kept as a specialist/baseline model, not a top default")

        if "nearest" in normalized or "kneighbors" in normalized:
            score += 5 if is_small else -8

        if is_large and any(
            token in normalized
            for token in ("histgradientboosting", "xgb", "sgd", "passiveaggressive")
        ):
            score += 10
            reasons.append("scales better to a larger row count")

        if profile.imbalance_ratio and profile.imbalance_ratio > 5:
            if any(token in normalized for token in ("xgb", "randomforest", "histgradient")):
                score += 6
                reasons.append("robust candidate for imbalanced classes")

        if mostly_numeric and descriptor.task_type == "regression":
            if any(token in normalized for token in ("ridge", "elasticnet", "xgb", "randomforest")):
                score += 4

        if not reasons:
            reasons.append("valid MLKit model retained as a secondary candidate")
        return score, reasons


class LLMModelRankingAgent:
    """One-shot LLM ranker constrained to exact MLKit model names."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def run(
        self,
        profile: DatasetProfile,
        descriptors: list[ModelDescriptor],
        max_models: int,
    ) -> list[RankedSuggestion]:
        allowed = [descriptor.model_name for descriptor in descriptors]
        prompt = self._prompt(profile, allowed, max_models)
        raw = self.client.complete(prompt)
        return self._parse(raw)

    @staticmethod
    def _prompt(profile: DatasetProfile, allowed: list[str], max_models: int) -> str:
        return (
            "You are MITRA's bounded model-ranking agent. Select the best models "
            "for the supplied dataset profile. You MUST select only exact names "
            "from ALLOWED_MODELS. Do not invent architectures or aliases.\n\n"
            f"DATASET_PROFILE={profile.model_dump_json()}\n"
            f"ALLOWED_MODELS={json.dumps(allowed)}\n"
            f"Select at most {max_models} models.\n"
            "Return JSON only in this form: "
            '[{"model_name":"ExactName","rationale":"under 30 words"}]'
        )

    @staticmethod
    def _parse(raw: str) -> list[RankedSuggestion]:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("LLM response must be a JSON array")
        suggestions: list[RankedSuggestion] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict) or not isinstance(item.get("model_name"), str):
                raise ValueError("Every LLM item must contain string model_name")
            suggestions.append(
                RankedSuggestion(
                    model_name=item["model_name"],
                    rationale=str(item.get("rationale", "LLM-ranked library model")),
                    score=float(1000 - index),
                )
            )
        return suggestions


class ModelSelectionValidationAgent:
    """Final trust boundary: unknown or wrong-task models never pass downstream."""

    def run(
        self,
        suggestions: list[RankedSuggestion],
        catalog: dict[str, ModelDescriptor],
        task_type: str,
        max_models: int,
    ) -> tuple[list[RankedSuggestion], list[str]]:
        accepted: list[RankedSuggestion] = []
        invalid: list[str] = []
        seen: set[str] = set()

        for suggestion in suggestions:
            name = suggestion.model_name
            descriptor = catalog.get(name)
            if descriptor is None or descriptor.task_type != task_type:
                invalid.append(name)
                continue
            if name in seen:
                continue
            seen.add(name)
            accepted.append(suggestion)
            if len(accepted) >= max_models:
                break
        return accepted, invalid


class ModelSelectionOrchestratorAgent:
    """Coordinate selection agents and write model_config.json atomically."""

    def __init__(
        self,
        model_library_root: str | Path,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.catalog_agent = ModelLibraryCatalogAgent(model_library_root)
        self.profile_agent = DatasetProfileAgent()
        self.deterministic_agent = DeterministicRankingAgent()
        self.validation_agent = ModelSelectionValidationAgent()
        self.llm_agent = LLMModelRankingAgent(llm_client) if llm_client else None

    def run(
        self,
        metadata_path: str | Path,
        feature_selection_path: str | Path,
        mini_data_path: str | Path | None,
        output_path: str | Path,
        max_models: int = 5,
        report_path: str | Path | None = None,
        excluded_model_names: list[str] | None = None,
    ) -> list[ModelCandidate]:
        if not 1 <= max_models <= MAX_SELECTABLE_MODELS:
            raise ValueError(f"max_models must be between 1 and {MAX_SELECTABLE_MODELS}")

        excluded_set = set(excluded_model_names or [])

        metadata = MetadataInput.model_validate_json(
            Path(metadata_path).read_text(encoding="utf-8")
        )
        feature_selection = FeatureSelectionInput.model_validate_json(
            Path(feature_selection_path).read_text(encoding="utf-8")
        )
        profile = self.profile_agent.run(metadata, feature_selection, mini_data_path)

        catalog = self.catalog_agent.run()
        task_descriptors = [
            descriptor
            for descriptor in catalog.values()
            if descriptor.task_type == profile.task_type
            and descriptor.model_name not in excluded_set
        ]
        if not task_descriptors:
            raise UnsupportedProblemTypeError(
                f"No {profile.task_type} model is exposed by MLKit.MODEL_REGISTRY"
            )

        fallback = self.deterministic_agent.run(
            profile, task_descriptors, max_models=max_models
        )
        suggestions = fallback
        selection_mode = "deterministic"
        warnings: list[str] = []
        rejected_llm_models: list[str] = []

        if self.llm_agent is not None:
            try:
                llm_suggestions = self.llm_agent.run(
                    profile, task_descriptors, max_models=max_models
                )
                valid_llm, invalid_llm = self.validation_agent.run(
                    llm_suggestions,
                    catalog,
                    profile.task_type,
                    max_models,
                )
                suggestions = valid_llm + [
                    item
                    for item in fallback
                    if item.model_name not in {entry.model_name for entry in valid_llm}
                ]
                suggestions = suggestions[:max_models]
                selection_mode = (
                    "llm" if len(valid_llm) == len(suggestions) else "llm_with_fallback"
                )
                rejected_llm_models = invalid_llm
                if invalid_llm:
                    warnings.append(
                        "Rejected LLM model names not present in the MLKit registry or wrong for task: "
                        + ", ".join(invalid_llm)
                    )
            except (ValueError, json.JSONDecodeError, ValidationError, RuntimeError) as exc:
                logger.warning("LLM ranking failed; using deterministic fallback: %s", exc)
                warnings.append(f"LLM ranking failed; deterministic fallback used: {exc}")
                selection_mode = "deterministic"

        accepted, invalid = self.validation_agent.run(
            suggestions, catalog, profile.task_type, max_models
        )
        if invalid:
            warnings.append(
                "Rejected final invalid model names: " + ", ".join(invalid)
            )
        if not accepted:
            raise UnsupportedProblemTypeError(
                "No valid model remained after validating against MLKit.MODEL_REGISTRY"
            )

        candidates: list[ModelCandidate] = []
        for priority, suggestion in enumerate(accepted, start=1):
            descriptor = catalog[suggestion.model_name]
            candidates.append(
                ModelCandidate(
                    name=descriptor.model_name,
                    model_name=descriptor.model_name,
                    task_type=descriptor.task_type,
                    priority=priority,
                    rationale=suggestion.rationale,
                    selection_score=suggestion.score,
                    default_hyperparameters=descriptor.default_hyperparameters,
                    hp_space={},
                )
            )

        self._atomic_write(output_path, [item.model_dump(mode="json") for item in candidates])

        report = SelectionReport(
            task_type=profile.task_type,
            data_format=profile.data_format,
            selected_count=len(candidates),
            available_model_count=len(task_descriptors),
            selection_mode=selection_mode,
            invalid_llm_models=rejected_llm_models,
            warnings=warnings,
        )
        if report_path:
            self._atomic_write(report_path, report.model_dump(mode="json"))
        return candidates

    @staticmethod
    def _atomic_write(path: str | Path, payload: Any) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
