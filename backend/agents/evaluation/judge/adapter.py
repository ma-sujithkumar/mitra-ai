"""
UpstreamAdapter: maps each upstream source into the judge's CandidateModel schema.

The judge never directly parses upstream output formats; all translation lives here.
This keeps the judge decoupled from upstream schema changes.

Upstream sources (one JSON file per model):
  - overfitting_analysis.json  (from epic_4/overfitting_analysis_tool)
  - inference_metrics          (per-model metrics dict)
  - shap_summary               (optional per-model SHAP dict)
  - hyperparam_sensitivity     (optional per-model tuning sensitivity dict)
  - complexity                 (explicit n_params/depth/family_rank dict)
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import (
    CandidateModel,
    ComplexityDescriptor,
    JudgeInput,
    OverfittingInfo,
)

logger = logging.getLogger(__name__)


class UpstreamAdapter:
    """Translates upstream tool outputs into the JudgeInput adapter schema."""

    def adapt_overfitting(self, overfitting_json: Dict[str, Any]) -> OverfittingInfo:
        """Map overfitting_analysis.json => OverfittingInfo.

        The upstream schema (epic_4/overfitting_analysis_tool/SPEC.md Section 6)
        provides is_overfitted, gaps (dict), and k_fold_cross_validation_results.
        We extract the primary_metric gap and train_vs_cv_gap.
        """
        primary_metric = overfitting_json.get("primary_metric", "accuracy")
        gaps = overfitting_json.get("gaps", {})
        primary_gap = gaps.get(primary_metric, 0.0)

        cv_results = overfitting_json.get("k_fold_cross_validation_results") or overfitting_json.get("cv_results") or {}
        train_vs_cv_gap = cv_results.get("train_vs_cv_gap", None)

        logger.debug(
            "=> Adapted overfitting: primary_metric=%s gap=%.4f train_vs_cv_gap=%s",
            primary_metric,
            primary_gap,
            train_vs_cv_gap,
        )
        return OverfittingInfo(
            is_overfitted=bool(overfitting_json.get("is_overfitted", False)),
            gap=float(primary_gap) if primary_gap is not None else 0.0,
            train_vs_cv_gap=float(train_vs_cv_gap) if train_vs_cv_gap is not None else None,
            train_metrics=overfitting_json.get("train_metrics"),
            test_metrics=overfitting_json.get("test_metrics"),
            cv_results=cv_results if cv_results else None,
            diagnostics=overfitting_json.get("diagnostics"),
        )

    def adapt_complexity(self, complexity_dict: Dict[str, Any]) -> ComplexityDescriptor:
        """Map a raw complexity dict => ComplexityDescriptor."""
        return ComplexityDescriptor(
            n_params=int(complexity_dict.get("n_params", 0)),
            depth=int(complexity_dict.get("depth", 0)),
            family_rank=int(complexity_dict.get("family_rank", 1)),
        )

    def adapt_candidate(
        self,
        model_name: str,
        task_type: str,
        metrics: Dict[str, Optional[float]],
        overfitting_json: Dict[str, Any],
        complexity_dict: Dict[str, Any],
        shap_summary: Optional[Dict[str, Any]] = None,
        hyperparam_sensitivity: Optional[Dict[str, Any]] = None,
    ) -> CandidateModel:
        """Build a single CandidateModel from all upstream sources for one model."""
        overfitting_info = self.adapt_overfitting(overfitting_json)
        complexity_descriptor = self.adapt_complexity(complexity_dict)
        logger.debug(
            "=> Adapted candidate: model=%s task=%s", model_name, task_type
        )
        return CandidateModel(
            model_name=model_name,
            task_type=task_type,
            metrics=metrics,
            overfitting=overfitting_info,
            complexity=complexity_descriptor,
            shap_summary=shap_summary,
            hyperparam_sensitivity=hyperparam_sensitivity,
        )

    def adapt_judge_input(
        self,
        candidate_raw_list: List[Dict[str, Any]],
        dataset_id: Optional[str] = None,
        minidata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JudgeInput:
        """Build a full JudgeInput from a list of raw per-model dicts.

        Each element of candidate_raw_list must have:
            model_name, task_type, metrics,
            overfitting_json, complexity,
        And optionally: shap_summary, hyperparam_sensitivity.
        """
        candidates = []
        for raw in candidate_raw_list:
            candidate = self.adapt_candidate(
                model_name=raw["model_name"],
                task_type=raw["task_type"],
                metrics=raw["metrics"],
                overfitting_json=raw["overfitting_json"],
                complexity_dict=raw["complexity"],
                shap_summary=raw.get("shap_summary"),
                hyperparam_sensitivity=raw.get("hyperparam_sensitivity"),
            )
            candidates.append(candidate)

        logger.debug("=> Adapted %d candidates into JudgeInput", len(candidates))
        return JudgeInput(
            dataset_id=dataset_id,
            candidates=candidates,
            minidata=minidata,
            metadata=metadata,
        )

    def build_shap_summary_from_csv(
        self,
        csv_path: str,
        top_n: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """Read a global_feature_importance.csv produced by the SHAP module and build a shap_summary dict.

        Expected CSV columns: feature_name, mean_absolute_shap_value (sorted descending).
        Returns None if the file is missing or empty.
        """
        shap_csv_path = Path(csv_path)
        if not shap_csv_path.exists():
            logger.warning("=> SHAP CSV not found: %s", csv_path)
            return None

        feature_rows: List[Dict[str, Any]] = []
        with open(shap_csv_path, "r", newline="") as csv_file_handle:
            reader = csv.DictReader(csv_file_handle)
            for row in reader:
                feature_rows.append({
                    "feature_name": row["feature_name"],
                    "mean_absolute_shap_value": float(row["mean_absolute_shap_value"]),
                })

        if not feature_rows:
            logger.warning("=> SHAP CSV is empty: %s", csv_path)
            return None

        total_shap = sum(row["mean_absolute_shap_value"] for row in feature_rows)
        if total_shap == 0.0:
            logger.warning("=> SHAP total is zero in: %s", csv_path)
            return None

        top_rows = feature_rows[:top_n]
        top_shap_sum = sum(row["mean_absolute_shap_value"] for row in top_rows)
        feature_concentration = round(top_shap_sum / total_shap, 6)

        top_features = [row["feature_name"] for row in top_rows]
        mean_abs_shap = {
            row["feature_name"]: round(row["mean_absolute_shap_value"], 6)
            for row in top_rows
        }

        logger.debug(
            "=> Built SHAP summary from %s: top_features=%s concentration=%.4f",
            csv_path,
            top_features,
            feature_concentration,
        )
        return {
            "top_features": top_features,
            "mean_abs_shap": mean_abs_shap,
            "feature_concentration": feature_concentration,
        }

    def adapt_from_hpt_results(
        self,
        hpt_json_path: str,
        task_type: str,
        shap_dir: Optional[str] = None,
        top_n_shap: int = 5,
        dataset_id: Optional[str] = None,
        minidata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> JudgeInput:
        """Build a JudgeInput directly from an hpt_results.json file.

        Optionally enriches each candidate with a shap_summary read from shap_dir.
        Looks for per-model CSV at <shap_dir>/<model_name>/csv/global_feature_importance.csv,
        then falls back to a shared <shap_dir>/csv/global_feature_importance.csv.

        Args:
            hpt_json_path: Path to hpt_results.json produced by HyperparameterTuningAgent.
            task_type: 'classification' or 'regression'.
            shap_dir: Optional root of SHAP output directory for this session.
            top_n_shap: Number of top features to include in shap_summary.
            dataset_id: Optional dataset identifier forwarded to JudgeInput.
            minidata: Optional pd.describe() dict forwarded to JudgeInput.
            metadata: Optional pipeline metadata dict forwarded to JudgeInput.
        """
        hpt_path = Path(hpt_json_path)
        if not hpt_path.exists():
            raise FileNotFoundError(f"hpt_results.json not found: {hpt_json_path}")

        with open(hpt_path, "r") as hpt_file:
            hpt_data = json.load(hpt_file)

        hpt_entries: List[Dict[str, Any]] = hpt_data.get("hpt_results", [])
        if not hpt_entries:
            raise ValueError(f"No hpt_results entries found in: {hpt_json_path}")

        # Determine the primary metric key used for overfitting gap lookup
        primary_metric_key = "accuracy" if task_type == "classification" else "r2"

        candidates = []
        for hpt_entry in hpt_entries:
            model_name = hpt_entry.get("name") or hpt_entry.get("model_name", "unknown")

            # Convert HPT overfitting format to the format adapt_overfitting expects
            raw_overfitting = hpt_entry.get("overfitting", {})
            overfitting_json: Dict[str, Any] = {
                "is_overfitted": raw_overfitting.get("is_overfitted", False),
                "primary_metric": primary_metric_key,
                "gaps": {
                    primary_metric_key: raw_overfitting.get("gap", 0.0)
                },
                "train_metrics": hpt_entry.get("train_metrics"),
                "test_metrics": hpt_entry.get("val_metrics"),
            }
            train_vs_cv_gap = raw_overfitting.get("train_vs_cv_gap")
            if train_vs_cv_gap is not None:
                overfitting_json["k_fold_cross_validation_results"] = {
                    "train_vs_cv_gap": train_vs_cv_gap
                }

            # Attach best_params to the sensitivity dict for prompt context
            hyperparam_sensitivity = hpt_entry.get("hyperparam_sensitivity")
            if hyperparam_sensitivity is not None:
                hyperparam_sensitivity = dict(hyperparam_sensitivity)
                best_params = hpt_entry.get("best_hyperparameters", {})
                if best_params:
                    hyperparam_sensitivity["best_params"] = best_params

            # Resolve SHAP CSV for this model
            shap_summary: Optional[Dict[str, Any]] = None
            if shap_dir is not None:
                model_specific_csv = (
                    Path(shap_dir) / model_name / "csv" / "global_feature_importance.csv"
                )
                shared_csv = Path(shap_dir) / "csv" / "global_feature_importance.csv"

                if model_specific_csv.exists():
                    shap_summary = self.build_shap_summary_from_csv(
                        str(model_specific_csv), top_n=top_n_shap
                    )
                elif shared_csv.exists():
                    shap_summary = self.build_shap_summary_from_csv(
                        str(shared_csv), top_n=top_n_shap
                    )
                else:
                    logger.warning(
                        "=> No SHAP CSV found for model %s under shap_dir=%s",
                        model_name,
                        shap_dir,
                    )

            candidate = self.adapt_candidate(
                model_name=model_name,
                task_type=task_type,
                metrics=hpt_entry.get("val_metrics", {}),
                overfitting_json=overfitting_json,
                complexity_dict=hpt_entry.get("complexity", {}),
                shap_summary=shap_summary,
                hyperparam_sensitivity=hyperparam_sensitivity,
            )
            candidates.append(candidate)

        logger.info(
            "=> Adapted %d candidates from HPT results: %s",
            len(candidates),
            hpt_json_path,
        )
        return JudgeInput(
            dataset_id=dataset_id,
            candidates=candidates,
            minidata=minidata,
            metadata=metadata,
        )
