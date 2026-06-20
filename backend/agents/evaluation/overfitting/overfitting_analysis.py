import argparse
import configparser
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# Bootstrap sys.path so the shared model_library package resolves regardless of
# cwd, without hardcoding any path. model_library lives at the repo root:
# overfitting -> evaluation -> agents -> backend -> repo root (parents[4]).
_MODEL_LIBRARY_ROOT = str(Path(__file__).resolve().parents[4] / "model_library")
if _MODEL_LIBRARY_ROOT not in sys.path:
    sys.path.insert(0, _MODEL_LIBRARY_ROOT)

from core.data_bundle import CommonData, DataBundle
from core.validators import validate_model_name
from metrics.evaluators import MetricsResult, compute_metrics
from ml_kit import MLKit


logger = logging.getLogger(__name__)

TASK_TYPE_CLASSIFICATION = "classification"
TASK_TYPE_REGRESSION = "regression"
VALID_TASK_TYPES = [TASK_TYPE_CLASSIFICATION, TASK_TYPE_REGRESSION]

# Classifier model names — suffix "Classifier" or known non-regressor names.
# Derived from the MODEL_REGISTRY comment block in ml_kit.py (30 classifiers).
CLASSIFIER_MODEL_NAMES = {
    "LogisticRegression", "RidgeClassifier", "SGDClassifier", "SVC", "LinearSVC",
    "NuSVC", "DecisionTreeClassifier", "RandomForestClassifier", "ExtraTreesClassifier",
    "GradientBoostingClassifier", "AdaBoostClassifier", "HistGradientBoostingClassifier",
    "KNeighborsClassifier", "RadiusNeighborsClassifier", "GaussianNB", "MultinomialNB",
    "ComplementNB", "BernoulliNB", "CategoricalNB", "MLPClassifier",
    "PassiveAggressiveClassifier", "QuadraticDiscriminantAnalysis",
    "LinearDiscriminantAnalysis", "BaggingClassifier", "DummyClassifier",
    "NearestCentroid", "CalibratedClassifierCV", "XGBClassifier",
    "PyTorchFCNNClassifier", "PyTorchCNNClassifier",
}

REGRESSOR_MODEL_NAMES = {
    "LinearRegression", "Ridge", "Lasso", "ElasticNet", "Lars", "LassoLars",
    "OrthogonalMatchingPursuit", "BayesianRidge", "ARDRegression", "SGDRegressor",
    "PassiveAggressiveRegressor", "SVR", "NuSVR", "LinearSVR",
    "KNeighborsRegressor", "RadiusNeighborsRegressor", "DecisionTreeRegressor",
    "RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor",
    "AdaBoostRegressor", "HistGradientBoostingRegressor", "MLPRegressor",
    "BaggingRegressor", "XGBRegressor", "PyTorchFCNNRegressor",
    "PyTorchCNNRegressor", "DummyRegressor", "HuberRegressor", "TheilSenRegressor",
}

MODEL_TYPE_REGISTRY = {
    TASK_TYPE_CLASSIFICATION: CLASSIFIER_MODEL_NAMES,
    TASK_TYPE_REGRESSION: REGRESSOR_MODEL_NAMES,
}

# Direction-aware gap functions — avoids if-else for metric direction (CLAUDE.md #23).
DIRECTION_GAP_FUNCS = {
    "higher_is_better": lambda train_score, test_score: train_score - test_score,
    "lower_is_better": lambda train_score, test_score: test_score - train_score,
}

# Metric fields on MetricsResult per task type — used for gap iteration.
CLASSIFICATION_METRIC_FIELDS = [
    "accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"
]
REGRESSION_METRIC_FIELDS = ["mse", "rmse", "mae", "r2"]

TASK_METRIC_FIELDS = {
    TASK_TYPE_CLASSIFICATION: CLASSIFICATION_METRIC_FIELDS,
    TASK_TYPE_REGRESSION: REGRESSION_METRIC_FIELDS,
}


def _load_config(ini_path: str) -> dict:
    """Read config.ini to locate config.yaml, then return the parsed overfitting config dict."""
    ini_parser = configparser.ConfigParser()
    ini_parser.read(ini_path)
    config_yaml_relative = ini_parser.get("paths", "config_yaml")
    project_root = os.path.dirname(ini_path.replace("config/config.ini", ""))
    # Resolve yaml path relative to the tool directory (parent of config/).
    tool_root = os.path.normpath(os.path.join(os.path.dirname(ini_path), ".."))
    config_yaml_path = os.path.join(tool_root, config_yaml_relative)
    with open(config_yaml_path, "r") as yaml_file:
        raw = yaml.safe_load(yaml_file)
    return raw["overfitting"]


@dataclass
class KFoldResult:
    """Typed container for K-fold cross validation output."""

    k: int
    scoring: str
    per_fold_scores: list
    mean: float
    std: float
    train_vs_cv_gap: Optional[float]


class OverfittingAnalyzer:
    """Analyzes whether a given ML model has overfit by computing holdout gaps and K-fold CV."""

    def __init__(self, input_json_path: str, output_dir: str, verbose: bool) -> None:
        self.output_dir = output_dir
        self.verbose = verbose

        ini_path = os.path.join(os.path.dirname(__file__), "config", "config.ini")
        self.cfg = _load_config(ini_path)

        self.metric_direction_map: dict = self.cfg["metric_direction_map"]
        self.gap_threshold: float = float(self.cfg["gap_threshold"])
        self.epsilon: float = float(self.cfg["epsilon"])

        with open(input_json_path, "r") as json_file:
            self.input_data = json.load(json_file)

        self._validate_input()

        self.model_type: str = self.input_data["model_type"]
        self.model_name: str = self.input_data["model_name"]
        self.dataset_path: str = self.input_data["dataset_path"]
        # Optional separate test CSV / target column for CSV-format datasets.
        self.test_dataset_path: Optional[str] = self.input_data.get("test_dataset_path")
        self.target_column: Optional[str] = self.input_data.get("target_column")
        self.precomputed_train_metrics: Optional[dict] = self.input_data.get("train_metrics") or None
        self.precomputed_test_metrics: Optional[dict] = self.input_data.get("test_metrics") or None

        task_cfg_key = self.model_type
        self.task_cfg: dict = self.cfg[task_cfg_key]
        self.primary_metric: str = self.task_cfg["primary_metric"]
        self.scoring_metric: str = self.task_cfg["scoring_metric"]

        logger.info("=> OverfittingAnalyzer initialized for model=%s type=%s", self.model_name, self.model_type)

    def _validate_input(self) -> None:
        """Validate required input fields and model_name/model_type consistency."""
        required_keys = ["model_type", "model_name", "dataset_path"]
        for key in required_keys:
            if key not in self.input_data or not self.input_data[key]:
                raise ValueError(f"Input JSON missing required field: '{key}'")

        model_type = self.input_data["model_type"]
        if model_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"Invalid model_type '{model_type}'. Must be one of: {VALID_TASK_TYPES}"
            )

        model_name = self.input_data["model_name"]
        # Reuse MLKit's validate_model_name for spelling-aware error messages.
        validate_model_name(model_name)

        expected_names = MODEL_TYPE_REGISTRY[model_type]
        if model_name not in expected_names:
            raise ValueError(
                f"model_name '{model_name}' is not a {model_type} model. "
                f"Check that model_type matches the model family."
            )

    def load_dataset(self, dataset_path: str) -> CommonData:
        """Load dataset from a .npz or CSV file pair.

        For .npz: expects X_train, y_train, X_test, y_test arrays.
        For .csv: loads as pandas DataFrame, splits X/y by target_column, and
        uses self.test_dataset_path for the test split when provided.
        """
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: '{dataset_path}'")

        file_extension = Path(dataset_path).suffix.lower()

        if file_extension == ".npz":
            return self._load_npz_dataset(dataset_path)
        return self._load_csv_dataset(dataset_path)

    def _load_npz_dataset(self, dataset_path: str) -> CommonData:
        """Load a .npz archive with X_train, y_train, X_test, y_test arrays."""
        npz = np.load(dataset_path, allow_pickle=True)
        required_arrays = ["X_train", "y_train", "X_test", "y_test"]
        missing = [arr_name for arr_name in required_arrays if arr_name not in npz]
        if missing:
            raise ValueError(
                f"Dataset .npz is missing required arrays: {missing}. "
                f"Found: {list(npz.keys())}"
            )
        common = CommonData(
            X_train=npz["X_train"].astype(np.float32),
            y_train=npz["y_train"],
            X_test=npz["X_test"].astype(np.float32),
            y_test=npz["y_test"],
        )
        logger.debug(
            "=> Loaded npz dataset: X_train=%s y_train=%s X_test=%s y_test=%s",
            common.X_train.shape, common.y_train.shape,
            common.X_test.shape, common.y_test.shape,
        )
        return common

    def _load_csv_dataset(self, train_csv_path: str) -> CommonData:
        """Load train (and optionally test) CSV files into CommonData arrays.

        Categorical columns are label-encoded to float32. When no separate test
        CSV is provided a stratified 80/20 split of the training data is used.
        """
        if not self.target_column:
            raise ValueError(
                "target_column must be set in the input JSON to load CSV datasets."
            )

        train_df = pd.read_csv(train_csv_path)

        if self.test_dataset_path and os.path.isfile(self.test_dataset_path):
            test_df = pd.read_csv(self.test_dataset_path)
        else:
            # Fall back to an internal 80/20 stratified split.
            logger.warning(
                "=> No test CSV provided; using 80/20 split of train data for overfitting."
            )
            stratify_col = train_df[self.target_column] if self.model_type == TASK_TYPE_CLASSIFICATION else None
            train_df, test_df = train_test_split(
                train_df, test_size=0.2, random_state=42, stratify=stratify_col
            )

        def encode_dataframe(dataframe: pd.DataFrame, target_col: str) -> tuple:
            """Encode a dataframe to float32 X array and integer/float y array."""
            feature_df = dataframe.drop(columns=[target_col]).copy()
            target_series = dataframe[target_col].copy()

            # Encode categorical feature columns with LabelEncoder.
            for col in feature_df.select_dtypes(include=["object", "category"]).columns:
                feature_df[col] = LabelEncoder().fit_transform(feature_df[col].astype(str))

            # Encode the target column if it is not numeric.
            if target_series.dtype == object or hasattr(target_series, "cat"):
                target_series = LabelEncoder().fit_transform(target_series.astype(str))
            else:
                target_series = target_series.values

            return feature_df.values.astype(np.float32), target_series

        X_train, y_train = encode_dataframe(train_df, self.target_column)
        X_test, y_test = encode_dataframe(test_df, self.target_column)

        common = CommonData(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
        logger.debug(
            "=> Loaded csv dataset: X_train=%s y_train=%s X_test=%s y_test=%s",
            common.X_train.shape, common.y_train.shape,
            common.X_test.shape, common.y_test.shape,
        )
        return common

    def _metrics_result_from_dict(self, metrics_dict: dict, split_label: str) -> MetricsResult:
        """Reconstruct a MetricsResult from a precomputed metrics dict."""
        # Provide None for any missing fields so MetricsResult is fully populated.
        all_fields = CLASSIFICATION_METRIC_FIELDS + REGRESSION_METRIC_FIELDS
        filled = {field: metrics_dict.get(field, None) for field in all_fields}
        return MetricsResult(
            task_type=self.model_type,
            model_name=f"{self.model_name}_{split_label}",
            accuracy=filled.get("accuracy"),
            f1_macro=filled.get("f1_macro"),
            f1_weighted=filled.get("f1_weighted"),
            precision_macro=filled.get("precision_macro"),
            recall_macro=filled.get("recall_macro"),
            mse=filled.get("mse"),
            rmse=filled.get("rmse"),
            mae=filled.get("mae"),
            r2=filled.get("r2"),
        )

    def compute_holdout_metrics(
        self, common: CommonData
    ) -> tuple[Optional[MetricsResult], Optional[MetricsResult]]:
        """Compute or reuse holdout train/test metrics.

        When precomputed metrics are provided in the input JSON, they are used as-is.
        Otherwise one MLKit train pass is performed and metrics are computed from predictions.
        """
        if self.precomputed_train_metrics and self.precomputed_test_metrics:
            logger.info("=> Using precomputed train/test metrics from input JSON.")
            train_result = self._metrics_result_from_dict(self.precomputed_train_metrics, "train")
            test_result = self._metrics_result_from_dict(self.precomputed_test_metrics, "test")
            return train_result, test_result

        logger.info("=> Training model to compute holdout metrics via MLKit.")
        data_bundle = DataBundle(common=common)
        kit = MLKit(model_name=self.model_name, data=data_bundle)
        kit.train()

        test_predictions = kit.test()
        # Access train predictions directly via the underlying model wrapper.
        train_predictions = kit.model.predict(common.X_train)

        train_result = compute_metrics(
            y_true=common.y_train,
            y_pred=train_predictions,
            task_type=self.model_type,
            model_name=f"{self.model_name}_train",
        )
        test_result = compute_metrics(
            y_true=common.y_test,
            y_pred=test_predictions,
            task_type=self.model_type,
            model_name=f"{self.model_name}_test",
        )
        logger.info("=> Train metrics:\n%s", train_result)
        logger.info("=> Test metrics:\n%s", test_result)
        return train_result, test_result

    def _assemble_cv_data(self, common: CommonData) -> tuple[np.ndarray, np.ndarray]:
        """Return (X_cv, y_cv) based on the configured cv_data_source."""
        cv_source = self.cfg.get("cv_data_source", "train_test_concat")
        cv_source_map = {
            "train_test_concat": lambda c: (
                np.concatenate([c.X_train, c.X_test], axis=0),
                np.concatenate([c.y_train, c.y_test], axis=0),
            ),
            "train_only": lambda c: (c.X_train, c.y_train),
            "test_only": lambda c: (c.X_test, c.y_test),
        }
        if cv_source not in cv_source_map:
            raise ValueError(
                f"Invalid cv_data_source '{cv_source}'. "
                f"Must be one of: {list(cv_source_map.keys())}"
            )
        return cv_source_map[cv_source](common)

    def run_kfold(
        self, common: CommonData, train_metric_result: Optional[MetricsResult]
    ) -> tuple[Optional[KFoldResult], Optional[str]]:
        """Run K-fold cross validation, retraining a fresh MLKit on each fold.

        Returns (KFoldResult, None) on success or (None, skip_reason) on failure.
        """
        num_folds = int(self.cfg["k_folds"])
        shuffle = bool(self.cfg["shuffle"])
        random_state = int(self.cfg["random_state"])
        use_stratified = bool(self.task_cfg.get("stratified", False))

        cv_X, cv_y = self._assemble_cv_data(common)
        num_samples = cv_X.shape[0]

        if num_samples < num_folds:
            skip_reason = (
                f"cv_data_source yielded only {num_samples} samples, "
                f"which is less than k_folds={num_folds}."
            )
            logger.warning("=> K-fold skipped: %s", skip_reason)
            return None, skip_reason

        if use_stratified:
            unique_classes = np.unique(cv_y)
            min_class_count = int(np.min(np.bincount(cv_y.astype(int))))
            if min_class_count < num_folds:
                skip_reason = (
                    f"Minimum class count ({min_class_count}) is less than "
                    f"k_folds={num_folds}. Reduce k_folds or use more data."
                )
                logger.warning("=> K-fold skipped: %s", skip_reason)
                return None, skip_reason
            splitter = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state)
        else:
            splitter = KFold(n_splits=num_folds, shuffle=shuffle, random_state=random_state)

        fold_scores = []
        for fold_index, (train_indices, val_indices) in enumerate(splitter.split(cv_X, cv_y)):
            fold_common = CommonData(
                X_train=cv_X[train_indices],
                y_train=cv_y[train_indices],
                X_test=cv_X[val_indices],
                y_test=cv_y[val_indices],
            )
            fold_bundle = DataBundle(common=fold_common)
            fold_kit = MLKit(model_name=self.model_name, data=fold_bundle)
            fold_kit.train()
            fold_predictions = fold_kit.test()
            fold_metrics = compute_metrics(
                y_true=fold_common.y_test,
                y_pred=fold_predictions,
                task_type=self.model_type,
                model_name=f"{self.model_name}_fold{fold_index}",
            )
            fold_score = getattr(fold_metrics, self.scoring_metric)
            fold_scores.append(float(fold_score))
            logger.debug(
                "=> Fold %d/%d: %s=%.4f", fold_index + 1, num_folds, self.scoring_metric, fold_score
            )

        cv_mean = float(np.mean(fold_scores))
        cv_std = float(np.std(fold_scores))

        # train_vs_cv_gap: direction-aware gap between holdout train score and CV mean.
        train_vs_cv_gap = None
        if train_metric_result is not None:
            train_score_on_primary = getattr(train_metric_result, self.scoring_metric, None)
            if train_score_on_primary is not None:
                direction = self.metric_direction_map.get(self.scoring_metric, "higher_is_better")
                gap_func = DIRECTION_GAP_FUNCS[direction]
                train_vs_cv_gap = float(gap_func(float(train_score_on_primary), cv_mean))

        kfold_result = KFoldResult(
            k=num_folds,
            scoring=self.scoring_metric,
            per_fold_scores=fold_scores,
            mean=cv_mean,
            std=cv_std,
            train_vs_cv_gap=train_vs_cv_gap,
        )
        logger.info(
            "=> K-fold complete: mean=%s=%.4f std=%.4f train_vs_cv_gap=%s",
            self.scoring_metric, cv_mean, cv_std,
            f"{train_vs_cv_gap:.4f}" if train_vs_cv_gap is not None else "N/A",
        )
        return kfold_result, None

    def compute_gaps(
        self, train_result: MetricsResult, test_result: MetricsResult
    ) -> tuple[dict, Optional[float]]:
        """Compute direction-aware per-metric gaps and relative RMSE gap (regression only)."""
        metric_fields = TASK_METRIC_FIELDS[self.model_type]
        gaps = {}
        for metric_name in metric_fields:
            train_value = getattr(train_result, metric_name, None)
            test_value = getattr(test_result, metric_name, None)
            if train_value is None or test_value is None:
                continue
            direction = self.metric_direction_map.get(metric_name, "higher_is_better")
            gap_func = DIRECTION_GAP_FUNCS[direction]
            gaps[metric_name] = float(gap_func(float(train_value), float(test_value)))

        rel_rmse_gap = None
        if self.model_type == TASK_TYPE_REGRESSION and train_result.rmse is not None and test_result.rmse is not None:
            rel_rmse_gap = float(
                (float(test_result.rmse) - float(train_result.rmse))
                / max(float(train_result.rmse), self.epsilon)
            )

        return gaps, rel_rmse_gap

    def decide_verdict(
        self, gaps: dict, kfold_result: Optional[KFoldResult]
    ) -> bool:
        """Return True if the model is considered overfit based on configured threshold."""
        if self.primary_metric in gaps:
            primary_gap = gaps[self.primary_metric]
            is_overfit = primary_gap > self.gap_threshold
            logger.info(
                "=> Verdict: primary_metric=%s gap=%.4f threshold=%.4f is_overfitted=%s",
                self.primary_metric, primary_gap, self.gap_threshold, is_overfit,
            )
            return is_overfit

        # Fallback: use train_vs_cv_gap when holdout primary metric is unavailable.
        if kfold_result is not None and kfold_result.train_vs_cv_gap is not None:
            is_overfit = kfold_result.train_vs_cv_gap > self.gap_threshold
            logger.info(
                "=> Verdict (fallback train_vs_cv_gap): gap=%.4f threshold=%.4f is_overfitted=%s",
                kfold_result.train_vs_cv_gap, self.gap_threshold, is_overfit,
            )
            return is_overfit

        logger.warning("=> Cannot determine verdict: no primary metric gap or CV gap available.")
        return False

    def _metrics_result_to_dict(self, result: Optional[MetricsResult]) -> Optional[dict]:
        """Convert MetricsResult to a plain dict, dropping None-valued fields."""
        if result is None:
            return None
        raw = asdict(result)
        return {key: value for key, value in raw.items() if value is not None}

    def write_output(self, payload: dict) -> str:
        """Write the analysis result JSON to output_dir/overfitting_analysis.json."""
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "overfitting_analysis.json")
        with open(output_path, "w") as output_file:
            json.dump(payload, output_file, indent=2)
        logger.info("=> Output written to: %s", output_path)
        return output_path

    def run(self) -> dict:
        """Orchestrate the full analysis: load data, compute metrics, CV, verdict, write output."""
        common = self.load_dataset(self.dataset_path)
        train_result, test_result = self.compute_holdout_metrics(common)
        kfold_result, cv_skipped_reason = self.run_kfold(common, train_result)

        gaps, rel_rmse_gap = self.compute_gaps(train_result, test_result)
        is_overfitted = self.decide_verdict(gaps, kfold_result)

        payload = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "is_overfitted": is_overfitted,
            "gap_threshold": self.gap_threshold,
            "primary_metric": self.primary_metric,
            "gaps": gaps,
            "rel_rmse_gap": rel_rmse_gap,
            "train_metrics": self._metrics_result_to_dict(train_result),
            "test_metrics": self._metrics_result_to_dict(test_result),
            "k_fold_cross_validation_results": (
                asdict(kfold_result) if kfold_result is not None else None
            ),
            "cv_skipped_reason": cv_skipped_reason,
        }

        self.write_output(payload)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overfitting Analysis Tool — computes metric gaps and K-fold CV for an ML model."
    )
    parser.add_argument(
        "-i", "--input_json",
        required=True,
        help="Path to the input JSON file (model_name, model_type, dataset_path).",
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        help="Directory where overfitting_analysis.json will be written.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    parsed_args = parser.parse_args()

    log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s => %(message)s",
    )

    analyzer = OverfittingAnalyzer(
        input_json_path=parsed_args.input_json,
        output_dir=parsed_args.output_dir,
        verbose=parsed_args.verbose,
    )
    result = analyzer.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
