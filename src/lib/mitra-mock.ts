// Mock data for MITRA v2 product UI (no backend wired yet).

export type AgentId =
  | "data_validator"
  | "metadata_gen"
  | "feature_selection"
  | "model_selection"
  | "classification"
  | "regression_usl"
  | "judge"
  | "hpt";

export type AgentStatus = "idle" | "running" | "ok" | "warn" | "error";

export interface AgentSpec {
  id: AgentId;
  name: string;
  owner: string;
  stage: number;
  description: string;
  reads: string[];
  writes: string[];
}

export const AGENTS: AgentSpec[] = [
  {
    id: "data_validator",
    name: "Data Validator",
    owner: "Eng 1",
    stage: 1,
    description: "Fails fast on bad data: nulls > 80%, zero-variance cols, wrong file format.",
    reads: ["raw_dataset"],
    writes: ["validation_report.json"],
  },
  {
    id: "metadata_gen",
    name: "Metadata Gen",
    owner: "Eng 2",
    stage: 2,
    description: "Profiles dataset: dtypes, cardinality, target distribution, class balance.",
    reads: ["raw_dataset", "validation_report.json"],
    writes: ["metadata.json"],
  },
  {
    id: "feature_selection",
    name: "Feature Selection",
    owner: "Eng 3",
    stage: 3,
    description: "Mutual information + correlation pruning. Emits kept/dropped features.",
    reads: ["metadata.json"],
    writes: ["features.json"],
  },
  {
    id: "model_selection",
    name: "Model Selection",
    owner: "Eng 4",
    stage: 4,
    description: "LLM routes to family agent based on metadata + task hint.",
    reads: ["metadata.json", "features.json"],
    writes: ["routing.json"],
  },
  {
    id: "classification",
    name: "Classification",
    owner: "Eng 5",
    stage: 5,
    description: "XGBoost, LightGBM, RandomForest from template library.",
    reads: ["routing.json", "features.json"],
    writes: ["candidates/*.pkl", "metrics.json"],
  },
  {
    id: "regression_usl",
    name: "Regression / USL",
    owner: "Eng 6",
    stage: 5,
    description: "K-Means / DBSCAN / IsolationForest or regressors.",
    reads: ["routing.json", "features.json"],
    writes: ["candidates/*.pkl", "metrics.json"],
  },
  {
    id: "judge",
    name: "Judge",
    owner: "Eng 7",
    stage: 6,
    description: "Decides accept / retry / escalate. Gap & floor thresholds, SHAP guard.",
    reads: ["metrics.json"],
    writes: ["verdict.json"],
  },
  {
    id: "hpt",
    name: "Hyperparameter Tuner",
    owner: "Eng 8",
    stage: 7,
    description: "Optuna sweep over hp_space.yaml from template library.",
    reads: ["verdict.json", "templates/*"],
    writes: ["best_params.json", "final_model.pkl"],
  },
];

export type EventLevel = "info" | "ok" | "warn" | "error";
export interface PipelineEvent {
  t: string;
  agent: AgentId | "orchestrator";
  level: EventLevel;
  msg: string;
}

export const SAMPLE_EVENTS: PipelineEvent[] = [
  { t: "00:00.01", agent: "orchestrator", level: "info", msg: "session_id=mit_8f2a · dataset=churn_q3.csv (24.1 MB)" },
  { t: "00:00.42", agent: "data_validator", level: "info", msg: "scanning 84 columns, 412,930 rows" },
  { t: "00:01.18", agent: "data_validator", level: "warn", msg: "column `legacy_score` 91% null — dropped" },
  { t: "00:01.91", agent: "data_validator", level: "ok", msg: "validation passed → validation_report.json" },
  { t: "00:02.10", agent: "metadata_gen", level: "info", msg: "profiling dtypes & cardinality" },
  { t: "00:03.84", agent: "metadata_gen", level: "ok", msg: "target=churned · binary · balance 0.78/0.22" },
  { t: "00:04.02", agent: "feature_selection", level: "info", msg: "MI scoring 83 features" },
  { t: "00:06.55", agent: "feature_selection", level: "ok", msg: "kept 41 · dropped 42 (corr>0.92 or MI<0.01)" },
  { t: "00:06.71", agent: "model_selection", level: "info", msg: "LLM routing → classification family" },
  { t: "00:07.02", agent: "classification", level: "info", msg: "spawning XGBoost · LightGBM · RandomForest on Ray" },
  { t: "00:12.44", agent: "classification", level: "ok", msg: "XGBoost val_auc=0.912 train_auc=0.934" },
  { t: "00:12.61", agent: "classification", level: "ok", msg: "LightGBM val_auc=0.908 train_auc=0.921" },
  { t: "00:13.02", agent: "classification", level: "ok", msg: "RandomForest val_auc=0.871 train_auc=0.998" },
  { t: "00:13.18", agent: "judge", level: "warn", msg: "RandomForest overfit gap=0.127 > 0.08 → exclude" },
  { t: "00:13.34", agent: "judge", level: "ok", msg: "accept XGBoost · gap=0.022 ≤ 0.08 · floor=0.85" },
  { t: "00:13.48", agent: "hpt", level: "info", msg: "Optuna sweep · 40 trials · search space hp_space.yaml" },
  { t: "00:41.92", agent: "hpt", level: "ok", msg: "best trial #27 val_auc=0.927 · final_model.pkl written" },
];

export interface LeaderboardRow {
  rank: number;
  model: string;
  val: number;
  train: number;
  gap: number;
  status: "accepted" | "rejected" | "tuning";
  notes: string;
}

export const LEADERBOARD: LeaderboardRow[] = [
  { rank: 1, model: "XGBoost (tuned)", val: 0.927, train: 0.945, gap: 0.018, status: "accepted", notes: "Final · Optuna trial #27" },
  { rank: 2, model: "XGBoost", val: 0.912, train: 0.934, gap: 0.022, status: "tuning", notes: "Baseline · selected for HPT" },
  { rank: 3, model: "LightGBM", val: 0.908, train: 0.921, gap: 0.013, status: "accepted", notes: "Within floor & gap" },
  { rank: 4, model: "RandomForest", val: 0.871, train: 0.998, gap: 0.127, status: "rejected", notes: "Overfit gap > 0.08" },
];

export interface TemplateSpec {
  family: string;
  model: string;
  files: string[];
  description: string;
}

export const TEMPLATES: TemplateSpec[] = [
  { family: "classification", model: "xgboost", files: ["train.py.j2", "hp_space.yaml", "resources.yaml"], description: "Gradient boosted trees · gpu-aware" },
  { family: "classification", model: "lightgbm", files: ["train.py.j2", "hp_space.yaml", "resources.yaml"], description: "Histogram-based GBDT" },
  { family: "classification", model: "random_forest", files: ["train.py.j2", "hp_space.yaml", "resources.yaml"], description: "Bagged decision trees" },
  { family: "regression", model: "xgboost_reg", files: ["train.py.j2", "hp_space.yaml", "resources.yaml"], description: "GBDT regressor · MAE/RMSE" },
  { family: "usl", model: "kmeans", files: ["train.py.j2", "hp_space.yaml", "resources.yaml"], description: "Cluster count via silhouette" },
  { family: "usl", model: "dbscan", files: ["train.py.j2", "hp_space.yaml", "resources.yaml"], description: "Density-based · eps heuristic" },
  { family: "usl", model: "isolation_forest", files: ["train.py.j2", "hp_space.yaml", "resources.yaml"], description: "Anomaly detection" },
];
