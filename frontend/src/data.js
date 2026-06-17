export const AGENTS = [
  {
    id: 'validator',
    name: 'Data Validator',
    short: 'DV',
    hue: 14,
    type: 'Python',
    role: 'Dataset gates and validation report',
    owner: 'P3',
  },
  {
    id: 'metadata',
    name: 'Metadata Gen',
    short: 'MG',
    hue: 262,
    type: 'LLM',
    role: 'Metadata schema contract',
    owner: 'P4',
  },
  {
    id: 'feature',
    name: 'Feature Selection',
    short: 'FS',
    hue: 212,
    type: 'LLM',
    role: 'Feature ranking and pruning',
    owner: 'P5',
  },
  {
    id: 'model',
    name: 'Model Selection',
    short: 'MS',
    hue: 286,
    type: 'LLM',
    role: 'Candidate model families',
    owner: 'P6',
  },
  {
    id: 'classification',
    name: 'Classification',
    short: 'CL',
    hue: 158,
    type: 'LLM',
    role: 'Classifier training plan',
    owner: 'P6',
  },
  {
    id: 'regression',
    name: 'Regression / USL',
    short: 'RU',
    hue: 188,
    type: 'LLM',
    role: 'Regression and unsupervised plan',
    owner: 'P6',
  },
  {
    id: 'judge',
    name: 'Judge',
    short: 'JD',
    hue: 38,
    type: 'LLM',
    role: 'Model verdict and rationale',
    owner: 'P7',
  },
  {
    id: 'hpt',
    name: 'HPT',
    short: 'HP',
    hue: 330,
    type: 'Python',
    role: 'Hyperparameter search budget',
    owner: 'P7',
  },
];

export const STAGES = [
  {
    key: 'validate',
    label: 'Data Validation',
    agent: 'validator',
    artifact: 'validation_report.json',
    sub: 'Schema, nulls, variance, format',
  },
  {
    key: 'metadata',
    label: 'Metadata Generation',
    agent: 'metadata',
    artifact: 'metadata.json',
    sub: 'Problem type, column types, contract',
  },
  {
    key: 'encode',
    label: 'Encoding',
    agent: null,
    artifact: 'data_encoded.csv',
    sub: 'Categorical to numeric maps',
  },
  {
    key: 'scale',
    label: 'Scaling',
    agent: null,
    artifact: 'data_scaled.csv',
    sub: 'Scaler maps and normalized data',
  },
  {
    key: 'feature',
    label: 'Feature Selection',
    agent: 'feature',
    artifact: 'feature_selection.json',
    sub: 'Keep/drop recommendations',
  },
  {
    key: 'model',
    label: 'Model Selection',
    agent: 'model',
    artifact: 'model_config.json',
    sub: 'Candidate families for task',
  },
  {
    key: 'train',
    label: 'Training on Ray',
    agent: 'classification',
    artifact: 'model.pkl',
    sub: 'Parallel candidate fits',
  },
  {
    key: 'eval',
    label: 'Evaluation + SHAP',
    agent: null,
    artifact: 'eval_metrics.json',
    sub: 'Holdout metrics and explanations',
  },
  {
    key: 'judge',
    label: 'Judge Deliberation',
    agent: 'judge',
    artifact: 'verdict.json',
    sub: 'Gap, floor, generalization',
  },
  {
    key: 'hpt',
    label: 'HPT Loop',
    agent: 'hpt',
    artifact: 'new_hp.json',
    sub: 'Five guided trials and retrain',
  },
];

export const STAGE_LOGS = {
  validate: [
    ['info', 'Received dataset and started validation'],
    ['info', 'Sniffing delimiter and encoding'],
    ['info', 'Null density and variance checks complete'],
    ['ok', 'validation_report.json written'],
  ],
  metadata: [
    ['info', 'Reading mini_data.csv'],
    ['llm', 'Inferring problem type and column roles'],
    ['ok', 'metadata.json validated against schema'],
  ],
  encode: [
    ['info', 'Reading data.csv in configured chunks'],
    ['ok', 'data_encoded.csv and encoder map written'],
  ],
  scale: [
    ['info', 'Fitting scaler on numeric columns'],
    ['ok', 'data_scaled.csv and scaler map written'],
  ],
  feature: [
    ['info', 'Scoring candidate features'],
    ['llm', 'Ranking keep/drop recommendations'],
    ['ok', 'feature_selection.json written'],
  ],
  model: [
    ['llm', 'Selecting candidate model families'],
    ['info', 'Queueing training candidates'],
    ['ok', 'model_config.json written'],
  ],
  train: [
    ['info', 'Scheduling model fits'],
    ['ray', 'candidate_001 assigned to worker-1'],
    ['ray', 'candidate_002 assigned to worker-2'],
    ['ok', 'training artifacts written'],
  ],
  eval: [
    ['info', 'Running holdout evaluation'],
    ['info', 'Computing explanation artifacts'],
    ['ok', 'eval_metrics.json written'],
  ],
  judge: [
    ['llm', 'Comparing metrics, gap, and budget'],
    ['ok', 'verdict.json written'],
  ],
  hpt: [
    ['info', 'Starting guided trial budget'],
    ['hpt', 'trial 3/5 improves validation score'],
    ['ok', 'new_hp.json written'],
  ],
};

export const LEADERBOARD = [
  {
    rank: 1,
    model: 'XGBoost',
    family: 'Gradient Boosting',
    acc: 0.973,
    f1: 0.972,
    auc: 0.996,
    gap: 0.018,
    time: 2.4,
    judge: 94.6,
    winner: true,
    hp: 'depth 3, lr 0.06, 220 est',
  },
  {
    rank: 2,
    model: 'LightGBM',
    family: 'Gradient Boosting',
    acc: 0.967,
    f1: 0.965,
    auc: 0.994,
    gap: 0.022,
    time: 1.9,
    judge: 92.1,
    winner: false,
    hp: 'leaves 31, lr 0.05',
  },
  {
    rank: 3,
    model: 'Random Forest',
    family: 'Bagging',
    acc: 0.96,
    f1: 0.958,
    auc: 0.991,
    gap: 0.031,
    time: 1.2,
    judge: 88.7,
    winner: false,
    hp: '400 trees, max_feat sqrt',
  },
];

export const SHAP = [
  { feature: 'petal_length', value: 0.412 },
  { feature: 'petal_width', value: 0.367 },
  { feature: 'sepal_length', value: 0.142 },
  { feature: 'sepal_width', value: 0.079 },
];

// Stable signature of the LLM settings used to invalidate a stale smoke-test
// result whenever the provider, model, key, or gateway changes.
export function llmConfigKey(settings) {
  return [
    settings.provider || '',
    settings.model || '',
    settings.gatewayUrl || '',
    settings.apiKey || '',
  ].join('|');
}

export const PROVIDERS = [
  { value: 'anthropic', label: 'Anthropic', placeholder: 'anthropic/claude...' },
  { value: 'openai', label: 'OpenAI', placeholder: 'openai/gpt...' },
  { value: 'gemini', label: 'Gemini', placeholder: 'gemini/gemini...' },
];

export const NAV_ITEMS = [
  { key: 'dashboard', label: 'Dashboard', icon: 'grid' },
  { key: 'upload', label: 'New Run', icon: 'upload' },
  { key: 'pipeline', label: 'Training', icon: 'flow' },
  { key: 'leaderboard', label: 'Leaderboard', icon: 'trophy' },
  { key: 'settings', label: 'Settings', icon: 'gear' },
];
