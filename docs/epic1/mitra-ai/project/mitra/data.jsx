/* ============================================================
   MITRA AI — mock data
   ============================================================ */

// ---- the 8 agents (1 per teammate) ----
const AGENTS = [
  { id:'validator',  name:'Data Validator',   short:'DV', hue:14,  type:'Python',
    role:'Gates bad data before any LLM is touched', owner:'P3' },
  { id:'metadata',   name:'Metadata Gen',     short:'MG', hue:262, type:'LLM',
    role:'Builds the metadata.json contract every agent reads', owner:'P4' },
  { id:'feature',    name:'Feature Selection',short:'FS', hue:212, type:'LLM',
    role:'Ranks & prunes features for downstream models', owner:'P5' },
  { id:'model',      name:'Model Selection',  short:'MS', hue:286, type:'LLM',
    role:'Chooses candidate model families for the task', owner:'P6' },
  { id:'classification', name:'Classification', short:'CL', hue:158, type:'LLM',
    role:'Trains & tunes classifier candidates', owner:'P6' },
  { id:'regression', name:'Regression / USL', short:'RU', hue:188, type:'LLM',
    role:'Regression + unsupervised model family', owner:'P6' },
  { id:'judge',      name:'Judge',            short:'JD', hue:38,  type:'LLM',
    role:'Weighs every proposal & converges on a verdict', owner:'P7' },
  { id:'hpt',        name:'HPT',              short:'HP', hue:330, type:'Python',
    role:'Guided hyperparameter search within budget', owner:'P7' },
];
const AGENT = Object.fromEntries(AGENTS.map(a => [a.id, a]));

// ---- pipeline stages (P2 flow), in execution order ----
// kind: 'agent' (owned by an agent) | 'system' (deterministic Python step)
const STAGES = [
  { key:'validate', label:'Data Validation',    agent:'validator', kind:'agent',
    artifact:'validation_report.json', sub:'Schema · nulls · variance · format' },
  { key:'metadata', label:'Metadata Generation',agent:'metadata',  kind:'agent',
    artifact:'metadata.json', sub:'Problem type · col types · contract' },
  { key:'encode',   label:'Encoding',           agent:null,        kind:'system',
    artifact:'data_encoded.csv', sub:'Categorical → numeric · encoder_map' },
  { key:'scale',    label:'Scaling',            agent:null,        kind:'system',
    artifact:'data_scaled.csv', sub:'StandardScaler · scaler_map' },
  { key:'feature',  label:'Feature Selection',  agent:'feature',   kind:'agent',
    artifact:'feature_selection.json', sub:'Chi² · mutual-info · keep/drop' },
  { key:'model',    label:'Model Selection',    agent:'model',     kind:'agent',
    artifact:'model_config.json', sub:'Candidate families for task' },
  { key:'train',    label:'Training on Ray',    agent:'classification', kind:'agent',
    artifact:'model.pkl · train_metrics.json', sub:'Parallel fits across workers' },
  { key:'eval',     label:'Evaluation + SHAP',  agent:null,        kind:'system',
    artifact:'eval_metrics.json · shap_values.npy', sub:'Holdout metrics · explanations' },
  { key:'judge',    label:'Judge Deliberation', agent:'judge',     kind:'agent',
    artifact:'verdict.json', sub:'Gap · floor · generalization' },
  { key:'hpt',      label:'HPT Loop',           agent:'hpt',       kind:'system',
    artifact:'new_hp.json', sub:'Optuna · 5 guided trials · retrain' },
];

// ---- per-stage SSE log lines (streamed during a run) ----
const STAGE_LOGS = {
  validate: [
    ['info','Received iris.csv · 150 rows × 5 cols'],
    ['info','Sniffing delimiter & encoding → utf-8, comma'],
    ['info','Null check: 0 cols exceed 80% threshold'],
    ['info','Zero-variance scan: none found'],
    ['ok','validation_report.json written · PASS'],
  ],
  metadata: [
    ['info','Sampling mini_data.csv (capped 1000 rows)'],
    ['info','describe(include="all") + dtype per column'],
    ['llm','LLM infer problem_type → classification'],
    ['info','output_cols=["species"] · 4 input cols'],
    ['ok','metadata.json validated against JSON Schema'],
  ],
  encode: [
    ['info','Reading data.csv in 50k-row chunks'],
    ['info','species → label encoded (3 classes)'],
    ['ok','data_encoded.csv + encoder_map.json written'],
  ],
  scale: [
    ['info','Fitting StandardScaler on 4 numeric cols'],
    ['ok','data_scaled.csv + scaler_map.json written'],
  ],
  feature: [
    ['info','Chi-square on 4 candidate features'],
    ['llm','Ranking by mutual information'],
    ['info','keep=[petal_len, petal_wid, sepal_len] drop=[sepal_wid]'],
    ['ok','feature_selection.json written'],
  ],
  model: [
    ['llm','Surveying families for 150-row classification'],
    ['info','Candidates: XGBoost · LightGBM · RandomForest · SVM · LogReg · KNN'],
    ['ok','model_config.json · 6 candidates queued'],
  ],
  train: [
    ['info','Ray head up · 4 workers · scheduling 6 fits'],
    ['ray','model_001 XGBoost → worker-2'],
    ['ray','model_002 LightGBM → worker-1'],
    ['ray','model_003 RandomForest → worker-3'],
    ['ray','model_004 SVM(RBF) → worker-4'],
    ['ray','model_005 LogReg → worker-2'],
    ['ray','model_006 KNN → worker-1'],
    ['ok','6 models trained · train_metrics.json ×6'],
  ],
  eval: [
    ['info','Holdout eval on 30-row test split'],
    ['info','Computing SHAP values for top candidates'],
    ['ok','eval_metrics.json + shap_values.npy written'],
  ],
  judge: [
    ['llm','Deliberation loop · weighing 6 proposals'],
    ['info','gap(train−test) ≤ 0.04 for 4 candidates'],
    ['info','floor: accuracy ≥ 0.90 satisfied by 5'],
    ['llm','XGBoost: best F1 + lowest overfit gap'],
    ['ok','verdict.json → winner = XGBoost (score 94.6)'],
  ],
  hpt: [
    ['info','Optuna study · budget 5 trials'],
    ['hpt','trial 1/5 max_depth=4 lr=0.10 → 0.967'],
    ['hpt','trial 3/5 max_depth=3 lr=0.08 → 0.973'],
    ['hpt','trial 5/5 max_depth=3 lr=0.06 → 0.973'],
    ['ok','new_hp.json · retrained winner on Ray'],
  ],
};

// ---- leaderboard (iris classification result) ----
const LEADERBOARD = [
  { rank:1, model:'XGBoost',            family:'Gradient Boosting', acc:0.973, f1:0.972, auc:0.996, gap:0.018, time:2.4, judge:94.6, winner:true,  hp:'depth 3 · lr 0.06 · 220 est' },
  { rank:2, model:'LightGBM',           family:'Gradient Boosting', acc:0.967, f1:0.965, auc:0.994, gap:0.022, time:1.9, judge:92.1, winner:false, hp:'leaves 31 · lr 0.05' },
  { rank:3, model:'Random Forest',      family:'Bagging',           acc:0.960, f1:0.958, auc:0.991, gap:0.031, time:1.2, judge:88.7, winner:false, hp:'400 trees · max_feat sqrt' },
  { rank:4, model:'SVM (RBF)',          family:'Kernel',            acc:0.953, f1:0.951, auc:0.989, gap:0.027, time:0.6, judge:86.4, winner:false, hp:'C 4.0 · γ scale' },
  { rank:5, model:'Logistic Regression',family:'Linear',            acc:0.947, f1:0.945, auc:0.985, gap:0.012, time:0.3, judge:84.0, winner:false, hp:'C 1.0 · L2' },
  { rank:6, model:'K-Nearest Neighbors',family:'Instance',          acc:0.933, f1:0.930, auc:0.972, gap:0.044, time:0.2, judge:78.9, winner:false, hp:'k 7 · distance wt' },
];

// ---- SHAP feature importances for the winner ----
const SHAP = [
  { feature:'petal_length', value:0.412 },
  { feature:'petal_width',  value:0.367 },
  { feature:'sepal_length', value:0.142 },
  { feature:'sepal_width',  value:0.079 },
];

// ---- recent runs (dashboard) ----
const RUNS = [
  { id:'run_4f2a', dataset:'iris.csv',           task:'Classification', models:6, best:'XGBoost',       acc:0.973, status:'done',    when:'just now',  drift:'stable' },
  { id:'run_3b91', dataset:'housing.csv',        task:'Regression',     models:5, best:'LightGBM',      acc:0.911, status:'done',    when:'2h ago',    drift:'stable' },
  { id:'run_2c77', dataset:'churn.csv',          task:'Classification', models:7, best:'Random Forest', acc:0.884, status:'done',    when:'yesterday', drift:'watch'  },
  { id:'run_1a05', dataset:'cats-dogs-10.zip',   task:'Image / CNN',    models:3, best:'ResNet-18',     acc:0.940, status:'done',    when:'2d ago',    drift:'stable' },
  { id:'run_0f30', dataset:'segments.csv',       task:'Unsupervised',   models:4, best:'K-Means (k=4)', acc:null,  status:'review',  when:'3d ago',    drift:'—'      },
];

// ---- example dataset cards for the upload screen ----
const SAMPLE_DATASETS = [
  { name:'iris.csv',         size:'4.5 KB',  rows:'150',    cols:'5',  task:'Classification', note:'Canonical fixture' },
  { name:'housing.csv',      size:'1.4 MB',  rows:'20,640', cols:'9',  task:'Regression',     note:'Tabular numeric' },
  { name:'cats-dogs-10.zip', size:'18.2 MB', rows:'2,000',  cols:'img',task:'Image / CNN',    note:'Image fixture' },
];

Object.assign(window, {
  AGENTS, AGENT, STAGES, STAGE_LOGS, LEADERBOARD, SHAP, RUNS, SAMPLE_DATASETS,
});
