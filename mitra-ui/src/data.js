export const AGENTS = [
  { id: 'validator',      name: 'Data Validator',   short: 'DV', hue: 14,  type: 'Python', role: 'Gates bad data before any LLM is touched' },
  { id: 'metadata',       name: 'Metadata Gen',     short: 'MG', hue: 262, type: 'LLM',    role: 'Builds the metadata.json contract every agent reads' },
  { id: 'feature',        name: 'Feature Selection',short: 'FS', hue: 212, type: 'LLM',    role: 'Ranks and prunes features for downstream models' },
  { id: 'model',          name: 'Model Selection',  short: 'MS', hue: 286, type: 'LLM',    role: 'Chooses candidate model families for the task' },
  { id: 'classification', name: 'Classification',   short: 'CL', hue: 158, type: 'LLM',    role: 'Trains and tunes classifier candidates' },
  { id: 'regression',     name: 'Regression / USL', short: 'RU', hue: 188, type: 'LLM',    role: 'Regression and unsupervised model family' },
  { id: 'judge',          name: 'Judge',            short: 'JD', hue: 38,  type: 'LLM',    role: 'Weighs every proposal and converges on a verdict' },
  { id: 'hpt',            name: 'HPT',              short: 'HP', hue: 330, type: 'Python',  role: 'Guided hyperparameter search within budget' },
];

export const AGENT_MAP = Object.fromEntries(AGENTS.map(agent => [agent.id, agent]));

export const ROUTE_META = {
  dashboard:   { title: 'Dashboard',      sub: 'Runs, agents, and system health at a glance',           icon: 'grid' },
  upload:      { title: 'New Run',        sub: 'Upload a dataset and let the agents take over',         icon: 'upload' },
  pipeline:    { title: 'Live Pipeline',  sub: 'Eight specialist agents · real-time deliberation',      icon: 'flow' },
  leaderboard: { title: 'Leaderboard',    sub: "Ranked models with the Judge's justification",          icon: 'trophy' },
};

export const SAMPLE_DATASETS = [
  { name: 'iris.csv',          size: '4.5 KB',  rows: '150',    cols: '5',   task: 'Classification', note: 'Canonical fixture' },
  { name: 'housing.csv',       size: '1.4 MB',  rows: '20,640', cols: '9',   task: 'Regression',     note: 'Tabular numeric' },
  { name: 'cats-dogs-10.zip',  size: '18.2 MB', rows: '2,000',  cols: 'img', task: 'Image / CNN',    note: 'Image fixture' },
];

export const PROVIDERS = [
  { value: 'anthropic', label: 'Anthropic', placeholder: 'sk-ant-...' },
  { value: 'openai',    label: 'OpenAI',    placeholder: 'sk-...' },
  { value: 'gemini',    label: 'Gemini',    placeholder: 'AIza...' },
];
