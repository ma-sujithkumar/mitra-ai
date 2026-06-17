export const TERMINAL_MODEL_STATUSES = new Set([
  'completed',
  'failed',
  'timed_out',
  'cancelled',
]);

export function createTrainingState() {
  return {
    models: {},
    modelOrder: [],
    logs: [],
    summary: null,
    complete: false,
  };
}

function normalizeProgress(status, pct) {
  if (TERMINAL_MODEL_STATUSES.has(status)) {
    return 100;
  }
  const numeric = Number(pct);
  if (!Number.isFinite(numeric)) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(numeric)));
}

export function applyTrainingEvent(state, event) {
  if (!event || typeof event !== 'object') {
    return state;
  }

  const logEntry = {
    sequence: event.sequence ?? state.logs.length + 1,
    ts: event.ts || new Date().toISOString(),
    level: event.level || 'info',
    status: event.status || 'running',
    modelId: event.model_id || null,
    modelName: event.model_name || null,
    message: event.msg || 'Training event',
  };
  const logs = [...state.logs, logEntry].slice(-500);

  if (event.status === 'all_completed') {
    return {
      ...state,
      logs,
      complete: true,
      summary: {
        status: event.details?.summary_status || 'completed',
        total: Number(event.details?.total_models ?? state.modelOrder.length),
        completed: Number(event.details?.completed ?? 0),
        failed: Number(event.details?.failed ?? 0),
        message: event.msg || 'All training jobs completed',
      },
    };
  }

  if (!event.model_id) {
    return { ...state, logs };
  }

  const isNew = !state.models[event.model_id];
  const details = event.details || {};
  const previous = state.models[event.model_id] || {
    modelId: event.model_id,
    modelName: event.model_name || event.model_id,
    priority: Number(details.priority || state.modelOrder.length + 1),
    rationale: details.rationale || 'Selected by the Model Selection agent.',
    status: 'queued',
    pct: 0,
    message: '',
    level: 'info',
    details: {},
  };

  const model = {
    ...previous,
    modelName: event.model_name || previous.modelName,
    priority: Number(details.priority || previous.priority),
    rationale: details.rationale || previous.rationale,
    status: event.status || previous.status,
    pct: normalizeProgress(event.status, event.pct),
    message: event.msg || previous.message,
    level: event.level || previous.level,
    timestamp: event.ts || previous.timestamp,
    details: {
      ...previous.details,
      ...details,
    },
  };

  return {
    ...state,
    logs,
    models: {
      ...state.models,
      [event.model_id]: model,
    },
    modelOrder: isNew
      ? [...state.modelOrder, event.model_id]
      : state.modelOrder,
  };
}

export function selectTrainingModels(state) {
  return state.modelOrder
    .map((modelId) => state.models[modelId])
    .filter(Boolean)
    .sort((left, right) => left.priority - right.priority);
}

export function trainingCounts(state) {
  const models = selectTrainingModels(state);
  return models.reduce(
    (counts, model) => {
      counts.total += 1;
      if (model.status === 'completed') {
        counts.completed += 1;
      } else if (['failed', 'timed_out', 'cancelled'].includes(model.status)) {
        counts.failed += 1;
      } else if (['running', 'submitted'].includes(model.status)) {
        counts.running += 1;
      } else {
        counts.queued += 1;
      }
      return counts;
    },
    { total: 0, queued: 0, running: 0, completed: 0, failed: 0 },
  );
}

export function overallTrainingProgress(state) {
  const models = selectTrainingModels(state);
  if (!models.length) {
    return state.complete ? 100 : 0;
  }
  return Math.round(
    models.reduce((total, model) => total + Number(model.pct || 0), 0) / models.length,
  );
}
