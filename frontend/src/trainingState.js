export const TERMINAL_MODEL_STATUSES = new Set([
  'completed',
  'failed',
  'timed_out',
  'cancelled',
]);

const TERMINAL_SESSION_STATUSES = new Set([
  'completed',
  'partial_failure',
  'failed',
  'cancelled',
]);

function statusDetails(modelState) {
  return {
    validation_score: modelState.validation_score ?? null,
    model_path: modelState.model_path ?? null,
    training_time_sec: modelState.training_time_sec ?? null,
    error: modelState.error ?? null,
  };
}

function statusSummary(payload, modelStates) {
  const total = Number(payload.total_models ?? modelStates.length);
  const completed = Number(payload.completed_models ?? modelStates.filter((item) => item.status === 'completed').length);
  const failed = Number(
    payload.failed_models
    ?? modelStates.filter((item) => ['failed', 'timed_out', 'cancelled'].includes(item.status)).length,
  );
  const status = payload.status || (failed > 0 ? 'partial_failure' : 'completed');
  const messages = {
    completed: `Training completed: ${completed}/${total} models succeeded`,
    partial_failure: `Training completed with failures: ${completed}/${total} models succeeded`,
    failed: 'Training session failed before completion',
    cancelled: 'Training session was cancelled',
  };

  return {
    status,
    total,
    completed,
    failed,
    message: messages[status] || 'Training session reached a terminal state',
  };
}

function orderedModelIds(existingOrder, nextModels) {
  const nextIds = Object.keys(nextModels).sort();
  return [
    ...existingOrder.filter((modelId) => nextModels[modelId]),
    ...nextIds.filter((modelId) => !existingOrder.includes(modelId)),
  ];
}

export function createTrainingState() {
  return {
    models: {},
    modelOrder: [],
    logs: [],
    summary: null,
    complete: false,
    // Highest event sequence applied so far. SSE reconnects re-replay the full
    // session history with the same sequence numbers; this lets us drop events
    // we have already applied so logs stay idempotent across reconnects.
    lastSequence: 0,
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

  // Idempotent replay handling: when the SSE stream reconnects the backend
  // re-sends the full history with the same (monotonic) sequence numbers. Drop
  // anything at or below the highest sequence we have already applied so logs
  // and completion are not duplicated. Events without a usable sequence are
  // always applied (best effort).
  const eventSequence = Number(event.sequence);
  const hasSequence = Number.isFinite(eventSequence) && eventSequence > 0;
  const priorSequence = state.lastSequence || 0;
  if (hasSequence && eventSequence <= priorSequence) {
    return state;
  }
  const lastSequence = hasSequence ? eventSequence : priorSequence;

  const logEntry = {
    sequence: event.sequence ?? state.logs.length + 1,
    ts: event.ts || new Date().toISOString(),
    level: event.level || 'info',
    status: event.status || 'running',
    stage: event.stage || 'training',
    modelId: event.model_id || null,
    modelName: event.model_name || null,
    message: event.msg || 'Training event',
  };
  const logs = [...state.logs, logEntry].slice(-500);

  // The training-stage all_completed event fires when all Ray model jobs finish.
  // Update summary so TrainingSummary can show model counts while eval is running.
  // Do NOT set complete=true yet -- that comes only from the final top-level event.
  const isTrainingStageComplete = event.status === 'all_completed' && event.stage === 'training';
  if (isTrainingStageComplete) {
    return {
      ...state,
      logs,
      lastSequence,
      summary: {
        status: event.details?.summary_status || 'completed',
        total: Number(event.details?.total_models ?? state.modelOrder.length),
        completed: Number(event.details?.completed ?? 0),
        failed: Number(event.details?.failed ?? 0),
        message: event.msg || 'All training jobs completed',
      },
    };
  }

  // Only the top-level pipeline completion (no stage) marks the session fully done
  // and shows the analytics section. This fires after SHAP, judge, and plots.
  const isTopLevelCompletion = event.status === 'all_completed' && !event.stage;
  if (isTopLevelCompletion) {
    return {
      ...state,
      logs,
      lastSequence,
      complete: true,
      // Preserve summary if already set from training-stage event.
      summary: state.summary || {
        status: event.details?.summary_status || 'completed',
        total: Number(event.details?.total_models ?? state.modelOrder.length),
        completed: Number(event.details?.completed ?? 0),
        failed: Number(event.details?.failed ?? 0),
        message: event.msg || 'Pipeline completed successfully',
      },
    };
  }

  if (!event.model_id) {
    return { ...state, logs, lastSequence };
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
    lastSequence,
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


export function applyTrainingStatus(state, payload) {
  if (!payload || typeof payload !== 'object') {
    return state;
  }

  const modelStates = Array.isArray(payload.model_states) ? payload.model_states : [];
  const nextModels = { ...state.models };

  modelStates.forEach((modelState, index) => {
    if (!modelState?.model_id) {
      return;
    }

    const previous = nextModels[modelState.model_id] || {
      modelId: modelState.model_id,
      modelName: modelState.model_name || modelState.model_id,
      priority: index + 1,
      rationale: 'Restored from backend training status.',
      status: 'queued',
      pct: 0,
      message: '',
      level: 'info',
      details: {},
    };

    nextModels[modelState.model_id] = {
      ...previous,
      modelName: modelState.model_name || previous.modelName,
      status: modelState.status || previous.status,
      pct: normalizeProgress(modelState.status || previous.status, modelState.pct),
      message: modelState.error || previous.message || `Backend status: ${modelState.status || previous.status}`,
      level: ['failed', 'timed_out'].includes(modelState.status) ? 'error' : previous.level,
      timestamp: modelState.updated_at || previous.timestamp,
      details: {
        ...previous.details,
        ...statusDetails(modelState),
      },
    };
  });

  const terminal = TERMINAL_SESSION_STATUSES.has(payload.status);
  return {
    ...state,
    complete: terminal ? true : state.complete,
    summary: terminal ? statusSummary(payload, modelStates) : state.summary,
    models: nextModels,
    modelOrder: orderedModelIds(state.modelOrder, nextModels),
  };
}
