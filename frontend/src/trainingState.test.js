import assert from 'node:assert/strict';
import test from 'node:test';

import {
  applyTrainingEvent,
  applyTrainingStatus,
  createTrainingState,
  overallTrainingProgress,
  selectTrainingModels,
  trainingCounts,
} from './trainingState.js';

function event(overrides = {}) {
  return {
    session_id: 'session-1',
    stage: 'training',
    level: 'info',
    msg: 'queued',
    pct: 0,
    status: 'queued',
    ts: '2026-06-17T10:00:00Z',
    sequence: 1,
    model_id: 'model_001',
    model_name: 'RandomForestClassifier',
    details: {},
    ...overrides,
  };
}

test('queued event creates a model with priority and rationale', () => {
  const state = applyTrainingEvent(createTrainingState(), event({
    details: { priority: 2, rationale: 'Strong tabular baseline' },
  }));
  const [model] = selectTrainingModels(state);

  assert.equal(model.priority, 2);
  assert.equal(model.rationale, 'Strong tabular baseline');
  assert.equal(model.status, 'queued');
  assert.equal(state.logs.length, 1);
});

test('completion events may arrive out of submission order', () => {
  let state = createTrainingState();
  state = applyTrainingEvent(state, event({ model_id: 'model_001', sequence: 1 }));
  state = applyTrainingEvent(state, event({
    model_id: 'model_002',
    model_name: 'LogisticRegression',
    sequence: 2,
    details: { priority: 2 },
  }));
  state = applyTrainingEvent(state, event({
    model_id: 'model_002',
    model_name: 'LogisticRegression',
    status: 'completed',
    pct: 100,
    sequence: 3,
    details: { validation_score: 0.93 },
  }));
  state = applyTrainingEvent(state, event({
    model_id: 'model_001',
    status: 'running',
    pct: 25,
    sequence: 4,
  }));

  assert.equal(state.models.model_002.status, 'completed');
  assert.equal(state.models.model_001.status, 'running');
  assert.equal(trainingCounts(state).completed, 1);
  assert.equal(overallTrainingProgress(state), 63);
});

test('terminal model status always displays one hundred percent', () => {
  const state = applyTrainingEvent(createTrainingState(), event({
    status: 'timed_out',
    pct: 25,
    details: { error: 'Ray task timed out' },
  }));

  assert.equal(state.models.model_001.pct, 100);
  assert.equal(trainingCounts(state).failed, 1);
});

test('all_completed event stores summary and finishes the session', () => {
  let state = applyTrainingEvent(createTrainingState(), event());
  state = applyTrainingEvent(state, event({
    model_id: null,
    model_name: null,
    status: 'all_completed',
    pct: 100,
    sequence: 2,
    msg: 'All model training jobs finished: 1 completed, 0 failed',
    details: {
      summary_status: 'completed',
      total_models: 1,
      completed: 1,
      failed: 0,
    },
  }));

  assert.equal(state.complete, true);
  assert.deepEqual(state.summary, {
    status: 'completed',
    total: 1,
    completed: 1,
    failed: 0,
    message: 'All model training jobs finished: 1 completed, 0 failed',
  });
});


test('backend status hydrates Page 2 after refresh without replayed events', () => {
  const state = applyTrainingStatus(createTrainingState(), {
    session_id: 'session-1',
    status: 'completed',
    total_models: 2,
    completed_models: 2,
    failed_models: 0,
    model_states: [
      {
        model_id: 'model_001',
        model_name: 'RandomForestClassifier',
        status: 'completed',
        pct: 100,
        updated_at: '2026-06-17T10:02:00Z',
        validation_score: 0.97,
        model_path: '.mitra/session-1/training/model_001/model.pkl',
        training_time_sec: 1.42,
        error: null,
      },
      {
        model_id: 'model_002',
        model_name: 'LogisticRegression',
        status: 'completed',
        pct: 100,
        updated_at: '2026-06-17T10:02:02Z',
        validation_score: 0.94,
        model_path: '.mitra/session-1/training/model_002/model.pkl',
        training_time_sec: 0.88,
        error: null,
      },
    ],
  });

  assert.equal(state.complete, true);
  assert.equal(state.summary.status, 'completed');
  assert.equal(state.summary.completed, 2);
  assert.equal(trainingCounts(state).completed, 2);
  assert.equal(state.models.model_001.details.validation_score, 0.97);
  assert.equal(state.models.model_002.details.model_path, '.mitra/session-1/training/model_002/model.pkl');
});

test('backend status preserves SSE priority and marks partial failures', () => {
  let state = createTrainingState();
  state = applyTrainingEvent(state, event({
    model_id: 'model_010',
    model_name: 'XGBoostClassifier',
    details: { priority: 7, rationale: 'High expected tabular score' },
  }));
  state = applyTrainingStatus(state, {
    session_id: 'session-1',
    status: 'partial_failure',
    total_models: 1,
    completed_models: 0,
    failed_models: 1,
    model_states: [
      {
        model_id: 'model_010',
        model_name: 'XGBoostClassifier',
        status: 'timed_out',
        pct: 35,
        updated_at: '2026-06-17T10:05:00Z',
        validation_score: null,
        model_path: null,
        training_time_sec: null,
        error: 'Ray task timed out',
      },
    ],
  });

  assert.equal(state.models.model_010.priority, 7);
  assert.equal(state.models.model_010.rationale, 'High expected tabular score');
  assert.equal(state.models.model_010.status, 'timed_out');
  assert.equal(state.models.model_010.pct, 100);
  assert.equal(state.summary.status, 'partial_failure');
  assert.equal(trainingCounts(state).failed, 1);
});

test('cancelled backend status restores cancelled cards and summary', () => {
  const state = applyTrainingStatus(createTrainingState(), {
    session_id: 'session-1',
    status: 'cancelled',
    total_models: 1,
    completed_models: 0,
    failed_models: 1,
    model_states: [
      {
        model_id: 'model_001',
        model_name: 'RandomForestClassifier',
        status: 'cancelled',
        pct: 10,
        updated_at: '2026-06-17T10:06:00Z',
        validation_score: null,
        model_path: null,
        training_time_sec: null,
        error: 'Training cancellation was requested',
      },
    ],
  });

  assert.equal(state.complete, true);
  assert.equal(state.summary.status, 'cancelled');
  assert.equal(state.models.model_001.status, 'cancelled');
  assert.equal(state.models.model_001.pct, 100);
  assert.equal(state.models.model_001.details.error, 'Training cancellation was requested');
});
