import assert from 'node:assert/strict';
import test from 'node:test';

import {
  applyTrainingEvent,
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
