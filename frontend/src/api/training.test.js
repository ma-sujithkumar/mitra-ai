import assert from 'node:assert/strict';
import test from 'node:test';

import { cancelTraining, fetchTrainingStatus, startTraining, resetTraining } from './training.js';

function jsonResponse(payload, ok = true, status = 200) {
  return {
    ok,
    status,
    async json() {
      return payload;
    },
  };
}

test('startTraining sends the backend training contract', async () => {
  let capturedPath = null;
  let capturedOptions = null;
  globalThis.fetch = async (path, options) => {
    capturedPath = path;
    capturedOptions = options;
    return jsonResponse({ session_id: 'session-1', status: 'created' }, true, 202);
  };

  const payload = await startTraining({
    sessionId: 'session-1',
    targetColumn: 'species',
    executionMode: 'ray',
    timeoutSec: 120,
  });

  assert.equal(capturedPath, '/api/training/start');
  assert.equal(capturedOptions.method, 'POST');
  assert.deepEqual(JSON.parse(capturedOptions.body), {
    session_id: 'session-1',
    metadata_path: null,
    model_config_path: null,
    train_path: null,
    test_path: null,
    target_column: 'species',
    problem_type: null,
    allow_fallback_artifacts: true,
    execution_mode: 'ray',
    timeout_sec: 120,
  });
  assert.equal(payload.status, 'created');
});

test('status, cancellation, and reset encode the session id', async () => {
  const paths = [];
  const options = [];
  globalThis.fetch = async (path, opts) => {
    paths.push(path);
    options.push(opts);
    return jsonResponse({ session_id: 'session with space', status: 'running' });
  };

  await fetchTrainingStatus('session with space');
  await cancelTraining('session with space');
  await resetTraining('session with space');

  assert.deepEqual(paths, [
    '/api/training/status/session%20with%20space',
    '/api/training/cancel/session%20with%20space',
    '/api/training/reset/session%20with%20space',
  ]);
  assert.equal(options[1].method, 'POST');
  assert.equal(options[2].method, 'POST');
});

test('training API surfaces structured backend errors', async () => {
  globalThis.fetch = async () => jsonResponse({
    detail: {
      error: 'TRAINING_ARTIFACTS_INVALID',
      message: 'Required training artifacts are missing',
    },
  }, false, 422);

  await assert.rejects(
    () => startTraining({ sessionId: 'missing' }),
    /Required training artifacts are missing/,
  );
});
