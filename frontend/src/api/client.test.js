import assert from 'node:assert/strict';
import test from 'node:test';

import { runLlmSmokeTest, startFeatureEngineering, startMetadata } from './client.js';

function jsonResponse(payload, ok = true, status = 200) {
  return {
    ok,
    status,
    async json() {
      return payload;
    },
  };
}

test('runLlmSmokeTest sends the user-entered LLM settings', async () => {
  let capturedPath = null;
  let capturedOptions = null;
  globalThis.fetch = async (path, options) => {
    capturedPath = path;
    capturedOptions = options;
    return jsonResponse({ status: 'ok' });
  };

  await runLlmSmokeTest({
    provider: 'anthropic',
    model: 'anthropic/demo-model',
    apiKey: 'secret-key',
    gatewayUrl: 'https://gateway.example.test',
  });

  assert.equal(capturedPath, '/api/llm/smoke-test');
  assert.equal(capturedOptions.method, 'POST');
  assert.deepEqual(JSON.parse(capturedOptions.body), {
    provider: 'anthropic',
    model: 'anthropic/demo-model',
    api_key: 'secret-key',
    gateway_url: 'https://gateway.example.test',
  });
});

test('downstream agent starts rely on the persisted backend LLM config', async () => {
  const capturedCalls = [];
  globalThis.fetch = async (path, options) => {
    capturedCalls.push({
      path,
      method: options.method,
      body: JSON.parse(options.body),
    });
    return jsonResponse({ session_id: 'session-1', status: 'accepted' });
  };

  const llmSettings = {
    provider: 'anthropic',
    model: 'anthropic/demo-model',
    apiKey: 'secret-key',
    gatewayUrl: 'https://gateway.example.test',
  };
  await startMetadata({
    sessionId: 'session-1',
    description: 'Predict the target.',
    targetCol: 'target',
    problemType: 'classification',
    force: true,
    ...llmSettings,
  });
  await startFeatureEngineering({
    sessionId: 'session-1',
    targetCol: 'target',
    problemType: 'classification',
    force: true,
    ...llmSettings,
  });

  assert.deepEqual(capturedCalls, [
    {
      path: '/api/metadata',
      method: 'POST',
      body: {
        session_id: 'session-1',
        description: 'Predict the target.',
        target_col: 'target',
        problem_type: 'classification',
        force: true,
      },
    },
    {
      path: '/api/feature-engineering',
      method: 'POST',
      body: {
        session_id: 'session-1',
        target_col: 'target',
        problem_type: 'classification',
        force: true,
      },
    },
  ]);
});
