import { requestJson } from './client.js';

export async function startTraining(payload) {
  return requestJson('/api/training/start', {
    method: 'POST',
    body: JSON.stringify({
      session_id: payload.sessionId,
      metadata_path: payload.metadataPath || null,
      model_config_path: payload.modelConfigPath || null,
      train_path: payload.trainPath || null,
      test_path: payload.testPath || null,
      target_column: payload.targetColumn || null,
      problem_type: payload.problemType || null,
      allow_fallback_artifacts: payload.allowFallbackArtifacts ?? true,
      execution_mode: payload.executionMode || null,
      timeout_sec: payload.timeoutSec ?? null,
    }),
  });
}

export async function fetchTrainingStatus(sessionId) {
  return requestJson(`/api/training/status/${encodeURIComponent(sessionId)}`);
}

export async function cancelTraining(sessionId) {
  return requestJson(`/api/training/cancel/${encodeURIComponent(sessionId)}`, {
    method: 'POST',
  });
}

export async function resetTraining(sessionId) {
  return requestJson(`/api/training/reset/${encodeURIComponent(sessionId)}`, {
    method: 'POST',
  });
}

export async function restartJudgeTurn(sessionId) {
  return requestJson(`/api/training/${encodeURIComponent(sessionId)}/judge/restart-turn`, {
    method: 'POST',
  });
}

