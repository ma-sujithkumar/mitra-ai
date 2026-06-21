async function parseJsonResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload?.detail?.message || payload?.message || 'Request failed';
    const error = new Error(message);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

export async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  });
  return parseJsonResponse(response);
}

export async function fetchPublicConfig() {
  return requestJson('/api/config/public');
}

export async function fetchHealth() {
  return requestJson('/api/health');
}

export async function uploadDataset({ datasetFile, metadataFile }) {
  const formData = new FormData();
  formData.append('dataset_file', datasetFile);
  if (metadataFile) {
    formData.append('metadata_file', metadataFile);
  }

  const response = await fetch('/api/upload', {
    method: 'POST',
    body: formData,
  });
  return parseJsonResponse(response);
}

export async function fetchRecentUploads(limit = 5) {
  return requestJson(`/api/uploads/recent?limit=${encodeURIComponent(limit)}`);
}

export async function startValidation(payload) {
  return requestJson('/api/validate', {
    method: 'POST',
    body: JSON.stringify({
      session_id: payload.sessionId,
      target_col: payload.targetCol || null,
      validation_split: payload.validationSplit ?? null,
      // Per-run null threshold override (null => use server default).
      null_threshold: payload.nullThreshold ?? null,
    }),
  });
}

export async function startMetadata(payload) {
  return requestJson('/api/metadata', {
    method: 'POST',
    body: JSON.stringify({
      session_id: payload.sessionId,
      description: payload.description || null,
      target_col: payload.targetCol || null,
      problem_type: payload.problemType || null,
      provider: payload.provider || null,
      model: payload.model || null,
      api_key: payload.apiKey || null,
      gateway_url: payload.gatewayUrl || null,
      // Skip the agent when metadata.json exists unless an explicit re-run.
      force: payload.force ?? false,
    }),
  });
}

export async function runLlmSmokeTest(payload) {
  return requestJson('/api/llm/smoke-test', {
    method: 'POST',
    body: JSON.stringify({
      provider: payload.provider || null,
      model: payload.model || null,
      api_key: payload.apiKey || null,
      gateway_url: payload.gatewayUrl || null,
    }),
  });
}

export async function fetchRuns(limit = 5) {
  return requestJson(`/api/runs?limit=${encodeURIComponent(limit)}`);
}

export async function fetchRunStats() {
  return requestJson('/api/runs/stats');
}

export async function fetchLeaderboard(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/leaderboard`);
}

export async function fetchVerdict(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/verdict`);
}

export async function fetchShap(sessionId, modelName = null) {
  const query = modelName ? `?model_name=${encodeURIComponent(modelName)}` : '';
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/shap${query}`);
}

export async function fetchTokens(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/tokens`);
}

export async function fetchPlots(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/plots`);
}

export function plotUrl(sessionId, plotPath) {
  return `/api/runs/${encodeURIComponent(sessionId)}/plots/${plotPath}`;
}

export async function fetchActivityLog(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/activity`);
}

export function activityLogDownloadUrl(sessionId) {
  return `/api/logs/download/${encodeURIComponent(sessionId)}`;
}

export async function fetchAdvancedConfig(sessionId = null) {
  const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
  return requestJson(`/api/config/advanced${query}`);
}

export async function saveAdvancedConfig(sessionId, overrides) {
  return requestJson(`/api/config/advanced?session_id=${encodeURIComponent(sessionId)}`, {
    method: 'PUT',
    body: JSON.stringify({ overrides }),
  });
}

export async function fetchFeatureEngineering(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/feature-engineering`);
}

export async function startFeatureEngineering(payload) {
  return requestJson('/api/feature-engineering', {
    method: 'POST',
    body: JSON.stringify({
      session_id: payload.sessionId,
      target_col: payload.targetCol || null,
      problem_type: payload.problemType || null,
      provider: payload.provider || null,
      model: payload.model || null,
      api_key: payload.apiKey || null,
      gateway_url: payload.gatewayUrl || null,
      // Skip the pipeline when FE artifacts exist unless an explicit re-run.
      force: payload.force ?? false,
    }),
  });
}

export async function fetchRunProgress(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/progress`);
}

export async function fetchFeatureEngineeringJobStatus(sessionId) {
  return requestJson(`/api/feature-engineering/status?session_id=${encodeURIComponent(sessionId)}`);
}

export async function fetchD2VPrior(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/d2v-prior`);
}

export async function fetchModelConfig(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/model-config`);
}

export async function fetchValidationReport(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/validation`);
}

export async function fetchHpt(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/hpt`);
}

export async function runHpt(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/hpt/run`, {
    method: 'POST',
  });
}

export function modelDownloadUrl(sessionId, modelName) {
  return `/api/runs/${encodeURIComponent(sessionId)}/models/${encodeURIComponent(modelName)}/download`;
}

export function modelsDownloadAllUrl(sessionId) {
  return `/api/runs/${encodeURIComponent(sessionId)}/models/download-all`;
}

export async function generatePlots(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/plots/generate`, {
    method: 'POST',
  });
}

export async function fetchShapStatus(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/evaluation/shap/status`);
}

export async function fetchOverfittingStatus(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/evaluation/overfitting/status`);
}

export async function fetchJudgeStatus(sessionId) {
  return requestJson(`/api/runs/${encodeURIComponent(sessionId)}/evaluation/judge/status`);
}

