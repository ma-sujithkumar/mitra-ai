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

async function requestJson(path, options = {}) {
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
