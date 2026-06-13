const API_BASE = 'http://localhost:8000/api';

export async function uploadFile(file, formData = {}) {
  const body = new FormData();
  body.append('file', file);
  if (formData.target_col)   body.append('target_col', formData.target_col);
  if (formData.problem_type) body.append('problem_type', formData.problem_type);
  if (formData.description)  body.append('description', formData.description);

  const response = await fetch(`${API_BASE}/upload`, { method: 'POST', body });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || 'Upload failed');
  }
  return response.json();
}

export function streamValidation(sessionId, targetCol, onCheck, onDone, onError) {
  fetch(`${API_BASE}/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, target_col: targetCol }),
  }).then(async (response) => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const event = JSON.parse(line.slice(6));
          if (event.type === 'check') onCheck(event);
          else if (event.type === 'done') onDone(event);
          else if (event.type === 'error') onError(new Error(event.message));
        } catch {
          // skip malformed lines
        }
      }
    }
  }).catch(onError);
}

export function streamMetadata(sessionId, description, targetCol, problemType, onEvent, onDone, onError) {
  fetch(`${API_BASE}/metadata`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: sessionId,
      description,
      target_col: targetCol,
      problem_type: problemType,
    }),
  }).then(async (response) => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const event = JSON.parse(line.slice(6));
          if (event.type === 'done') onDone(event);
          else if (event.type === 'error') onError(new Error(event.message));
          else onEvent(event);
        } catch {
          // skip malformed lines
        }
      }
    }
  }).catch(onError);
}

export async function getRuns(limit = 5) {
  const response = await fetch(`${API_BASE}/runs?limit=${limit}`);
  if (!response.ok) return { runs: [] };
  return response.json();
}

export async function getRunStats() {
  const response = await fetch(`${API_BASE}/runs/stats`);
  if (!response.ok) return { total_runs: 0, models_trained: 0, best_accuracy: null, avg_run_time_min: null };
  return response.json();
}
