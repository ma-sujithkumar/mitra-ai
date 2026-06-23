import { requestJson } from '../api/client.js';

// Thin wrappers over the backend auth endpoints. Backend errors are surfaced by
// requestJson via payload.detail.message (e.g. "user id already exists").

export async function signup({ name, username, password }) {
  return requestJson('/api/auth/signup', {
    method: 'POST',
    body: JSON.stringify({ name, username, password }),
  });
}

export async function login({ username, password }) {
  return requestJson('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  });
}
