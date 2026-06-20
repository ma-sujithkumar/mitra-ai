import errorMessages from './data/errorMessages.json';

// Resolve a thrown API error (from client.js) into a friendly, actionable
// shape. The backend attaches a stable code at payload.detail.error; we map it
// to a human message + remediation via errorMessages.json so there is no
// if-else ladder over error strings (CLAUDE.md rule 4/23).
export function resolveError(error) {
  if (!error) {
    return null;
  }
  const code = error?.payload?.detail?.error;
  const mapped = code ? errorMessages[code] : null;
  if (mapped) {
    return { code, message: mapped.message, fix: mapped.fix };
  }
  // Fall back to the server-provided message (or a generic one) with no fix.
  return {
    code: code || null,
    message: error.message || 'Something went wrong.',
    fix: null,
  };
}

// Convenience formatter for toasts: "message Fix: ..." on a single line.
export function formatErrorText(error) {
  const resolved = resolveError(error);
  if (!resolved) {
    return '';
  }
  return resolved.fix ? `${resolved.message} Fix: ${resolved.fix}` : resolved.message;
}
