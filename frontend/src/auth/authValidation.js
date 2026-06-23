// Pure validation helpers for the auth forms. Returned shape is
// { valid: boolean, errors: string[] } so callers can block submit and show
// every failing rule at once. Kept dependency-free for node --test coverage.

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const USERNAME_PATTERN = /^[A-Za-z0-9._-]+$/;
const PASSWORD_MIN_LENGTH = 8;

export function validateUsername(rawUsername) {
  const username = (rawUsername || '').trim();
  const errors = [];
  if (!username) {
    errors.push('Username is required.');
    return { valid: false, errors };
  }
  const looksLikeEmail = username.includes('@');
  if (looksLikeEmail) {
    if (!EMAIL_PATTERN.test(username)) {
      errors.push('Enter a valid email address.');
    }
  } else if (username.length < 3 || !USERNAME_PATTERN.test(username)) {
    errors.push('Username must be at least 3 characters (letters, digits, . _ -).');
  }
  return { valid: errors.length === 0, errors };
}

export function validatePassword(rawPassword) {
  const password = rawPassword || '';
  const errors = [];
  if (password.length < PASSWORD_MIN_LENGTH) {
    errors.push(`Password must be at least ${PASSWORD_MIN_LENGTH} characters.`);
  }
  if (!/[A-Za-z]/.test(password)) {
    errors.push('Password must include at least one letter.');
  }
  if (!/[0-9]/.test(password)) {
    errors.push('Password must include at least one digit.');
  }
  return { valid: errors.length === 0, errors };
}

export function validateName(rawName) {
  const name = (rawName || '').trim();
  const errors = [];
  if (!name) {
    errors.push('Name is required.');
  }
  return { valid: errors.length === 0, errors };
}

export function validateSignup({ name, username, password }) {
  const nameResult = validateName(name);
  const usernameResult = validateUsername(username);
  const passwordResult = validatePassword(password);
  const errors = [
    ...nameResult.errors,
    ...usernameResult.errors,
    ...passwordResult.errors,
  ];
  return { valid: errors.length === 0, errors };
}

export function validateLogin({ username, password }) {
  const errors = [];
  if (!(username || '').trim()) {
    errors.push('Username is required.');
  }
  if (!(password || '')) {
    errors.push('Password is required.');
  }
  return { valid: errors.length === 0, errors };
}
