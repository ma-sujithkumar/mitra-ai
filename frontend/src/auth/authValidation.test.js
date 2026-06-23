import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  validateLogin,
  validateName,
  validatePassword,
  validateSignup,
  validateUsername,
} from './authValidation.js';

test('valid email username passes', () => {
  const result = validateUsername('user@example.com');
  assert.equal(result.valid, true);
  assert.deepEqual(result.errors, []);
});

test('malformed email username fails', () => {
  const result = validateUsername('user@@example');
  assert.equal(result.valid, false);
});

test('short plain username fails', () => {
  const result = validateUsername('ab');
  assert.equal(result.valid, false);
});

test('plain username with valid chars passes', () => {
  assert.equal(validateUsername('john_doe-1').valid, true);
});

test('password requires length, letter, and digit', () => {
  assert.equal(validatePassword('short1').valid, false);
  assert.equal(validatePassword('allletters').valid, false);
  assert.equal(validatePassword('12345678').valid, false);
  assert.equal(validatePassword('secret12').valid, true);
});

test('name must not be empty', () => {
  assert.equal(validateName('   ').valid, false);
  assert.equal(validateName('Ada').valid, true);
});

test('signup aggregates all failing rules', () => {
  const result = validateSignup({ name: '', username: 'ab', password: 'x' });
  assert.equal(result.valid, false);
  assert.equal(result.errors.length >= 3, true);
});

test('login only requires non-empty fields', () => {
  assert.equal(validateLogin({ username: '', password: '' }).valid, false);
  assert.equal(validateLogin({ username: 'a', password: 'b' }).valid, true);
});
