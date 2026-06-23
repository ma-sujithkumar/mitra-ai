import { useState } from 'react';

import FormField from '../components/FormField.jsx';
import { validateSignup } from './authValidation.js';

function SignupForm({ onSubmit, submitting }) {
  const [name, setName] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [fieldErrors, setFieldErrors] = useState([]);

  function handleSubmit(event) {
    event.preventDefault();
    const validation = validateSignup({ name, username, password });
    if (!validation.valid) {
      setFieldErrors(validation.errors);
      return;
    }
    setFieldErrors([]);
    onSubmit({ name, username, password });
  }

  return (
    <form className="auth-form" onSubmit={handleSubmit}>
      <FormField label="Name">
        <input
          autoComplete="name"
          className="input"
          onChange={(event) => setName(event.target.value)}
          placeholder="your full name"
          type="text"
          value={name}
        />
      </FormField>
      <FormField label="Username" hint="email or 3+ chars">
        <input
          autoComplete="username"
          className="input"
          onChange={(event) => setUsername(event.target.value)}
          placeholder="username or email"
          type="text"
          value={username}
        />
      </FormField>
      <FormField label="Password" hint="8+ chars, letter + digit">
        <input
          autoComplete="new-password"
          className="input"
          onChange={(event) => setPassword(event.target.value)}
          placeholder="create a password"
          type="password"
          value={password}
        />
      </FormField>
      {fieldErrors.length > 0 ? (
        <ul className="auth-errors">
          {fieldErrors.map((message) => (
            <li key={message}>{message}</li>
          ))}
        </ul>
      ) : null}
      <button className="btn btn-primary full-width" disabled={submitting} type="submit">
        {submitting ? 'Creating account...' : 'Signup'}
      </button>
    </form>
  );
}

export default SignupForm;
