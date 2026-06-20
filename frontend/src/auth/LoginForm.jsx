import { useState } from 'react';

import FormField from '../components/FormField.jsx';

function LoginForm({ onSubmit, submitting }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  function handleSubmit(event) {
    event.preventDefault();
    onSubmit({ username, password });
  }

  return (
    <form className="auth-form" onSubmit={handleSubmit}>
      <FormField label="Username">
        <input
          autoComplete="username"
          className="input"
          onChange={(event) => setUsername(event.target.value)}
          placeholder="your username or email"
          type="text"
          value={username}
        />
      </FormField>
      <FormField label="Password">
        <input
          autoComplete="current-password"
          className="input"
          onChange={(event) => setPassword(event.target.value)}
          placeholder="your password"
          type="password"
          value={password}
        />
      </FormField>
      <button className="btn btn-primary full-width" disabled={submitting} type="submit">
        {submitting ? 'Signing in...' : 'Login'}
      </button>
    </form>
  );
}

export default LoginForm;
