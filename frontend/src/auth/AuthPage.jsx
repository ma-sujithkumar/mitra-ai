import { useState } from 'react';

import Segmented from '../components/Segmented.jsx';
import ForgotPasswordModal from './ForgotPasswordModal.jsx';
import LoginForm from './LoginForm.jsx';
import SignupForm from './SignupForm.jsx';
import { login as loginRequest, signup as signupRequest } from './authApi.js';
import './auth.css';

const MODE_OPTIONS = [
  { value: 'login', label: 'Login' },
  { value: 'signup', label: 'Signup' },
];

// Mode-to-handler map avoids an if/else ladder when submitting either form.
const SUBMIT_HANDLERS = {
  login: loginRequest,
  signup: signupRequest,
};

function AuthPage({ onAuthenticated }) {
  const [mode, setMode] = useState('login');
  const [submitting, setSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [showForgot, setShowForgot] = useState(false);

  function changeMode(nextMode) {
    setMode(nextMode);
    setErrorMessage('');
  }

  async function handleSubmit(payload) {
    setSubmitting(true);
    setErrorMessage('');
    try {
      const submit = SUBMIT_HANDLERS[mode];
      const user = await submit(payload);
      onAuthenticated(user);
    } catch (authError) {
      setErrorMessage(authError.message || 'Authentication failed.');
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-shell">
      <div className="auth-card">
        <div className="auth-head">
          <div className="auth-brand-mark">M</div>
          <h1>MITRA AI</h1>
          <p className="muted">Sign in to continue to your workspace.</p>
        </div>
        <Segmented label="Auth mode" onChange={changeMode} options={MODE_OPTIONS} value={mode} />
        {errorMessage ? <div className="auth-alert">{errorMessage}</div> : null}
        {mode === 'login' ? (
          <LoginForm onSubmit={handleSubmit} submitting={submitting} />
        ) : (
          <SignupForm onSubmit={handleSubmit} submitting={submitting} />
        )}
        <button className="auth-link" onClick={() => setShowForgot(true)} type="button">
          Forgot password?
        </button>
      </div>
      {showForgot ? <ForgotPasswordModal onClose={() => setShowForgot(false)} /> : null}
    </div>
  );
}

export default AuthPage;
