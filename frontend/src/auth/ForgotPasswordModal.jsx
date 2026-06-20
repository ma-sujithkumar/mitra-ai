function ForgotPasswordModal({ onClose }) {
  return (
    <div className="auth-modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="auth-modal"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Forgot password"
      >
        <h2>Forgot password</h2>
        <p className="muted">
          Forgot-password feature is not implemented. If you forgot your id and
          password, contact deeplearning1227@gmail.com or create a new id.
        </p>
        <button className="btn btn-secondary full-width" onClick={onClose} type="button">
          Close
        </button>
      </div>
    </div>
  );
}

export default ForgotPasswordModal;
