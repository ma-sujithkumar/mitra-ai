import { useEffect, useRef } from 'react';

import { Icons } from '../icons.jsx';

// Lightweight modal confirmation for expensive/irreversible actions (start
// training, cancel a run). Renders only when `open`. Closes on Escape and on
// backdrop click; focuses the confirm button on open for keyboard users.
function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  tone = 'primary',
  onConfirm,
  onCancel,
}) {
  const confirmRef = useRef(null);

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    confirmRef.current?.focus();
    function handleKey(event) {
      if (event.key === 'Escape') {
        onCancel?.();
      }
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [open, onCancel]);

  if (!open) {
    return null;
  }

  return (
    <div className="dialog-backdrop" onClick={onCancel}>
      <div
        className="dialog-card"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="dialog-head">
          <Icons.alert size={18} />
          <h3>{title}</h3>
        </div>
        {body ? <p className="dialog-body">{body}</p> : null}
        <div className="dialog-actions">
          <button className="btn btn-ghost" onClick={onCancel} type="button">
            {cancelLabel}
          </button>
          <button
            className={`btn ${tone === 'danger' ? 'btn-danger' : 'btn-primary'}`}
            onClick={onConfirm}
            ref={confirmRef}
            type="button"
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ConfirmDialog;
