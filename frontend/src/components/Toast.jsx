import { useEffect } from 'react';

import { Icons } from '../icons.jsx';

// Error tones stay until the user dismisses them; other tones (e.g. success)
// auto-dismiss so they do not pile up.
const AUTO_DISMISS_MS_BY_TONE = {
  error: null,
  warn: 6000,
  success: 4000,
};

// Fixed-position popup so feedback is visible immediately, regardless of how
// far down the page the user has scrolled (the inline top-of-page banner
// alone required scrolling back up to notice it).
function Toast({ message, tone = 'error', onDismiss }) {
  useEffect(() => {
    if (!message) {
      return undefined;
    }
    const autoDismissMs = AUTO_DISMISS_MS_BY_TONE[tone];
    if (!autoDismissMs) {
      return undefined;
    }
    const timerId = window.setTimeout(onDismiss, autoDismissMs);
    return () => window.clearTimeout(timerId);
  }, [message, tone, onDismiss]);

  if (!message) {
    return null;
  }

  return (
    <div className={`toast toast-${tone}`} role="alert">
      <Icons.alert size={16} />
      <span>{message}</span>
      <button aria-label="Dismiss" className="toast-close" onClick={onDismiss} type="button">
        <Icons.x size={14} />
      </button>
    </div>
  );
}

export default Toast;
