const STATUS_MAP = {
  idle: ['pill-idle', 'Idle'],
  queued: ['pill-queued', 'Queued'],
  running: ['pill-run', 'Running'],
  submitted: ['pill-run', 'Submitted'],
  done: ['pill-done', 'Done'],
  completed: ['pill-done', 'Completed'],
  complete: ['pill-done', 'Complete'],
  review: ['pill-warn', 'Review'],
  warn: ['pill-warn', 'Warning'],
  warning: ['pill-warn', 'Warning'],
  blocked: ['pill-err', 'Blocked'],
  error: ['pill-err', 'Error'],
  failed: ['pill-err', 'Failed'],
  timed_out: ['pill-err', 'Timed out'],
  cancelled: ['pill-warn', 'Cancelled'],
  passed: ['pill-done', 'Passed'],
  pending: ['pill-queued', 'Pending'],
};

function StatusPill({ status, label, spin = false }) {
  const [className, defaultLabel] = STATUS_MAP[status] || ['pill-idle', status];
  const resolvedLabel = label || defaultLabel;

  return (
    <span className={`pill ${className}`} role="status" aria-label={`Status: ${resolvedLabel}`}>
      {status === 'running' && spin ? <span className="spinner small" /> : <span className="dot" />}
      {resolvedLabel}
    </span>
  );
}

export default StatusPill;
