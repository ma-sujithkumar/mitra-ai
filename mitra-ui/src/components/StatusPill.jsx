import React from 'react';

const STATUS_MAP = {
  idle:    ['pill-idle',   'Idle'],
  queued:  ['pill-queued', 'Queued'],
  running: ['pill-run',    'Running'],
  done:    ['pill-done',   'Done'],
  review:  ['pill-warn',   'Review'],
  warn:    ['pill-warn',   'Watch'],
  error:   ['pill-err',    'Error'],
  passed:  ['pill-done',   'Passed'],
};

export function StatusPill({ status, label, spin = false }) {
  const [cls, defaultLabel] = STATUS_MAP[status] || ['pill-idle', status];
  return (
    <span className={`pill ${cls}`}>
      {status === 'running' && spin
        ? <span className="spinner" style={{ width: 9, height: 9, borderWidth: 1.6 }} />
        : <span className="dot" />}
      {label || defaultLabel}
    </span>
  );
}

export default StatusPill;
