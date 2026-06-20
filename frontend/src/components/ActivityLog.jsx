import { useEffect, useMemo, useRef, useState } from 'react';

import { activityLogDownloadUrl } from '../api/client.js';
import { Icons } from '../icons.jsx';

const LEVEL_FILTERS = ['all', 'info', 'warning', 'error'];

function formatTime(timestamp) {
  if (!timestamp) {
    return '--:--:--';
  }
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    // Backend timestamps are local ISO without timezone; show the time portion.
    const timePart = String(timestamp).split('T')[1];
    return timePart || timestamp;
  }
  return date.toLocaleTimeString([], { hour12: false });
}

// Collapsible activity feed shown on the Upload/Training screens. Renders a
// merged, chronological list of activity entries ({timestamp, level, stage,
// message}) so the user can see exactly what happened, filter by level, copy
// the text, or download the raw .log for the run.
function ActivityLog({ entries, sessionId, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  const [levelFilter, setLevelFilter] = useState('all');
  const bodyRef = useRef(null);

  const visibleEntries = useMemo(() => {
    if (levelFilter === 'all') {
      return entries;
    }
    return entries.filter(
      (entry) => (entry.level || 'info').toLowerCase() === levelFilter,
    );
  }, [entries, levelFilter]);

  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [visibleEntries.length]);

  function copyEntries() {
    const text = entries
      .map(
        (entry) =>
          `${entry.timestamp || ''} ${(entry.level || 'INFO').toUpperCase()} ${entry.stage || ''} ${entry.message || ''}`,
      )
      .join('\n');
    navigator.clipboard?.writeText(text);
  }

  return (
    <section className="card activity-log" aria-label="Activity log">
      <div className="activity-head">
        <button
          type="button"
          className="activity-toggle"
          aria-expanded={open}
          onClick={() => setOpen((value) => !value)}
        >
          <Icons.activity size={16} />
          <span>Activity log</span>
          <small>{entries.length} entries</small>
        </button>
        <div className="activity-actions">
          <label className="activity-filter">
            <Icons.filter size={13} />
            <select
              aria-label="Filter activity by level"
              value={levelFilter}
              onChange={(event) => setLevelFilter(event.target.value)}
            >
              {LEVEL_FILTERS.map((level) => (
                <option key={level} value={level}>
                  {level}
                </option>
              ))}
            </select>
          </label>
          <button className="btn btn-ghost btn-sm" onClick={copyEntries} type="button">
            Copy
          </button>
          {sessionId ? (
            <a
              className="btn btn-ghost btn-sm"
              href={activityLogDownloadUrl(sessionId)}
              download
            >
              <Icons.download size={13} /> .log
            </a>
          ) : null}
        </div>
      </div>

      {open ? (
        <div className="activity-body" ref={bodyRef} role="log" aria-live="polite">
          {visibleEntries.length ? (
            visibleEntries.map((entry, index) => (
              <div
                className={`activity-line level-${(entry.level || 'info').toLowerCase()}`}
                key={`${entry.timestamp}-${index}`}
              >
                <span className="activity-time">{formatTime(entry.timestamp)}</span>
                <span className="activity-level">{(entry.level || 'info').toUpperCase()}</span>
                <span className="activity-stage">{entry.stage}</span>
                <span className="activity-msg">{entry.message}</span>
              </div>
            ))
          ) : (
            <span className="activity-empty">No activity yet.</span>
          )}
        </div>
      ) : null}
    </section>
  );
}

export default ActivityLog;
