import { useEffect, useRef } from 'react';

function formatTime(timestamp) {
  if (!timestamp) {
    return '--:--:--';
  }
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return '--:--:--';
  }
  return date.toLocaleTimeString([], { hour12: false });
}

function TrainingLogs({ logs, selectedModelId, onClearFilter }) {
  const logRef = useRef(null);
  const visibleLogs = selectedModelId
    ? logs.filter((entry) => entry.modelId === selectedModelId || !entry.modelId)
    : logs;

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [visibleLogs.length]);

  return (
    <section className="card terminal-panel training-log-panel">
      <div className="terminal-head">
        <span>SSE training event stream</span>
        <button className="terminal-filter" onClick={onClearFilter} type="button">
          {selectedModelId ? `filter: ${selectedModelId}` : 'all models'}
        </button>
      </div>
      <div className="terminal-body training-log-body" ref={logRef}>
        {visibleLogs.length ? visibleLogs.map((entry, index) => (
          <div className="terminal-line training-log-line" key={`${entry.sequence}-${index}`}>
            <span>{formatTime(entry.ts)}</span>
            <strong className={`level-${entry.level}`}>{entry.status}</strong>
            <em>{entry.modelName ? `${entry.modelName}: ` : ''}{entry.message}</em>
          </div>
        )) : (
          <span className="terminal-empty">awaiting training events</span>
        )}
      </div>
    </section>
  );
}

export default TrainingLogs;
