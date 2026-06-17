import StatusPill from '../StatusPill.jsx';

function TrainingProgress({ counts, progress, connectionStatus }) {
  const finished = counts.total > 0 && counts.completed + counts.failed === counts.total;
  const status = finished ? 'done' : counts.running > 0 ? 'running' : 'queued';

  return (
    <section className="card training-overview">
      <div>
        <p className="section-kicker">Page 2 · Ray Training</p>
        <h2>Parallel model training</h2>
        <p className="muted">
          Live lifecycle updates from the Epic-3 SSE stream.
        </p>
      </div>
      <div className="training-overview-actions">
        <StatusPill
          label={connectionStatus === 'open' ? 'Live' : connectionStatus}
          spin={connectionStatus === 'connecting' || connectionStatus === 'reconnecting'}
          status={connectionStatus === 'open' ? status : connectionStatus === 'error' ? 'failed' : ['connecting', 'reconnecting'].includes(connectionStatus) ? 'running' : 'queued'}
        />
        <span className="mono progress-value">{progress}%</span>
      </div>
      <div className="bar training-overview-bar">
        <i style={{ width: `${progress}%` }} />
      </div>
      <div className="training-count-grid">
        <span><strong>{counts.total}</strong><small>selected</small></span>
        <span><strong>{counts.running}</strong><small>active</small></span>
        <span><strong>{counts.completed}</strong><small>completed</small></span>
        <span><strong>{counts.failed}</strong><small>failed</small></span>
      </div>
    </section>
  );
}

export default TrainingProgress;
