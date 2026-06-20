import StatusPill from '../StatusPill.jsx';
import { Icons } from '../../icons.jsx';

function TrainingSummary({ summary, canContinue, onContinue, judgePending }) {
  if (!summary) {
    return (
      <section className="card panel-section">
        <div className="section-head">
          <div>
            <p className="section-kicker">Session Summary</p>
            <h2>Waiting for completion</h2>
          </div>
          <StatusPill status="queued" />
        </div>
        <p className="muted">The final summary appears after all Ray jobs reach a terminal state.</p>
      </section>
    );
  }

  const successful = summary.completed > 0;
  return (
    <section className="card panel-section">
      <div className="section-head">
        <div>
          <p className="section-kicker">Session Summary</p>
          <h2>{summary.message}</h2>
        </div>
        <StatusPill status={summary.failed ? 'warn' : 'done'} label={summary.status} />
      </div>
      <div className="summary-strip training-summary-strip">
        <span>{summary.total} models</span>
        <span>{summary.completed} completed</span>
        <span>{summary.failed} failed</span>
      </div>
      <button
        className="btn btn-primary full-width"
        disabled={!canContinue || !successful || judgePending}
        onClick={onContinue}
        type="button"
      >
        {judgePending ? (
          <>
            <span className="spinner" style={{ marginRight: 8 }} />
            Judge Agent is evaluating...
          </>
        ) : (
          <>
            <Icons.trophy size={16} />
            Continue to leaderboard
          </>
        )}
      </button>
    </section>
  );
}

export default TrainingSummary;
