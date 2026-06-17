import StatusPill from '../StatusPill.jsx';
import { Icons } from '../../icons.jsx';

function formatScore(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return '—';
  }
  return numeric.toFixed(4);
}

function formatDuration(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return '—';
  }
  return `${numeric.toFixed(2)}s`;
}

function ModelTrainingCard({ model, selected, onSelect }) {
  const failed = ['failed', 'timed_out', 'cancelled'].includes(model.status);
  const completed = model.status === 'completed';

  return (
    <button
      className={`training-model-card ${selected ? 'active' : ''} ${failed ? 'failed' : ''}`}
      onClick={() => onSelect(model.modelId)}
      type="button"
    >
      <div className="training-model-head">
        <span className="training-priority mono">#{model.priority}</span>
        <div>
          <strong>{model.modelName}</strong>
          <small className="mono">{model.modelId}</small>
        </div>
        <StatusPill
          spin={model.status === 'running'}
          status={model.status}
        />
      </div>

      <p>{model.rationale}</p>

      <div className="training-card-progress">
        <div className="bar">
          <i style={{ width: `${model.pct}%` }} />
        </div>
        <span className="mono">{model.pct}%</span>
      </div>

      <div className="training-model-metrics">
        <span>
          <small>Validation</small>
          <strong className="mono">{formatScore(model.details.validation_score)}</strong>
        </span>
        <span>
          <small>Duration</small>
          <strong className="mono">{formatDuration(model.details.training_time_sec)}</strong>
        </span>
      </div>

      {completed && model.details.model_path ? (
        <div className="training-artifact">
          <Icons.checkCircle size={15} />
          <span className="mono" title={model.details.model_path}>{model.details.model_path}</span>
        </div>
      ) : null}

      {failed ? (
        <div className="training-error">
          <Icons.alert size={15} />
          <span>{model.details.error || model.message || 'Training failed.'}</span>
        </div>
      ) : (
        <div className="training-last-message">{model.message || 'Waiting for training events.'}</div>
      )}
    </button>
  );
}

export default ModelTrainingCard;
