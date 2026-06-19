import { Icons } from '../icons.jsx';

// Ordered stages shown in the metadata loader. Keys are stable and map to the
// backend progress event "step" values (see routers/metadata.py).
const STEP_DEFINITIONS = [
  { key: 'reading_data', label: 'Reading dataset sample', icon: Icons.database },
  { key: 'inferring_schema', label: 'Inferring schema', icon: Icons.brain },
  { key: 'writing', label: 'Writing metadata.json', icon: Icons.doc },
];

// Map each incoming progress event "step" to the stage index it activates. Steps
// not listed here (e.g. llm_settings_resolved) do not advance the stepper.
const STEP_INDEX_BY_EVENT = {
  reading_data: 0,
  inferring_schema: 1,
};

function deriveActiveIndex(events, phase) {
  const isComplete = phase === 'done' || events.some((event) => event.type === 'done');
  if (isComplete) {
    return STEP_DEFINITIONS.length;
  }
  let activeIndex = phase === 'running' ? 0 : -1;
  for (const event of events) {
    if (event.step in STEP_INDEX_BY_EVENT) {
      activeIndex = Math.max(activeIndex, STEP_INDEX_BY_EVENT[event.step]);
    }
  }
  return activeIndex;
}

function stepStatus(stepIndex, activeIndex, phase) {
  if (stepIndex < activeIndex) {
    return 'done';
  }
  if (stepIndex === activeIndex) {
    return phase === 'error' ? 'error' : 'active';
  }
  return 'queued';
}

function StepIcon({ status }) {
  if (status === 'done') {
    return <Icons.checkCircle size={16} />;
  }
  if (status === 'active') {
    return <span className="spinner small" />;
  }
  if (status === 'error') {
    return <Icons.alert size={16} />;
  }
  return <Icons.dot size={12} />;
}

function MetadataProgress({ phase, events, llm, errorMessage }) {
  if (phase === 'idle') {
    return <p className="muted">Metadata starts automatically after validation passes.</p>;
  }

  const activeIndex = deriveActiveIndex(events, phase);
  const modelLabel = llm?.provider && llm?.model ? `${llm.provider}/${llm.model}` : null;

  return (
    <div className="metadata-progress">
      <div className="metadata-steps">
        {STEP_DEFINITIONS.map((stepDefinition, stepIndex) => {
          const status = stepStatus(stepIndex, activeIndex, phase);
          const showModel = stepDefinition.key === 'inferring_schema' && modelLabel;
          return (
            <div className={`metadata-step ${status}`} key={stepDefinition.key}>
              <span className="metadata-step-icon">
                <StepIcon status={status} />
              </span>
              <span className="metadata-step-label">
                {stepDefinition.label}
                {showModel ? <small className="mono"> {modelLabel}</small> : null}
              </span>
            </div>
          );
        })}
      </div>

      {phase === 'running' ? (
        <>
          <div className="progress-bar indeterminate">
            <span />
          </div>
          <p className="muted">
            {modelLabel
              ? `Generating metadata with ${modelLabel}... this can take ~20-60s.`
              : 'Generating metadata... this can take ~20-60s.'}
          </p>
        </>
      ) : null}

      {phase === 'done' ? (
        <p className="muted">Metadata generated successfully.</p>
      ) : null}

      {phase === 'error' ? (
        <div className="callout error compact">
          <strong>Metadata generation failed</strong>
          <span>{errorMessage || 'See logs for details.'}</span>
        </div>
      ) : null}
    </div>
  );
}

export default MetadataProgress;
