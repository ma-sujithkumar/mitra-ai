import { Icons } from '../icons.jsx';

// Horizontal workflow stepper that shows where the user is in the run lifecycle
// and lets them jump to enabled stages. Each step has a status:
// 'done' | 'active' | 'queued'. Steps marked enabled are clickable.
function Stepper({ steps, onNavigate }) {
  return (
    <nav className="stepper" aria-label="Run progress">
      <ol className="stepper-list">
        {steps.map((step, index) => {
          const isClickable = step.enabled && typeof onNavigate === 'function';
          return (
            <li className={`stepper-item ${step.status}`} key={step.key}>
              <button
                type="button"
                className="stepper-node"
                aria-current={step.status === 'active' ? 'step' : undefined}
                disabled={!isClickable}
                onClick={() => isClickable && onNavigate(step)}
              >
                <span className="stepper-index">
                  {step.status === 'done' ? <Icons.check size={14} /> : index + 1}
                </span>
                <span className="stepper-label">{step.label}</span>
              </button>
              {index < steps.length - 1 ? (
                <span className="stepper-sep" aria-hidden="true" />
              ) : null}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

export default Stepper;
