import Tooltip from './Tooltip.jsx';

// Labelled form control. `hint` is a short inline note; `tooltip` adds an info
// affordance with longer help; `recommended` renders a small badge so users
// know the suggested value at a glance.
function FormField({ label, hint, tooltip, recommended, children }) {
  return (
    <label className="form-field">
      <span className="form-field-label">
        <span className="form-field-label-text">
          {label}
          {tooltip ? <Tooltip text={tooltip} label={`About ${label}`} /> : null}
        </span>
        {recommended !== undefined && recommended !== null ? (
          <span className="form-field-recommended">recommended {String(recommended)}</span>
        ) : null}
        {hint ? <span className="form-field-hint">{hint}</span> : null}
      </span>
      {children}
    </label>
  );
}

export default FormField;
