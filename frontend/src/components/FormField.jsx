function FormField({ label, hint, children }) {
  return (
    <label className="form-field">
      <span className="form-field-label">
        <span>{label}</span>
        {hint ? <span className="form-field-hint">{hint}</span> : null}
      </span>
      {children}
    </label>
  );
}

export default FormField;
