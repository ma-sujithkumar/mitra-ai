import React from 'react';

export const FIELD_INPUT_STYLE = {
  width: '100%', padding: '10px 12px', borderRadius: 9,
  border: '1px solid var(--line-3)', fontSize: 13,
  fontFamily: 'var(--sans)', color: 'var(--ink)',
  background: '#fff', outline: 'none',
};

export function FormField({ label, hint, children }) {
  return (
    <div className="col gap-8" style={{ marginBottom: 16 }}>
      <div className="row" style={{ justifyContent: 'space-between' }}>
        <label style={{ fontSize: 12.5, fontWeight: 600 }}>{label}</label>
        {hint && <span className="faint" style={{ fontSize: 11 }}>{hint}</span>}
      </div>
      {children}
    </div>
  );
}

export default FormField;
