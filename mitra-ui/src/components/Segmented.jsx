import React from 'react';

export function Segmented({ options, value, onChange }) {
  return (
    <div className="row" style={{
      background: 'var(--panel-3)', border: '1px solid var(--line-3)',
      borderRadius: 9, padding: 3, gap: 2,
    }}>
      {options.map(option => {
        const isActive = option.value === value;
        return (
          <button
            key={option.value}
            onClick={() => onChange(option.value)}
            style={{
              border: 'none', cursor: 'pointer', padding: '5px 12px', borderRadius: 6,
              fontSize: 12.5, fontWeight: 600, fontFamily: 'var(--sans)',
              background: isActive ? '#fff' : 'transparent',
              color: isActive ? 'var(--ink)' : 'var(--ink-2)',
              boxShadow: isActive ? 'var(--sh-sm)' : 'none',
              transition: 'all .14s',
            }}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

export default Segmented;
