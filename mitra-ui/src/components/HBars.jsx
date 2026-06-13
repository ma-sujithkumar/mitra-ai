import React from 'react';

export function HBars({ data, max, fmt = (value) => value.toFixed(3), color = 'var(--accent)' }) {
  const maxVal = max || Math.max(...data.map(item => item.value));
  return (
    <div className="col gap-12">
      {data.map((item, index) => (
        <div key={item.feature} className="col gap-6">
          <div className="row" style={{ justifyContent: 'space-between', fontSize: 12.5 }}>
            <span className="mono" style={{ color: 'var(--ink-2)' }}>{item.feature}</span>
            <span className="mono" style={{ fontWeight: 600 }}>{fmt(item.value)}</span>
          </div>
          <div className="bar" style={{ height: 9 }}>
            <i style={{
              width: `${(item.value / maxVal) * 100}%`,
              background: color,
              animation: `fadeUp .5s ${index * 0.08}s both`,
            }} />
          </div>
        </div>
      ))}
    </div>
  );
}

export default HBars;
