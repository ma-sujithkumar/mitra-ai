import React from 'react';
import { Icons } from '../icons.jsx';

export function Stat({ icon, label, value, unit, delta, accent }) {
  const IconComponent = Icons[icon];
  return (
    <div className="card" style={{ padding: '16px 18px', display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div className="row gap-8" style={{ justifyContent: 'space-between' }}>
        <span style={{ fontSize: 12.5, color: 'var(--ink-2)', fontWeight: 550 }}>{label}</span>
        {IconComponent && (
          <div style={{
            width: 28, height: 28, borderRadius: 8,
            background: accent ? 'var(--accent-soft)' : 'var(--panel-3)',
            display: 'grid', placeItems: 'center',
            color: accent ? 'var(--accent)' : 'var(--ink-3)',
          }}>
            <IconComponent size={16} />
          </div>
        )}
      </div>
      <div className="row" style={{ alignItems: 'baseline', gap: 5 }}>
        <span style={{ fontSize: 27, fontWeight: 750, letterSpacing: '-0.03em' }} className="mono">
          {value !== null && value !== undefined ? value : '—'}
        </span>
        {unit && <span style={{ fontSize: 13, color: 'var(--ink-3)', fontWeight: 600 }}>{unit}</span>}
        {delta && <span style={{ marginLeft: 'auto', fontSize: 11.5, fontWeight: 650, color: 'var(--ok)' }}>{delta}</span>}
      </div>
    </div>
  );
}

export default Stat;
