import React from 'react';
import { Icons } from '../icons.jsx';

export function TopBar({ title, sub, icon, right }) {
  const IconComponent = icon ? Icons[icon] : null;
  return (
    <header style={{
      height: 64, flex: 'none', borderBottom: '1px solid var(--line)',
      background: 'rgba(255,255,255,0.8)', backdropFilter: 'blur(10px)',
      display: 'flex', alignItems: 'center', padding: '0 28px', gap: 14,
      position: 'sticky', top: 0, zIndex: 5,
    }}>
      {IconComponent && (
        <div style={{
          width: 32, height: 32, borderRadius: 9, background: 'var(--accent-soft)',
          display: 'grid', placeItems: 'center', color: 'var(--accent)', flex: 'none',
        }}>
          <IconComponent size={18} />
        </div>
      )}
      <div className="col" style={{ lineHeight: 1.25 }}>
        <div style={{ fontWeight: 700, fontSize: 16, letterSpacing: '-0.02em' }}>{title}</div>
        {sub && <div style={{ fontSize: 12, color: 'var(--ink-3)' }}>{sub}</div>}
      </div>
      <div className="row gap-10" style={{ marginLeft: 'auto' }}>{right}</div>
    </header>
  );
}

export default TopBar;
