import React from 'react';
import { Icons } from '../icons.jsx';

const NAV = [
  { key: 'dashboard',   label: 'Dashboard',   icon: 'grid' },
  { key: 'upload',      label: 'New Run',      icon: 'upload' },
  { key: 'pipeline',    label: 'Pipeline',     icon: 'flow' },
  { key: 'leaderboard', label: 'Leaderboard',  icon: 'trophy' },
];

export function Sidebar({ route, go, runState }) {
  return (
    <aside style={{
      background: '#fff', borderRight: '1px solid var(--line)',
      display: 'flex', flexDirection: 'column', padding: '20px 14px', gap: 4, height: '100%',
    }}>
      {/* brand */}
      <div className="row gap-10" style={{ padding: '4px 8px 18px' }}>
        <div style={{
          width: 34, height: 34, borderRadius: 10, flex: 'none',
          background: 'linear-gradient(150deg, var(--accent), var(--accent-strong))',
          display: 'grid', placeItems: 'center', color: '#fff',
          boxShadow: '0 4px 12px rgba(108,71,255,.35)',
        }}>
          <Icons.layers size={19} sw={1.9} />
        </div>
        <div className="col" style={{ lineHeight: 1.1 }}>
          <div style={{ fontWeight: 800, fontSize: 16, letterSpacing: '-0.03em' }}>
            MITRA<span style={{ color: 'var(--accent)' }}> AI</span>
          </div>
          <div className="mono" style={{ fontSize: 9.5, color: 'var(--ink-3)', letterSpacing: '.02em', marginTop: 2 }}>
            AGENTIC AUTOML
          </div>
        </div>
      </div>

      <div className="mono" style={{ fontSize: 10, color: 'var(--ink-4)', letterSpacing: '.08em', padding: '0 10px 6px' }}>
        WORKSPACE
      </div>

      {NAV.map(navItem => {
        const IconComponent = Icons[navItem.icon];
        const isActive = route === navItem.key;
        const showBadge = navItem.key === 'pipeline' && runState === 'running';
        const showDone  = navItem.key === 'pipeline' && runState === 'done';
        return (
          <button
            key={navItem.key}
            onClick={() => go(navItem.key)}
            className={`nav-item focusable ${isActive ? 'active' : ''}`}
          >
            <IconComponent size={18} sw={isActive ? 1.9 : 1.7} className="nav-ic" />
            {navItem.label}
            {showBadge && <span className="spinner" style={{ marginLeft: 'auto', width: 11, height: 11 }} />}
            {showDone  && <Icons.checkCircle size={15} style={{ marginLeft: 'auto', color: 'var(--ok)' }} />}
          </button>
        );
      })}

      <div style={{ flex: 1 }} />

      <button
        onClick={() => go('settings')}
        className={`nav-item focusable ${route === 'settings' ? 'active' : ''}`}
        style={{ marginBottom: 6 }}
      >
        <Icons.gear size={18} sw={route === 'settings' ? 1.9 : 1.7} className="nav-ic" />
        Settings
      </button>

      <div className="row gap-10" style={{ padding: '12px 8px 0', borderTop: '1px solid var(--line)' }}>
        <div style={{
          width: 30, height: 30, borderRadius: 99, background: 'var(--panel-3)',
          border: '1px solid var(--line-3)', display: 'grid', placeItems: 'center',
          fontWeight: 700, fontSize: 12, color: 'var(--ink-2)',
        }}>
          M
        </div>
        <div className="col" style={{ lineHeight: 1.2 }}>
          <div style={{ fontSize: 12.5, fontWeight: 600 }}>Course Team</div>
          <div style={{ fontSize: 11, color: 'var(--ink-3)' }}>Self-hosted · local</div>
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;
