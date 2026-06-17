import { NAV_ITEMS } from '../data.js';
import { Icons } from '../icons.jsx';

function Sidebar({ route, go, runState = 'idle' }) {
  return (
    <aside className="sidebar">
      <button className="brand-button" onClick={() => go('dashboard')} type="button">
        <div className="brand-mark">
          <Icons.layers size={19} strokeWidth={1.9} />
        </div>
        <div className="brand-copy">
          <div className="brand-name">MITRA AI</div>
          <div className="brand-subtitle">AGENTIC AUTOML</div>
        </div>
      </button>

      <div className="sidebar-label">Workspace</div>
      <nav className="nav-list" aria-label="Main navigation">
        {NAV_ITEMS.map((navItem) => {
          const Icon = Icons[navItem.icon];
          const active = route === navItem.key;
          const showRunning = navItem.key === 'pipeline' && runState === 'running';
          const showDone = navItem.key === 'pipeline' && runState === 'done';

          return (
            <button
              className={`nav-item ${active ? 'active' : ''}`}
              key={navItem.key}
              onClick={() => go(navItem.key)}
              type="button"
            >
              <Icon className="nav-icon" size={18} strokeWidth={active ? 1.9 : 1.7} />
              <span>{navItem.label}</span>
              {showRunning ? <span className="spinner" /> : null}
              {showDone ? <Icons.checkCircle className="nav-state" size={15} /> : null}
            </button>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="user-chip">
          <div className="user-avatar">M</div>
          <div>
            <div className="user-name">Course Team</div>
            <div className="user-subtitle">Local workspace</div>
          </div>
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;
