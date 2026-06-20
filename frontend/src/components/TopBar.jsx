import { Icons } from '../icons.jsx';

function TopBar({ title, sub, right, icon, darkMode, onToggleDark }) {
  const Icon = icon ? Icons[icon] : null;
  const ThemeIcon = darkMode ? Icons.sun : Icons.moon;

  return (
    <header className="topbar">
      {Icon ? (
        <div className="topbar-icon">
          <Icon size={18} />
        </div>
      ) : null}
      <div className="topbar-copy">
        <h1>{title}</h1>
        {sub ? <p>{sub}</p> : null}
      </div>
      <div className="topbar-actions">
        {onToggleDark ? (
          <button
            aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
            className="btn btn-secondary btn-sm"
            onClick={onToggleDark}
            type="button"
          >
            <ThemeIcon size={15} />
          </button>
        ) : null}
        {right}
      </div>
    </header>
  );
}

export default TopBar;
