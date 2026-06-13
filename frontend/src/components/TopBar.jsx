import { Icons } from '../icons.jsx';

function TopBar({ title, sub, right, icon }) {
  const Icon = icon ? Icons[icon] : null;

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
      <div className="topbar-actions">{right}</div>
    </header>
  );
}

export default TopBar;
