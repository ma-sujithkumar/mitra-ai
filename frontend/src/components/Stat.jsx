import { Icons } from '../icons.jsx';

function Stat({ icon, label, value, unit, delta, accent = false }) {
  const Icon = icon ? Icons[icon] : null;

  return (
    <div className="card stat-card">
      <div className="stat-head">
        <span>{label}</span>
        {Icon ? (
          <div className={`stat-icon ${accent ? 'accent' : ''}`}>
            <Icon size={16} />
          </div>
        ) : null}
      </div>
      <div className="stat-value">
        <span className="mono">{value}</span>
        {unit ? <small>{unit}</small> : null}
        {delta ? <em>{delta}</em> : null}
      </div>
    </div>
  );
}

export default Stat;
