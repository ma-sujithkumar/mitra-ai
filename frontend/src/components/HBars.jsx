function HBars({ data, max, formatValue = (value) => value.toFixed(3), color = 'var(--accent)' }) {
  const maxValue = max || Math.max(...data.map((item) => item.value));

  return (
    <div className="hbars">
      {data.map((item, index) => (
        <div className="hbar-row" key={item.feature}>
          <div className="hbar-labels">
            <span className="mono">{item.feature}</span>
            <span className="mono">{formatValue(item.value)}</span>
          </div>
          <div className="bar">
            <i
              style={{
                '--bar-color': color,
                '--bar-delay': `${index * 0.08}s`,
                width: `${(item.value / maxValue) * 100}%`,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export default HBars;
