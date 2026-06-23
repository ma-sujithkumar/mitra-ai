function Sparkline({ points, width = 120, height = 34, color = 'var(--accent)' }) {
  const maxPoint = Math.max(...points);
  const minPoint = Math.min(...points);
  const range = maxPoint - minPoint || 1;
  const step = width / (points.length - 1);
  const polylinePoints = points
    .map((point, index) => {
      const x = index * step;
      const y = height - 2 - ((point - minPoint) / range) * (height - 4);
      return `${x},${y}`;
    })
    .join(' ');
  const lastY = height - 2 - ((points[points.length - 1] - minPoint) / range) * (height - 4);

  return (
    <svg className="sparkline" height={height} width={width}>
      <polyline
        fill="none"
        points={polylinePoints}
        stroke={color}
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
      />
      <circle cx={(points.length - 1) * step} cy={lastY} fill={color} r="3" />
    </svg>
  );
}

export default Sparkline;
