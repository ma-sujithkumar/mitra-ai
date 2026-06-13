import React from 'react';

export function Sparkline({ points, w = 120, h = 34, color = 'var(--accent)' }) {
  const maxVal = Math.max(...points);
  const minVal = Math.min(...points);
  const range = maxVal - minVal || 1;
  const step = w / (points.length - 1);
  const pts = points
    .map((point, index) => `${index * step},${h - 2 - ((point - minVal) / range) * (h - 4)}`)
    .join(' ');
  return (
    <svg width={w} height={h} style={{ display: 'block', overflow: 'visible' }}>
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle
        cx={(points.length - 1) * step}
        cy={h - 2 - ((points[points.length - 1] - minVal) / range) * (h - 4)}
        r="3"
        fill={color}
      />
    </svg>
  );
}

export default Sparkline;
