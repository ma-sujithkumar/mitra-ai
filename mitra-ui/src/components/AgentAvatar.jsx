import React from 'react';
import { Icons } from '../icons.jsx';

function agentColor(hue) {
  return {
    fg:    `oklch(0.55 0.16 ${hue})`,
    bg:    `oklch(0.96 0.045 ${hue})`,
    line:  `oklch(0.90 0.07 ${hue})`,
    solid: `oklch(0.62 0.17 ${hue})`,
  };
}

export function AgentAvatar({ agent, size = 34, state = 'idle' }) {
  const colors = agentColor(agent.hue);
  const isRunning = state === 'running';
  const isDone = state === 'done';
  return (
    <div style={{
      width: size, height: size, borderRadius: size * 0.3,
      display: 'grid', placeItems: 'center', flex: 'none',
      background: isDone ? colors.solid : colors.bg,
      border: `1px solid ${isDone ? colors.solid : colors.line}`,
      color: isDone ? '#fff' : colors.fg,
      fontFamily: 'var(--mono)', fontWeight: 700, fontSize: size * 0.36,
      letterSpacing: '-0.02em', position: 'relative',
      animation: isRunning ? 'pulse-ring 1.6s infinite' : 'none',
      transition: 'background .3s, color .3s, border-color .3s',
    }}>
      {isDone ? <Icons.check size={size * 0.5} sw={2.4} /> : agent.short}
    </div>
  );
}

export default AgentAvatar;
