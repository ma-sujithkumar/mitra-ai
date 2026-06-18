import { Icons } from '../icons.jsx';

export function agentColor(hue) {
  return {
    fg: `oklch(0.55 0.16 ${hue})`,
    bg: `oklch(0.96 0.045 ${hue})`,
    line: `oklch(0.90 0.07 ${hue})`,
    solid: `oklch(0.62 0.17 ${hue})`,
  };
}

function AgentAvatar({ agent, size = 34, state = 'idle' }) {
  const colors = agentColor(agent.hue);
  const Check = Icons.check;
  const isRunning = state === 'running';
  const isDone = state === 'done';

  return (
    <div
      className={`agent-avatar ${isRunning ? 'running' : ''}`}
      style={{
        '--agent-bg': isDone ? colors.solid : colors.bg,
        '--agent-border': isDone ? colors.solid : colors.line,
        '--agent-ink': isDone ? '#fff' : colors.fg,
        width: size,
        height: size,
        borderRadius: size * 0.3,
        fontSize: size * 0.36,
      }}
    >
      {isDone ? <Check size={size * 0.5} strokeWidth={2.4} /> : agent.short}
    </div>
  );
}

export default AgentAvatar;
