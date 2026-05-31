import type { AgentStatus } from "@/lib/mitra-mock";

const MAP: Record<AgentStatus, { color: string; label: string; ring: boolean }> = {
  idle:    { color: "bg-muted-foreground/50", label: "idle",    ring: false },
  running: { color: "bg-primary",             label: "running", ring: true  },
  ok:      { color: "bg-success",             label: "ok",      ring: false },
  warn:    { color: "bg-warning",             label: "warn",    ring: false },
  error:   { color: "bg-destructive",         label: "error",   ring: false },
};

export function StatusDot({ status, label }: { status: AgentStatus; label?: boolean }) {
  const cfg = MAP[status];
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={`status-dot ${cfg.color} ${cfg.ring ? "pulse-ring" : ""}`} />
      {label ? (
        <span className="mono text-[10px] uppercase tracking-wider text-muted-foreground">
          {cfg.label}
        </span>
      ) : null}
    </span>
  );
}
