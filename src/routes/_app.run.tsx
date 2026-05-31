import { createFileRoute, Link } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { PageHeader } from "@/components/mitra/AppShell";
import { StatusDot } from "@/components/mitra/StatusDot";
import { AGENTS, SAMPLE_EVENTS, type AgentStatus, type PipelineEvent } from "@/lib/mitra-mock";
import { Pause, Play, Square } from "lucide-react";

export const Route = createFileRoute("/_app/run")({
  head: () => ({
    meta: [
      { title: "Live run — MITRA v2" },
      { name: "description", content: "Real-time SSE stream of an agentic AutoML run." },
    ],
  }),
  component: RunPage,
});

function RunPage() {
  const [events, setEvents] = useState<PipelineEvent[]>([SAMPLE_EVENTS[0]]);
  const [running, setRunning] = useState(true);
  const idx = useRef(1);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      if (idx.current >= SAMPLE_EVENTS.length) { setRunning(false); return; }
      setEvents((e) => [...e, SAMPLE_EVENTS[idx.current++]]);
    }, 900);
    return () => clearInterval(id);
  }, [running]);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: "smooth" });
  }, [events.length]);

  const agentStatus = computeStatus(events);
  const completedStages = Math.max(...Object.entries(agentStatus).map(([id, s]) =>
    s === "ok" ? (AGENTS.find((a) => a.id === id)!.stage) : 0
  ), 0);

  return (
    <>
      <PageHeader
        eyebrow="page 2 · live run"
        title="mit_8f2a · churn_q3.csv"
        description="SSE event bus · 1 asyncio.Queue per session · typed event schema."
        actions={
          <div className="flex items-center gap-2">
            <span className="mono text-[10px] text-muted-foreground">
              stage {completedStages}/7
            </span>
            <button
              onClick={() => setRunning((r) => !r)}
              className="inline-flex items-center gap-1.5 rounded-md border border-border bg-background px-2.5 py-1.5 text-[12px] hover:bg-accent"
            >
              {running ? <><Pause className="h-3 w-3" /> Pause</> : <><Play className="h-3 w-3" /> Resume</>}
            </button>
            <button className="inline-flex items-center gap-1.5 rounded-md border border-destructive/30 bg-destructive/10 text-destructive px-2.5 py-1.5 text-[12px] hover:bg-destructive/20">
              <Square className="h-3 w-3" /> Abort
            </button>
          </div>
        }
      />

      <div className="px-6 lg:px-10 py-6 grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div className="xl:col-span-1 rounded-lg border border-border bg-surface">
          <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
            <h2 className="text-[12px] font-medium">Agent pipeline</h2>
            <span className="mono text-[10px] text-muted-foreground">8 agents · 7 stages</span>
          </div>
          <ol className="p-2">
            {AGENTS.map((a, i) => {
              const status = agentStatus[a.id] ?? "idle";
              return (
                <li key={a.id} className="relative">
                  {i < AGENTS.length - 1 ? (
                    <span className="absolute left-[19px] top-9 bottom-0 w-px bg-border" />
                  ) : null}
                  <div className="flex items-start gap-3 rounded-md px-2 py-2 hover:bg-accent/40">
                    <div className="relative mt-1 h-4 w-4 flex items-center justify-center">
                      <StatusDot status={status} />
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center justify-between">
                        <span className="text-[13px] font-medium">{a.name}</span>
                        <span className="mono text-[10px] text-muted-foreground">S{a.stage}</span>
                      </div>
                      <p className="mono text-[10px] text-muted-foreground truncate">
                        {a.writes.join(", ")}
                      </p>
                      {status === "running" ? (
                        <div className="mt-2 h-0.5 rounded-full bg-border overflow-hidden">
                          <div className="h-full w-1/3 shimmer bg-primary/40" />
                        </div>
                      ) : null}
                    </div>
                  </div>
                </li>
              );
            })}
          </ol>
        </div>

        <div className="xl:col-span-2 rounded-lg border border-border bg-surface flex flex-col min-h-[520px]">
          <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
            <h2 className="text-[12px] font-medium flex items-center gap-2">
              <span className="status-dot bg-primary pulse-ring" /> SSE stream
            </h2>
            <span className="mono text-[10px] text-muted-foreground">
              {events.length} events
            </span>
          </div>
          <div ref={logRef} className="flex-1 overflow-y-auto px-4 py-3 mono text-[11px] leading-relaxed">
            {events.map((e, i) => (
              <div key={i} className="flex gap-3 py-0.5 hover:bg-accent/30 -mx-2 px-2 rounded">
                <span className="text-muted-foreground/60 w-14 shrink-0">{e.t}</span>
                <span className={`w-3 shrink-0 ${levelColor(e.level)}`}>{levelGlyph(e.level)}</span>
                <span className="text-muted-foreground w-32 shrink-0 truncate">{e.agent}</span>
                <span className="text-foreground/90">{e.msg}</span>
              </div>
            ))}
            {running ? (
              <div className="flex gap-3 py-1 text-muted-foreground">
                <span className="w-14" />
                <span className="inline-block h-2 w-2 rounded-full bg-primary animate-pulse" />
                <span>awaiting next event…</span>
              </div>
            ) : (
              <div className="mt-2 rounded-md border border-success/30 bg-success/10 px-3 py-2 text-success">
                ✓ run complete · <Link to="/results" className="underline">view results →</Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

function computeStatus(events: PipelineEvent[]): Record<string, AgentStatus> {
  const map: Record<string, AgentStatus> = {};
  for (const a of AGENTS) map[a.id] = "idle";
  for (const e of events) {
    if (e.agent === "orchestrator") continue;
    if (e.level === "ok") map[e.agent] = "ok";
    else if (e.level === "warn" && map[e.agent] !== "ok") map[e.agent] = "warn";
    else if (e.level === "error") map[e.agent] = "error";
    else if (map[e.agent] === "idle") map[e.agent] = "running";
  }
  // any agent past the last "running" stays idle/running appropriately
  return map;
}

function levelColor(l: string) {
  return l === "ok" ? "text-success" : l === "warn" ? "text-warning" : l === "error" ? "text-destructive" : "text-info";
}
function levelGlyph(l: string) {
  return l === "ok" ? "✓" : l === "warn" ? "!" : l === "error" ? "✕" : "·";
}
