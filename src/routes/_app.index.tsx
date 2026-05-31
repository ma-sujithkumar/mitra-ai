import { createFileRoute, Link } from "@tanstack/react-router";
import { PageHeader } from "@/components/mitra/AppShell";
import { StatusDot } from "@/components/mitra/StatusDot";
import { AGENTS, LEADERBOARD, SAMPLE_EVENTS } from "@/lib/mitra-mock";
import { ArrowUpRight, Cpu, Database, GitBranch, Zap } from "lucide-react";

export const Route = createFileRoute("/_app/")({
  head: () => ({
    meta: [
      { title: "Overview — MITRA v2" },
      { name: "description", content: "Cluster status, recent runs, and agent health for MITRA v2." },
    ],
  }),
  component: Overview,
});

function Overview() {
  return (
    <>
      <PageHeader
        eyebrow="workspace · overview"
        title="Pipeline at a glance"
        description="Self-hosted agentic AutoML. 8 agents, 3 pages, SSE event bus, Ray compute, BYOM."
        actions={
          <Link
            to="/upload"
            className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-[13px] font-medium text-primary-foreground hover:bg-primary/90"
          >
            <Zap className="h-3.5 w-3.5" /> New run
          </Link>
        }
      />

      <div className="px-6 lg:px-10 py-6 space-y-6">
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          <Stat label="Ray workers" value="4" sub="12 cpu · 0 gpu" icon={Cpu} />
          <Stat label="Runs (24h)" value="17" sub="14 ok · 2 warn · 1 err" icon={GitBranch} />
          <Stat label="Datasets" value="42" sub="2.1 GB cached" icon={Database} />
          <Stat label="Best val_auc" value="0.927" sub="XGBoost · trial #27" mono icon={Zap} />
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-5 gap-4">
          <div className="lg:col-span-3 rounded-lg border border-border bg-surface">
            <Header title="Agent registry" hint="8 agents · 1 per engineer" />
            <ul className="divide-y divide-border">
              {AGENTS.map((a) => (
                <li key={a.id} className="flex items-center gap-3 px-4 py-2.5">
                  <span className="mono text-[10px] text-muted-foreground w-8">S{a.stage}</span>
                  <StatusDot status={a.id === "data_validator" ? "ok" : a.id === "hpt" ? "running" : "idle"} />
                  <div className="flex-1 min-w-0">
                    <div className="text-[13px] font-medium leading-tight truncate">{a.name}</div>
                    <div className="mono text-[10px] text-muted-foreground truncate">
                      reads {a.reads.join(", ")} → writes {a.writes.join(", ")}
                    </div>
                  </div>
                  <span className="mono text-[10px] text-muted-foreground/80">{a.owner}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="lg:col-span-2 rounded-lg border border-border bg-surface">
            <Header
              title="Recent leaderboard"
              hint={<Link to="/results" className="text-primary hover:underline inline-flex items-center gap-0.5">view <ArrowUpRight className="h-3 w-3" /></Link>}
            />
            <ul className="divide-y divide-border">
              {LEADERBOARD.map((row) => (
                <li key={row.rank} className="flex items-center gap-3 px-4 py-2.5">
                  <span className="mono text-[10px] text-muted-foreground w-5">#{row.rank}</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-[13px] truncate">{row.model}</div>
                    <div className="mono text-[10px] text-muted-foreground">
                      val {row.val.toFixed(3)} · gap {row.gap.toFixed(3)}
                    </div>
                  </div>
                  <Badge status={row.status} />
                </li>
              ))}
            </ul>
          </div>
        </section>

        <section className="rounded-lg border border-border bg-surface">
          <Header title="Event stream (last session)" hint={<span className="mono text-[10px] text-muted-foreground">mit_8f2a</span>} />
          <div className="max-h-[280px] overflow-y-auto px-4 py-3 mono text-[11px] leading-relaxed">
            {SAMPLE_EVENTS.slice(0, 10).map((e, i) => (
              <div key={i} className="flex gap-3 py-0.5">
                <span className="text-muted-foreground/70 w-14 shrink-0">{e.t}</span>
                <span className={`w-3 shrink-0 ${levelColor(e.level)}`}>{levelGlyph(e.level)}</span>
                <span className="text-muted-foreground w-32 shrink-0 truncate">{e.agent}</span>
                <span className="text-foreground/90 truncate">{e.msg}</span>
              </div>
            ))}
          </div>
        </section>
      </div>
    </>
  );
}

function Header({ title, hint }: { title: string; hint?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
      <h2 className="text-[12px] font-medium tracking-tight">{title}</h2>
      <div className="text-[11px] text-muted-foreground">{hint}</div>
    </div>
  );
}

function Stat({ label, value, sub, mono, icon: Icon }: { label: string; value: string; sub?: string; mono?: boolean; icon: React.ComponentType<{ className?: string }> }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-4">
      <div className="flex items-center justify-between">
        <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">{label}</div>
        <Icon className="h-3.5 w-3.5 text-muted-foreground" />
      </div>
      <div className={`mt-2 text-2xl font-semibold tracking-tight ${mono ? "mono" : ""}`}>{value}</div>
      {sub ? <div className="mono text-[10px] text-muted-foreground mt-1">{sub}</div> : null}
    </div>
  );
}

function Badge({ status }: { status: "accepted" | "rejected" | "tuning" }) {
  const map = {
    accepted: "text-success bg-success/10 ring-success/20",
    rejected: "text-destructive bg-destructive/10 ring-destructive/20",
    tuning:   "text-info bg-info/10 ring-info/20",
  } as const;
  return (
    <span className={`mono text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded ring-1 ${map[status]}`}>
      {status}
    </span>
  );
}

function levelColor(l: string) {
  return l === "ok" ? "text-success" : l === "warn" ? "text-warning" : l === "error" ? "text-destructive" : "text-info";
}
function levelGlyph(l: string) {
  return l === "ok" ? "✓" : l === "warn" ? "!" : l === "error" ? "✕" : "·";
}
