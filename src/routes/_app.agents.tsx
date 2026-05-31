import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/mitra/AppShell";
import { AGENTS } from "@/lib/mitra-mock";
import { ArrowRight, FileInput, FileOutput } from "lucide-react";

export const Route = createFileRoute("/_app/agents")({
  head: () => ({
    meta: [
      { title: "Agents — MITRA v2" },
      { name: "description", content: "All 8 MITRA agents with I/O contracts and owners." },
    ],
  }),
  component: AgentsPage,
});

function AgentsPage() {
  return (
    <>
      <PageHeader
        eyebrow="reference · agents"
        title="Agent registry"
        description="One agent per engineer. Strict I/O contracts. Built on Google ADK (adk-python)."
      />
      <div className="px-6 lg:px-10 py-6 grid grid-cols-1 md:grid-cols-2 gap-4">
        {AGENTS.map((a) => (
          <div key={a.id} className="rounded-lg border border-border bg-surface p-4 hover:border-border-strong transition-colors">
            <div className="flex items-start justify-between gap-3">
              <div>
                <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  stage {a.stage} · {a.owner}
                </div>
                <h3 className="mt-1 text-[15px] font-semibold tracking-tight">{a.name}</h3>
              </div>
              <span className="mono text-[10px] text-muted-foreground/70 truncate">{a.id}</span>
            </div>
            <p className="mt-2 text-[12px] text-muted-foreground leading-relaxed">{a.description}</p>

            <div className="mt-4 rounded-md bg-background/50 border border-border p-3 space-y-2">
              <Row icon={FileInput} label="reads" items={a.reads} />
              <div className="flex justify-center"><ArrowRight className="h-3 w-3 text-muted-foreground/50" /></div>
              <Row icon={FileOutput} label="writes" items={a.writes} accent />
            </div>
          </div>
        ))}
      </div>
    </>
  );
}

function Row({ icon: Icon, label, items, accent }: { icon: React.ComponentType<{ className?: string }>; label: string; items: string[]; accent?: boolean }) {
  return (
    <div className="flex items-start gap-2">
      <Icon className={`h-3 w-3 mt-0.5 ${accent ? "text-primary" : "text-muted-foreground"}`} />
      <span className="mono text-[10px] uppercase tracking-wider text-muted-foreground w-12">{label}</span>
      <div className="flex flex-wrap gap-1 flex-1">
        {items.map((it) => (
          <span key={it} className={`mono text-[10px] px-1.5 py-0.5 rounded ring-1 ${
            accent ? "text-primary bg-primary/10 ring-primary/20" : "text-foreground/80 bg-background ring-border"
          }`}>{it}</span>
        ))}
      </div>
    </div>
  );
}
