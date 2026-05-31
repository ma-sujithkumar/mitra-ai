import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/mitra/AppShell";

export const Route = createFileRoute("/_app/settings")({
  head: () => ({
    meta: [
      { title: "Settings — MITRA v2" },
      { name: "description", content: "Cluster, thresholds, and event bus settings for MITRA v2." },
    ],
  }),
  component: SettingsPage,
});

function SettingsPage() {
  return (
    <>
      <PageHeader
        eyebrow="workspace · settings"
        title="Settings"
        description="Cluster, judge thresholds, and event bus configuration."
      />
      <div className="px-6 lg:px-10 py-6 grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card title="Ray cluster">
          <Field k="head" v="ray://localhost:6379" />
          <Field k="workers" v="4" />
          <Field k="cpu_per_worker" v="3" />
          <Field k="gpu_per_worker" v="0" />
        </Card>
        <Card title="Judge thresholds">
          <Field k="gap_max" v="0.08" />
          <Field k="floor" v="0.85" />
          <Field k="max_retries" v="3" />
          <Field k="shap_memory_budget_mb" v="2048" />
        </Card>
        <Card title="SSE event bus">
          <Field k="transport" v="asyncio.Queue · 1 per session" />
          <Field k="schema" v="typed · pydantic v2" />
          <Field k="heartbeat_ms" v="15000" />
        </Card>
        <Card title="BYOM defaults">
          <Field k="provider" v="OpenAI" />
          <Field k="model" v="gpt-4o-mini" />
          <Field k="timeout_s" v="60" />
        </Card>
      </div>
    </>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-surface">
      <div className="border-b border-border px-4 py-2.5">
        <h2 className="text-[12px] font-medium tracking-tight">{title}</h2>
      </div>
      <dl className="p-4 space-y-2.5">{children}</dl>
    </div>
  );
}

function Field({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-border/60 last:border-0 pb-2 last:pb-0">
      <dt className="mono text-[10px] uppercase tracking-wider text-muted-foreground">{k}</dt>
      <dd className="mono text-[12px] text-foreground/90 text-right">{v}</dd>
    </div>
  );
}
