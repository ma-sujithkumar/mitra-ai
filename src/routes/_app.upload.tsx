import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { PageHeader } from "@/components/mitra/AppShell";
import { Upload, FileSpreadsheet, KeyRound, Cpu, ArrowRight } from "lucide-react";

export const Route = createFileRoute("/_app/upload")({
  head: () => ({
    meta: [
      { title: "New run — MITRA v2" },
      { name: "description", content: "Upload a dataset, choose task & model provider, and launch an agentic AutoML run." },
    ],
  }),
  component: UploadPage,
});

const PROVIDERS = ["OpenAI", "Anthropic", "Gemini", "Local (Ollama)"] as const;
const TASKS = [
  { id: "auto", label: "Auto-detect", hint: "Let model_selection route" },
  { id: "classification", label: "Classification", hint: "Binary or multi-class" },
  { id: "regression", label: "Regression", hint: "Continuous target" },
  { id: "usl", label: "Unsupervised", hint: "Cluster / anomaly" },
] as const;

function UploadPage() {
  const nav = useNavigate();
  const [task, setTask] = useState<(typeof TASKS)[number]["id"]>("auto");
  const [provider, setProvider] = useState<(typeof PROVIDERS)[number]>("OpenAI");
  const [file, setFile] = useState<File | null>(null);
  const [target, setTarget] = useState("");

  return (
    <>
      <PageHeader
        eyebrow="page 1 · upload"
        title="Configure a new run"
        description="Drop a CSV or Parquet, pick a task, and bring your own model key."
      />

      <div className="px-6 lg:px-10 py-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card icon={FileSpreadsheet} title="Dataset" hint="CSV · Parquet · max 2 GB">
            <label
              htmlFor="ds"
              className="flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed border-border-strong bg-background/40 px-6 py-12 cursor-pointer hover:bg-background/60 hover:border-primary/40 transition-colors"
            >
              <div className="rounded-full bg-primary/10 p-3 ring-1 ring-primary/20">
                <Upload className="h-5 w-5 text-primary" />
              </div>
              <div className="text-center">
                <div className="text-sm font-medium">{file ? file.name : "Drop file or click to browse"}</div>
                <div className="mono text-[10px] text-muted-foreground mt-1">
                  {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : "data_validator runs first · bad rows fail fast"}
                </div>
              </div>
              <input
                id="ds"
                type="file"
                accept=".csv,.parquet"
                className="sr-only"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </label>

            <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
              <Field label="Target column">
                <input
                  value={target}
                  onChange={(e) => setTarget(e.target.value)}
                  placeholder="e.g. churned"
                  className="w-full bg-background border border-input rounded-md px-3 py-2 text-sm placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-ring"
                />
              </Field>
              <Field label="Validation split">
                <input
                  defaultValue="0.20"
                  className="w-full bg-background border border-input rounded-md px-3 py-2 text-sm mono focus:outline-none focus:ring-1 focus:ring-ring"
                />
              </Field>
            </div>
          </Card>

          <Card icon={Cpu} title="Task family" hint="model_selection · agent S4">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {TASKS.map((t) => {
                const active = task === t.id;
                return (
                  <button
                    key={t.id}
                    onClick={() => setTask(t.id)}
                    className={[
                      "text-left rounded-md border px-3 py-2.5 transition-colors",
                      active
                        ? "border-primary/60 bg-primary/10 ring-1 ring-primary/30"
                        : "border-border bg-background/40 hover:border-border-strong",
                    ].join(" ")}
                  >
                    <div className="text-[13px] font-medium">{t.label}</div>
                    <div className="mono text-[10px] text-muted-foreground mt-0.5">{t.hint}</div>
                  </button>
                );
              })}
            </div>
          </Card>

          <Card icon={KeyRound} title="Bring your own model" hint="LiteLLM factory">
            <div className="flex flex-wrap gap-2 mb-3">
              {PROVIDERS.map((p) => (
                <button
                  key={p}
                  onClick={() => setProvider(p)}
                  className={[
                    "px-2.5 py-1 rounded-md text-[12px] border transition-colors",
                    provider === p
                      ? "border-primary/50 bg-primary/10 text-foreground"
                      : "border-border text-muted-foreground hover:text-foreground",
                  ].join(" ")}
                >
                  {p}
                </button>
              ))}
            </div>
            <Field label="API key">
              <input
                type="password"
                placeholder="sk-…"
                className="w-full bg-background border border-input rounded-md px-3 py-2 text-sm mono focus:outline-none focus:ring-1 focus:ring-ring"
              />
            </Field>
            <p className="mono text-[10px] text-muted-foreground mt-2">
              Keys stay in the session. Never logged. Never written to disk.
            </p>
          </Card>
        </div>

        <aside className="space-y-3">
          <div className="rounded-lg border border-border bg-surface p-4">
            <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">Run summary</div>
            <dl className="mt-3 space-y-2 text-[12px]">
              <Row k="dataset" v={file?.name ?? "—"} />
              <Row k="target" v={target || "—"} />
              <Row k="task" v={task} />
              <Row k="provider" v={provider} />
              <Row k="compute" v="ray://local · 4w" />
            </dl>
          </div>

          <button
            disabled={!file}
            onClick={() => nav({ to: "/run" })}
            className="w-full inline-flex items-center justify-center gap-2 rounded-md bg-primary px-4 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Launch run <ArrowRight className="h-4 w-4" />
          </button>

          <p className="mono text-[10px] text-muted-foreground px-1">
            Triggers data_validator → metadata_gen → feature_selection → model_selection → family agent → judge → hpt.
          </p>
        </aside>
      </div>
    </>
  );
}

function Card({ icon: Icon, title, hint, children }: { icon: React.ComponentType<{ className?: string }>; title: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-surface">
      <div className="flex items-center gap-2 border-b border-border px-4 py-2.5">
        <Icon className="h-3.5 w-3.5 text-muted-foreground" />
        <h2 className="text-[12px] font-medium tracking-tight flex-1">{title}</h2>
        {hint ? <span className="mono text-[10px] text-muted-foreground">{hint}</span> : null}
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mono text-[10px] uppercase tracking-widest text-muted-foreground">{label}</span>
      <div className="mt-1.5">{children}</div>
    </label>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="mono text-[10px] uppercase tracking-wider text-muted-foreground">{k}</dt>
      <dd className="mono text-[11px] text-foreground/90 truncate text-right max-w-[60%]">{v}</dd>
    </div>
  );
}
