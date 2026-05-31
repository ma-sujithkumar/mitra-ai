import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/mitra/AppShell";
import { LEADERBOARD } from "@/lib/mitra-mock";
import { Download, FileJson, Scale, Sparkles } from "lucide-react";

export const Route = createFileRoute("/_app/results")({
  head: () => ({
    meta: [
      { title: "Results — MITRA v2" },
      { name: "description", content: "Leaderboard, judge verdict, and artifacts from the last MITRA run." },
    ],
  }),
  component: ResultsPage,
});

function ResultsPage() {
  const best = LEADERBOARD[0];
  return (
    <>
      <PageHeader
        eyebrow="page 3 · results"
        title="Run mit_8f2a"
        description="churn_q3.csv · target=churned · binary classification · 41/83 features kept"
        actions={
          <button className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-[13px] font-medium text-primary-foreground hover:bg-primary/90">
            <Download className="h-3.5 w-3.5" /> Download bundle
          </button>
        }
      />

      <div className="px-6 lg:px-10 py-6 space-y-6">
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2 rounded-lg border border-primary/30 bg-gradient-to-br from-primary/10 to-transparent p-5">
            <div className="flex items-center gap-2 mono text-[10px] uppercase tracking-widest text-primary">
              <Sparkles className="h-3 w-3" /> Best model
            </div>
            <div className="mt-2 flex items-baseline gap-3">
              <h2 className="text-2xl font-semibold tracking-tight">{best.model}</h2>
              <span className="mono text-[11px] text-muted-foreground">final_model.pkl · 4.2 MB</span>
            </div>
            <div className="mt-4 grid grid-cols-4 gap-4">
              <Metric k="val_auc" v={best.val.toFixed(3)} accent />
              <Metric k="train_auc" v={best.train.toFixed(3)} />
              <Metric k="gap" v={best.gap.toFixed(3)} />
              <Metric k="trials" v="40" />
            </div>
          </div>

          <div className="rounded-lg border border-border bg-surface p-5">
            <div className="flex items-center gap-2 mono text-[10px] uppercase tracking-widest text-muted-foreground">
              <Scale className="h-3 w-3" /> Judge verdict
            </div>
            <div className="mt-2 text-[13px] leading-relaxed">
              <p className="text-foreground/90">
                <span className="text-success font-medium">accept</span> · gap{" "}
                <span className="mono">0.018</span> ≤{" "}
                <span className="mono">0.08</span> · floor{" "}
                <span className="mono">0.85</span> cleared.
              </p>
              <p className="mt-2 text-muted-foreground text-[12px]">
                RandomForest excluded earlier (gap 0.127). HPT accepted Optuna trial #27.
                SHAP guard: within memory budget.
              </p>
            </div>
          </div>
        </section>

        <section className="rounded-lg border border-border bg-surface">
          <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
            <h2 className="text-[12px] font-medium">Leaderboard</h2>
            <span className="mono text-[10px] text-muted-foreground">4 candidates · 1 tuned</span>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]">
              <thead>
                <tr className="text-left mono text-[10px] uppercase tracking-widest text-muted-foreground border-b border-border">
                  <th className="px-4 py-2 font-normal w-10">#</th>
                  <th className="px-4 py-2 font-normal">Model</th>
                  <th className="px-4 py-2 font-normal text-right">val_auc</th>
                  <th className="px-4 py-2 font-normal text-right">train_auc</th>
                  <th className="px-4 py-2 font-normal text-right">gap</th>
                  <th className="px-4 py-2 font-normal">status</th>
                  <th className="px-4 py-2 font-normal">notes</th>
                </tr>
              </thead>
              <tbody>
                {LEADERBOARD.map((r) => (
                  <tr key={r.rank} className="border-b border-border/60 last:border-0 hover:bg-accent/30">
                    <td className="px-4 py-2.5 mono text-muted-foreground">{r.rank}</td>
                    <td className="px-4 py-2.5 font-medium">{r.model}</td>
                    <td className="px-4 py-2.5 text-right mono">{r.val.toFixed(3)}</td>
                    <td className="px-4 py-2.5 text-right mono text-muted-foreground">{r.train.toFixed(3)}</td>
                    <td className={`px-4 py-2.5 text-right mono ${r.gap > 0.08 ? "text-destructive" : "text-success"}`}>
                      {r.gap.toFixed(3)}
                    </td>
                    <td className="px-4 py-2.5">
                      <span className={[
                        "mono text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded ring-1",
                        r.status === "accepted" ? "text-success bg-success/10 ring-success/20" :
                        r.status === "rejected" ? "text-destructive bg-destructive/10 ring-destructive/20" :
                        "text-info bg-info/10 ring-info/20",
                      ].join(" ")}>{r.status}</span>
                    </td>
                    <td className="px-4 py-2.5 text-muted-foreground">{r.notes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="rounded-lg border border-border bg-surface">
            <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
              <h2 className="text-[12px] font-medium flex items-center gap-2">
                <FileJson className="h-3 w-3 text-muted-foreground" /> verdict.json
              </h2>
              <button className="mono text-[10px] text-muted-foreground hover:text-foreground">copy</button>
            </div>
            <pre className="px-4 py-3 mono text-[11px] leading-relaxed overflow-x-auto">
{`{
  "verdict": "accept",
  "model": "xgboost",
  "val_auc": 0.927,
  "gap": 0.018,
  "threshold": { "gap_max": 0.08, "floor": 0.85 },
  "retries": 0,
  "shap_available": true
}`}
            </pre>
          </div>

          <div className="rounded-lg border border-border bg-surface">
            <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
              <h2 className="text-[12px] font-medium flex items-center gap-2">
                <FileJson className="h-3 w-3 text-muted-foreground" /> best_params.json
              </h2>
              <button className="mono text-[10px] text-muted-foreground hover:text-foreground">copy</button>
            </div>
            <pre className="px-4 py-3 mono text-[11px] leading-relaxed overflow-x-auto">
{`{
  "n_estimators": 380,
  "max_depth": 6,
  "learning_rate": 0.041,
  "subsample": 0.82,
  "colsample_bytree": 0.73,
  "reg_lambda": 1.4,
  "trial_id": 27
}`}
            </pre>
          </div>
        </section>
      </div>
    </>
  );
}

function Metric({ k, v, accent }: { k: string; v: string; accent?: boolean }) {
  return (
    <div>
      <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">{k}</div>
      <div className={`mt-1 text-xl font-semibold mono ${accent ? "text-primary" : ""}`}>{v}</div>
    </div>
  );
}
