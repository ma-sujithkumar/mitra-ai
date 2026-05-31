import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/mitra/AppShell";
import { TEMPLATES } from "@/lib/mitra-mock";
import { FileCode2 } from "lucide-react";

export const Route = createFileRoute("/_app/templates")({
  head: () => ({
    meta: [
      { title: "Templates — MITRA v2" },
      { name: "description", content: "Jinja2 training templates and Optuna search spaces per model family." },
    ],
  }),
  component: TemplatesPage,
});

function TemplatesPage() {
  const families = Array.from(new Set(TEMPLATES.map((t) => t.family)));
  return (
    <>
      <PageHeader
        eyebrow="reference · templates"
        title="Template library"
        description="Each family ships train.py.j2, hp_space.yaml, resources.yaml. HPT searches the declared space."
      />
      <div className="px-6 lg:px-10 py-6 space-y-6">
        {families.map((fam) => (
          <section key={fam}>
            <div className="flex items-baseline gap-3 mb-3">
              <h2 className="text-[13px] font-semibold tracking-tight uppercase">{fam}</h2>
              <span className="mono text-[10px] text-muted-foreground">
                templates/{fam}/
              </span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {TEMPLATES.filter((t) => t.family === fam).map((t) => (
                <div key={t.model} className="rounded-lg border border-border bg-surface p-4 hover:border-primary/40 transition-colors">
                  <div className="flex items-center gap-2">
                    <FileCode2 className="h-3.5 w-3.5 text-primary" />
                    <h3 className="text-[13px] font-medium mono">{t.model}</h3>
                  </div>
                  <p className="mt-2 text-[12px] text-muted-foreground">{t.description}</p>
                  <ul className="mt-3 space-y-0.5">
                    {t.files.map((f) => (
                      <li key={f} className="mono text-[10px] text-muted-foreground/80 flex items-center gap-2">
                        <span className="text-muted-foreground/40">›</span>{f}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>
    </>
  );
}
