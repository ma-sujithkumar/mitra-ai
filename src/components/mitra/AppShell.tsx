import { Link, Outlet, useRouterState } from "@tanstack/react-router";
import { Activity, Database, FlaskConical, LayoutDashboard, Layers, Settings2, Workflow } from "lucide-react";
import type { ComponentType, SVGProps } from "react";

type NavItem = {
  to: string;
  label: string;
  icon: ComponentType<SVGProps<SVGSVGElement>>;
  hint?: string;
};

const NAV: NavItem[] = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/upload", label: "New Run", icon: Database, hint: "P1" },
  { to: "/run", label: "Live Run", icon: Activity, hint: "P2" },
  { to: "/results", label: "Results", icon: FlaskConical, hint: "P3" },
  { to: "/agents", label: "Agents", icon: Workflow },
  { to: "/templates", label: "Templates", icon: Layers },
  { to: "/settings", label: "Settings", icon: Settings2 },
];

export function AppShell() {
  const path = useRouterState({ select: (s) => s.location.pathname });

  return (
    <div className="flex min-h-screen w-full bg-background text-foreground">
      <aside className="hidden md:flex w-60 shrink-0 flex-col border-r border-border bg-sidebar">
        <div className="flex h-14 items-center gap-2 border-b border-sidebar-border px-4">
          <div className="relative h-6 w-6 rounded-md bg-primary/15 ring-1 ring-primary/40 flex items-center justify-center">
            <div className="h-2 w-2 rounded-sm bg-primary" />
          </div>
          <div className="leading-tight">
            <div className="text-[13px] font-semibold tracking-tight">MITRA</div>
            <div className="text-[10px] mono uppercase tracking-[0.14em] text-muted-foreground">v2 · self-hosted</div>
          </div>
        </div>

        <nav className="flex-1 px-2 py-3">
          <div className="px-2 pb-2 text-[10px] mono uppercase tracking-widest text-muted-foreground">Workspace</div>
          <ul className="space-y-0.5">
            {NAV.map((item) => {
              const active = item.to === "/" ? path === "/" : path.startsWith(item.to);
              const Icon = item.icon;
              return (
                <li key={item.to}>
                  <Link
                    to={item.to}
                    className={[
                      "group flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[13px] transition-colors",
                      active
                        ? "bg-sidebar-accent text-sidebar-accent-foreground"
                        : "text-sidebar-foreground/80 hover:bg-sidebar-accent/60 hover:text-sidebar-accent-foreground",
                    ].join(" ")}
                  >
                    <Icon className="h-3.5 w-3.5 shrink-0 opacity-80" />
                    <span className="flex-1 truncate">{item.label}</span>
                    {item.hint ? (
                      <span className="mono text-[9px] tracking-wider text-muted-foreground/80 group-hover:text-foreground/60">
                        {item.hint}
                      </span>
                    ) : null}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        <div className="border-t border-sidebar-border p-3">
          <div className="rounded-md bg-sidebar-accent/40 p-2.5">
            <div className="flex items-center gap-2 text-[11px]">
              <span className="status-dot bg-success pulse-ring" />
              <span className="mono text-muted-foreground">ray://local</span>
            </div>
            <div className="mt-1 mono text-[10px] text-muted-foreground/70">4 workers · 12 cpu · 0 gpu</div>
          </div>
        </div>
      </aside>

      <main className="flex-1 min-w-0">
        <Outlet />
      </main>
    </div>
  );
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="border-b border-border bg-background/80 backdrop-blur">
      <div className="px-6 lg:px-10 py-6 flex items-end justify-between gap-6">
        <div>
          {eyebrow ? (
            <div className="mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground mb-2">
              {eyebrow}
            </div>
          ) : null}
          <h1 className="text-[22px] font-semibold tracking-tight">{title}</h1>
          {description ? (
            <p className="mt-1 text-sm text-muted-foreground max-w-2xl">{description}</p>
          ) : null}
        </div>
        {actions ? <div className="flex items-center gap-2">{actions}</div> : null}
      </div>
    </div>
  );
}
