import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Outlet,
  createRootRouteWithContext,
  useRouter,
  HeadContent,
  Scripts,
} from "@tanstack/react-router";
import { useEffect, type ReactNode } from "react";

import appCss from "../styles.css?url";
import { reportLovableError } from "../lib/lovable-error-reporting";

function NotFoundComponent() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <div className="mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">404 · not_found</div>
        <h1 className="mt-3 text-2xl font-semibold tracking-tight text-foreground">Route not in the pipeline</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          That page doesn't exist. Head back to the overview.
        </p>
        <a
          href="/"
          className="mt-6 inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
        >
          Go to overview
        </a>
      </div>
    </div>
  );
}

function ErrorComponent({ error, reset }: { error: Error; reset: () => void }) {
  console.error(error);
  const router = useRouter();
  useEffect(() => {
    reportLovableError(error, { boundary: "tanstack_root_error_component" });
  }, [error]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background px-4">
      <div className="max-w-md text-center">
        <div className="mono text-[10px] uppercase tracking-[0.2em] text-destructive">runtime · error</div>
        <h1 className="mt-3 text-xl font-semibold tracking-tight text-foreground">Something broke</h1>
        <p className="mt-2 text-sm text-muted-foreground">Try again or return to the overview.</p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <button
            onClick={() => { router.invalidate(); reset(); }}
            className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90"
          >
            Try again
          </button>
          <a href="/" className="inline-flex items-center justify-center rounded-md border border-border bg-background px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent">
            Go home
          </a>
        </div>
      </div>
    </div>
  );
}

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      { title: "MITRA v2 — Self-Hosted Agentic AutoML" },
      { name: "description", content: "MITRA v2 is a self-hosted agentic AutoML platform. Upload a dataset, watch 8 agents collaborate, ship a tuned model." },
      { name: "author", content: "MITRA" },
      { property: "og:title", content: "MITRA v2 — Self-Hosted Agentic AutoML" },
      { property: "og:description", content: "MITRA v2 is a self-hosted agentic AutoML platform. Upload a dataset, watch 8 agents collaborate, ship a tuned model." },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary" },
      { name: "twitter:title", content: "MITRA v2 — Self-Hosted Agentic AutoML" },
      { name: "twitter:description", content: "MITRA v2 is a self-hosted agentic AutoML platform. Upload a dataset, watch 8 agents collaborate, ship a tuned model." },
      { property: "og:image", content: "https://pub-bb2e103a32db4e198524a2e9ed8f35b4.r2.dev/7b357a96-2c38-4033-9fcf-57f795980c30/id-preview-7759ef04--0bac6f7f-11ef-4e57-9a6a-8075728179b9.lovable.app-1780222245698.png" },
      { name: "twitter:image", content: "https://pub-bb2e103a32db4e198524a2e9ed8f35b4.r2.dev/7b357a96-2c38-4033-9fcf-57f795980c30/id-preview-7759ef04--0bac6f7f-11ef-4e57-9a6a-8075728179b9.lovable.app-1780222245698.png" },
    ],
    links: [{ rel: "stylesheet", href: appCss }],
  }),
  shellComponent: RootShell,
  component: RootComponent,
  notFoundComponent: NotFoundComponent,
  errorComponent: ErrorComponent,
});

function RootShell({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head><HeadContent /></head>
      <body>{children}<Scripts /></body>
    </html>
  );
}

function RootComponent() {
  const { queryClient } = Route.useRouteContext();
  return (
    <QueryClientProvider client={queryClient}>
      <Outlet />
    </QueryClientProvider>
  );
}
