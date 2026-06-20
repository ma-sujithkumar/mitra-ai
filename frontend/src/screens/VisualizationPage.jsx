import { useEffect, useState } from 'react';

import StatusPill from '../components/StatusPill.jsx';
import { fetchPlots, plotUrl } from '../api/client.js';
import { Icons } from '../icons.jsx';

// Bounded polling so the gallery syncs as plots are generated.
const VISUALIZATION_POLL_MS = 4000;
const VISUALIZATION_MAX_POLLS = 60;

// Stage labels shown as section headers in the gallery.
const STAGE_LABELS = {
  'plots/eda':         'Exploratory Data Analysis',
  'plots/training':    'Training',
  'plots/overfitting': 'Overfitting Analysis',
  'plots/hpt':         'Hyperparameter Tuning',
  'plots/judge':       'Judge',
  'plots/shap':        'SHAP / Explainability',
  'evaluation/shap':   'SHAP / Explainability',
  'evaluation/hpt':    'Hyperparameter Tuning',
};

function stageLabelFor(stageStr) {
  return STAGE_LABELS[stageStr] || stageStr.replace(/[/_-]/g, ' ').replace(/\b\w/g, (ch) => ch.toUpperCase());
}

// Group a flat plots list by stage string for section rendering.
function groupPlotsByStage(plots) {
  const groups = new Map();
  for (const plot of plots) {
    const stage = plot.stage || 'other';
    if (!groups.has(stage)) groups.set(stage, []);
    groups.get(stage).push(plot);
  }
  return groups;
}

function LightboxOverlay({ src, name, onClose }) {
  return (
    <div className="lightbox-overlay" onClick={onClose} role="dialog" aria-modal="true">
      <div className="lightbox-content" onClick={(event) => event.stopPropagation()}>
        <button className="lightbox-close btn-icon" onClick={onClose} type="button" aria-label="Close">
          <Icons.x size={20} />
        </button>
        <img alt={name} className="lightbox-image" src={src} />
        <p className="lightbox-label muted">{name}</p>
      </div>
    </div>
  );
}

function PlotCard({ sessionId, plot }) {
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const src = plotUrl(sessionId, plot.path);

  return (
    <>
      <div className="plot-card" onClick={() => setLightboxOpen(true)} role="button" tabIndex={0}>
        <img alt={plot.name} className="plot-thumb" loading="lazy" src={src} />
        <p className="plot-name muted">{plot.name.replace(/_/g, ' ')}</p>
      </div>
      {lightboxOpen ? (
        <LightboxOverlay name={plot.name} onClose={() => setLightboxOpen(false)} src={src} />
      ) : null}
    </>
  );
}

function VisualizationPage({ activeSessionId }) {
  const [plots, setPlots] = useState([]);
  const [loadState, setLoadState] = useState('idle');

  useEffect(() => {
    if (!activeSessionId) {
      setLoadState('idle');
      setPlots([]);
      return undefined;
    }

    let cancelled = false;
    let timeoutId = null;
    let attempts = 0;
    setLoadState('loading');

    // Bounded poll so new plots appear as the pipeline produces them. Plots
    // arrive incrementally (eda -> training -> overfitting -> hpt -> judge/shap),
    // so keep refreshing for a while rather than fetching once on mount.
    async function pollOnce() {
      attempts += 1;
      try {
        const data = await fetchPlots(activeSessionId);
        if (cancelled) return;
        setPlots(data?.plots || []);
        setLoadState('done');
        if (attempts < VISUALIZATION_MAX_POLLS) {
          timeoutId = setTimeout(pollOnce, VISUALIZATION_POLL_MS);
        }
      } catch (pollError) {
        if (!cancelled) setLoadState('error');
      }
    }

    pollOnce();

    return () => {
      cancelled = true;
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [activeSessionId]);

  if (!activeSessionId) {
    return (
      <div className="screen-stack">
        <div className="callout compact">
          <strong>No active session</strong>
          <span>Start a training run to view generated plots.</span>
        </div>
      </div>
    );
  }

  if (loadState === 'loading') {
    return (
      <div className="screen-stack">
        <StatusPill status="running" spin label="Loading plots..." />
      </div>
    );
  }

  if (loadState === 'error') {
    return (
      <div className="screen-stack">
        <div className="callout error compact">
          <strong>Failed to load plots.</strong>
          <span>Check that the session has completed at least one pipeline stage.</span>
        </div>
      </div>
    );
  }

  if (plots.length === 0) {
    return (
      <div className="screen-stack">
        <div className="callout compact">
          <strong>No plots yet</strong>
          <span>Plots are generated during training and evaluation. Run the pipeline first.</span>
        </div>
      </div>
    );
  }

  const groupedPlots = groupPlotsByStage(plots);

  return (
    <div className="screen-stack">
      <section className="card hero-panel" style={{ paddingBottom: 14 }}>
        <div>
          <StatusPill status="done" label={`${plots.length} plots`} />
          <h2>All Generated Visualizations</h2>
          <p className="muted">Click any plot to enlarge it.</p>
        </div>
      </section>

      {Array.from(groupedPlots.entries()).map(([stage, stagePlots]) => (
        <section className="card panel-section" key={stage}>
          <div className="section-head">
            <div>
              <p className="section-kicker">{stage}</p>
              <h2>{stageLabelFor(stage)}</h2>
            </div>
            <span className="pill pill-queued">{stagePlots.length}</span>
          </div>
          <div className="plot-gallery">
            {stagePlots.map((plot) => (
              <PlotCard key={plot.path} plot={plot} sessionId={activeSessionId} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

export default VisualizationPage;
