import { useEffect, useState } from 'react';

import StatusPill from '../components/StatusPill.jsx';
import Toast from '../components/Toast.jsx';
import { fetchPlots, plotUrl, generatePlots } from '../api/client.js';
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
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationMessage, setGenerationMessage] = useState(null);
  const [generationError, setGenerationError] = useState(null);

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
  }, [activeSessionId, refreshTrigger]);

  const handleGenerate = async () => {
    setGenerationMessage(null);
    setGenerationError(null);
    setIsGenerating(true);
    try {
      const response = await generatePlots(activeSessionId);
      setGenerationMessage(response.message || 'Visualizations generated successfully');
      setRefreshTrigger((prev) => prev + 1);
    } catch (err) {
      setGenerationError(err.message || 'Failed to generate visualizations');
    } finally {
      setIsGenerating(false);
    }
  };

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

  const groupedPlots = groupPlotsByStage(plots);

  return (
    <div className="screen-stack">
      <section className="card hero-panel" style={{ paddingBottom: 14, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 15 }}>
        <div>
          <StatusPill status={plots.length > 0 ? 'done' : 'queued'} label={`${plots.length} plots`} />
          <h2>All Generated Visualizations</h2>
          <p className="muted">Generate or refresh visual analytics for the current top 10 models.</p>
        </div>
        <div>
          <button
            className="btn btn-primary"
            onClick={handleGenerate}
            disabled={isGenerating}
            style={{ display: 'flex', alignItems: 'center', gap: 8 }}
            type="button"
          >
            {isGenerating ? <div className="spinner small" /> : <Icons.spark size={15} />}
            {isGenerating ? 'Generating...' : plots.length > 0 ? 'Refresh Visualizations' : 'Generate Visualizations'}
          </button>
        </div>
      </section>

      {/* Compact auto-dismissing toast instead of a full-width inline banner. */}
      <Toast
        message={generationMessage}
        tone="success"
        onDismiss={() => setGenerationMessage(null)}
      />

      {generationError && (
        <div className="callout error compact" style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
          <Icons.x size={16} style={{ color: 'var(--error)' }} />
          <span>{generationError}</span>
        </div>
      )}

      {loadState === 'loading' && plots.length === 0 && (
        <div style={{ marginTop: 20 }}>
          <StatusPill status="running" spin label="Loading plots..." />
        </div>
      )}

      {loadState === 'error' && plots.length === 0 && (
        <div className="callout error compact" style={{ marginTop: 20 }}>
          <strong>Failed to load plots.</strong>
          <span>Check that the session has completed at least one pipeline stage.</span>
        </div>
      )}

      {plots.length === 0 && loadState !== 'loading' && (
        <div className="callout compact" style={{ marginTop: 20 }}>
          <strong>No plots yet</strong>
          <span>Plots are generated during training and evaluation. Click 'Generate Visualizations' above to generate plots now.</span>
        </div>
      )}

      {plots.length > 0 && Array.from(groupedPlots.entries()).map(([stage, stagePlots]) => (
        <section className="card panel-section" key={stage} style={{ marginTop: 20 }}>
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
