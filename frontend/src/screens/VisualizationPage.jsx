import { useEffect, useState } from 'react';

import StatusPill from '../components/StatusPill.jsx';
import Toast from '../components/Toast.jsx';
import { fetchPlots, plotUrl, generatePlots, fetchFEVisuals, feVisualUrl, fetchFeatureLeaderboard } from '../api/client.js';
import { Icons } from '../icons.jsx';


const VIZ_TYPE_TRAINING = 'training';
const VIZ_TYPE_FEATURE_ENGINEERING = 'feature_engineering';

const VIZ_TYPE_OPTIONS = [
  { value: VIZ_TYPE_TRAINING, label: 'Training & Evaluation' },
  { value: VIZ_TYPE_FEATURE_ENGINEERING, label: 'Feature Engineering' },
];

// Stage labels shown as section headers in the training gallery.
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

// ---- Feature Engineering visual groupings ----
// Maps chart filenames into logical sections, matching the dashboard categories.
const FE_CATEGORY_GROUPS = [
  {
    key: 'importance',
    kicker: 'Selection',
    label: 'Feature Importance & Selection',
    filenames: new Set(['01_feature_importance.html', '07_selection_rationale.html']),
  },
  {
    key: 'preprocessing',
    kicker: 'Data Quality',
    label: 'Preprocessing Decisions',
    filenames: new Set(['03_imputation_decisions.html', '04_outlier_decisions.html', '05_scaling_decisions.html']),
  },
  {
    key: 'structure',
    kicker: 'Statistics',
    label: 'Feature Structure & Engineering',
    filenames: new Set(['02_correlation_clusters.html', '06_created_features.html', '08_pca_variance.html']),
  },
  {
    key: 'pipeline',
    kicker: 'Execution',
    label: 'Pipeline Overview',
    filenames: new Set(['09_pipeline_timeline.html']),
  },
];

function groupFEVisuals(visuals) {
  const grouped = [];
  for (const group of FE_CATEGORY_GROUPS) {
    const matching = visuals.filter((visual) => group.filenames.has(visual.filename));
    if (matching.length > 0) {
      grouped.push({ ...group, visuals: matching });
    }
  }
  // Any chart not belonging to a category group lands in an "Other" section.
  const categorised = new Set(FE_CATEGORY_GROUPS.flatMap((group) => [...group.filenames]));
  const uncategorised = visuals.filter((visual) => !categorised.has(visual.filename));
  if (uncategorised.length > 0) {
    grouped.push({ key: 'other', kicker: 'Misc', label: 'Other Charts', filenames: new Set(), visuals: uncategorised });
  }
  return grouped;
}

// ---- Shared components ----

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

// ---- Feature Engineering gallery with category sections ----

function FEVisualCard({ sessionId, visual }) {
  return (
    <div className="fe-visual-card">
      <iframe
        src={feVisualUrl(sessionId, visual.filename)}
        style={{ width: '100%', height: visual.height, border: 'none', display: 'block' }}
        title={visual.title}
        loading="lazy"
      />
    </div>
  );
}

function FEVisualsGallery({ sessionId, visuals, dashboardAvailable }) {
  if (!visuals.length) return null;

  const groups = groupFEVisuals(visuals);

  return (
    <div className="screen-stack" style={{ marginTop: 4 }}>
      {dashboardAvailable ? (
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <a
            className="btn btn-ghost"
            href={feVisualUrl(sessionId, 'dashboard.html')}
            rel="noopener noreferrer"
            style={{ fontSize: 13 }}
            target="_blank"
          >
            <Icons.spark size={14} style={{ marginRight: 6 }} />
            Open Full Dashboard
          </a>
        </div>
      ) : null}

      {groups.map((group) => (
        <section className="card panel-section" key={group.key} style={{ marginTop: 16 }}>
          <div className="section-head">
            <div>
              <p className="section-kicker">{group.kicker}</p>
              <h2>{group.label}</h2>
            </div>
            <span className="pill pill-queued">{group.visuals.length}</span>
          </div>
          <div className="fe-visuals-stack" style={{ marginTop: 12 }}>
            {group.visuals.map((visual) => (
              <FEVisualCard key={visual.filename} sessionId={sessionId} visual={visual} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

// ---- Feature Engineering multi-metric comparison chart ----

const VIZ_FE_ALGO_META = [
  { key: 'mi',        label: 'Mutual Info',  color: '#6366f1' },
  { key: 'ig',        label: 'Info Gain',    color: '#22c55e' },
  { key: 'mrmr',      label: 'mRMR',         color: '#f59e0b' },
  { key: 'laplacian', label: 'Laplacian',    color: '#06b6d4' },
  { key: 'variance',  label: 'Variance',     color: '#a855f7' },
];

// Cap at top-25 features to keep the chart readable.
const FE_MULTI_METRIC_MAX_FEATURES = 25;

function FEMultiMetricChart({ leaderboardFeatures }) {
  if (!leaderboardFeatures || leaderboardFeatures.length === 0) return null;

  const topFeatures = leaderboardFeatures.slice(0, FE_MULTI_METRIC_MAX_FEATURES);

  return (
    <section className="card panel-section" style={{ marginTop: 16 }}>
      <div className="section-head">
        <div>
          <p className="section-kicker">Multi-Algorithm</p>
          <h2>Feature Score Comparison</h2>
        </div>
        <span className="pill pill-queued">{topFeatures.length} features</span>
      </div>
      <p className="muted" style={{ marginTop: 4, marginBottom: 12, fontSize: '0.82rem' }}>
        Normalized scores (0-1) per algorithm for each feature. Hover a mini bar for the exact score. Top {FE_MULTI_METRIC_MAX_FEATURES} by combined score shown.
      </p>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 16, flexWrap: 'wrap' }}>
        {VIZ_FE_ALGO_META.map((algo) => (
          <span key={algo.key} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.75rem', color: 'var(--ink-muted)' }}>
            <span style={{ width: 10, height: 10, borderRadius: 2, background: algo.color, display: 'inline-block', flexShrink: 0 }} />
            {algo.label}
          </span>
        ))}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {topFeatures.map((feature) => {
          const isSelected = feature.status === 'selected';
          return (
            <div key={feature.feature} style={{ display: 'grid', gridTemplateColumns: '180px 1fr', gap: 10, alignItems: 'center' }}>
              <span className="mono" style={{
                fontSize: '0.78rem', overflow: 'hidden', textOverflow: 'ellipsis',
                whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', gap: 6,
              }}>
                <span style={{
                  width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                  background: isSelected ? 'var(--ok)' : 'var(--error)', display: 'inline-block',
                }} />
                {feature.feature}
              </span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {VIZ_FE_ALGO_META.map((algo) => {
                  const score = feature.algo_scores?.[algo.key];
                  if (score === undefined) return null;
                  return (
                    <div key={algo.key} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ flex: 1, height: 7, background: 'var(--surface-2)', borderRadius: 3, overflow: 'hidden' }}
                        title={`${algo.label}: ${score.toFixed(3)}`}>
                        <div style={{ width: `${score * 100}%`, height: '100%', background: algo.color, borderRadius: 3 }} />
                      </div>
                      <span className="mono" style={{ fontSize: '0.68rem', color: 'var(--ink-muted)', minWidth: 30, textAlign: 'right' }}>
                        {score.toFixed(2)}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ display: 'flex', gap: 16, marginTop: 14, fontSize: '0.78rem', color: 'var(--ink-muted)' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--ok)', display: 'inline-block' }} />
          Selected by pipeline
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--error)', display: 'inline-block' }} />
          Dropped
        </span>
      </div>
    </section>
  );
}

// ---- Main page ----

function VisualizationPage({ activeSessionId }) {
  const [vizType, setVizType] = useState(VIZ_TYPE_TRAINING);

  // --- Training / Evaluation plots state ---
  const [plots, setPlots] = useState([]);
  const [loadState, setLoadState] = useState('idle');
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationMessage, setGenerationMessage] = useState(null);
  const [generationError, setGenerationError] = useState(null);

  // --- Feature Engineering visuals state ---
  const [feVisuals, setFeVisuals] = useState([]);
  const [feDashboardAvailable, setFeDashboardAvailable] = useState(false);
  const [feVisualsLoaded, setFeVisualsLoaded] = useState(false);
  const [feVisualsLoading, setFeVisualsLoading] = useState(false);
  const [feVisualsError, setFeVisualsError] = useState(null);
  const [feLeaderboardFeatures, setFeLeaderboardFeatures] = useState([]);

  // Reset per-type state when the session changes.
  useEffect(() => {
    setPlots([]);
    setLoadState('idle');
    setRefreshTrigger(0);
    setFeVisuals([]);
    setFeDashboardAvailable(false);
    setFeVisualsLoaded(false);
    setFeVisualsError(null);
    setFeLeaderboardFeatures([]);
    setGenerationMessage(null);
    setGenerationError(null);
  }, [activeSessionId]);

  // Fetch training/evaluation plots once on mount and after each refresh trigger.
  // Plots are pre-generated during the pipeline run, so a single fetch suffices.
  useEffect(() => {
    if (!activeSessionId || vizType !== VIZ_TYPE_TRAINING) return undefined;

    let cancelled = false;
    setLoadState('loading');

    fetchPlots(activeSessionId)
      .then((data) => {
        if (!cancelled) {
          setPlots(data?.plots || []);
          setLoadState('done');
        }
      })
      .catch(() => {
        if (!cancelled) setLoadState('error');
      });

    return () => { cancelled = true; };
  }, [activeSessionId, vizType, refreshTrigger]);

  // Reset FE state when switching away from FE tab so a fresh generate is needed.
  useEffect(() => {
    if (vizType !== VIZ_TYPE_FEATURE_ENGINEERING) {
      setFeVisuals([]);
      setFeDashboardAvailable(false);
      setFeVisualsLoaded(false);
      setFeVisualsError(null);
      setFeLeaderboardFeatures([]);
    }
    setGenerationMessage(null);
    setGenerationError(null);
  }, [vizType]);

  const handleGenerateTraining = async () => {
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

  const handleGenerateFE = async () => {
    setFeVisualsError(null);
    setGenerationMessage(null);
    setGenerationError(null);
    setFeVisualsLoading(true);
    try {
      // Fetch visuals and leaderboard data in parallel.
      const [result, leaderboardResult] = await Promise.all([
        fetchFEVisuals(activeSessionId),
        fetchFeatureLeaderboard(activeSessionId).catch(() => null),
      ]);
      setFeVisuals(result?.visuals || []);
      setFeDashboardAvailable(result?.dashboard_available || false);
      setFeVisualsLoaded(true);
      setFeLeaderboardFeatures(leaderboardResult?.features || []);
      if (!result?.visuals?.length) {
        setFeVisualsError('No feature engineering visuals found. Run the Feature Engineering pipeline first.');
      }
    } catch (err) {
      setFeVisualsError(err.message || 'Failed to load feature engineering visuals.');
    } finally {
      setFeVisualsLoading(false);
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

  const isFeTab = vizType === VIZ_TYPE_FEATURE_ENGINEERING;
  const groupedPlots = groupPlotsByStage(plots);

  const buttonBusy = isFeTab ? feVisualsLoading : isGenerating;
  const buttonLabel = isFeTab
    ? (feVisualsLoaded ? 'Refresh Visuals' : 'Generate Visuals')
    : (plots.length > 0 ? 'Refresh Visualizations' : 'Generate Visualizations');
  const handleButtonClick = isFeTab ? handleGenerateFE : handleGenerateTraining;

  // Count shown in hero StatusPill
  const feChartCount = feVisuals.length;
  const heroPillStatus = isFeTab
    ? (feVisualsLoaded && feChartCount > 0 ? 'done' : 'queued')
    : (plots.length > 0 ? 'done' : 'queued');
  const heroPillLabel = isFeTab ? `${feChartCount} charts` : `${plots.length} plots`;

  return (
    <div className="screen-stack">
      {/* Hero panel */}
      <section className="card hero-panel" style={{ paddingBottom: 14, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 15 }}>
        <div>
          <StatusPill status={heroPillStatus} label={heroPillLabel} />
          <h2>Visualizations</h2>
          <p className="muted">
            {isFeTab
              ? 'Interactive charts from the feature engineering pipeline grouped by analysis category.'
              : 'Visual analytics generated at the end of the pipeline. Click the button if plots are not showing.'}
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <select
            className="viz-type-select"
            onChange={(event) => setVizType(event.target.value)}
            value={vizType}
          >
            {VIZ_TYPE_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
          <button
            className="btn btn-primary"
            disabled={buttonBusy}
            onClick={handleButtonClick}
            style={{ display: 'flex', alignItems: 'center', gap: 8 }}
            type="button"
          >
            {buttonBusy ? <div className="spinner small" /> : <Icons.spark size={15} />}
            {buttonBusy ? 'Loading...' : buttonLabel}
          </button>
        </div>
      </section>

      {/* Loading spinner for in-progress operations */}
      {(isGenerating || feVisualsLoading) && (
        <div className="callout compact" style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 10 }}>
          <div className="spinner small" />
          <span>{isFeTab ? 'Loading feature engineering visuals...' : 'Generating visualizations...'}</span>
        </div>
      )}

      {/* Compact auto-dismissing toast for success messages */}
      <Toast
        message={generationMessage}
        tone="success"
        onDismiss={() => setGenerationMessage(null)}
      />

      {(generationError || feVisualsError) && (
        <div className="callout error compact" style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
          <Icons.x size={16} style={{ color: 'var(--error)' }} />
          <span>{generationError || feVisualsError}</span>
        </div>
      )}

      {/* Feature Engineering visuals — grouped sections */}
      {isFeTab && !feVisualsLoading && feVisualsLoaded && (
        <>
          <FEMultiMetricChart leaderboardFeatures={feLeaderboardFeatures} />
          <FEVisualsGallery
            dashboardAvailable={feDashboardAvailable}
            sessionId={activeSessionId}
            visuals={feVisuals}
          />
        </>
      )}

      {isFeTab && !feVisualsLoaded && !feVisualsLoading && !feVisualsError && (
        <section className="card panel-section" style={{ marginTop: 20 }}>
          <div className="section-head">
            <div>
              <p className="section-kicker">Feature Engineering</p>
              <h2>Pipeline Charts</h2>
            </div>
          </div>
          <div className="callout compact" style={{ marginTop: 12 }}>
            <span>Click <strong>Generate Visuals</strong> to load the feature engineering pipeline charts for this session.</span>
          </div>
        </section>
      )}

      {/* Training / Evaluation plots — stage sections */}
      {!isFeTab && (
        <>
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
            <section className="card panel-section" style={{ marginTop: 20 }}>
              <div className="section-head">
                <div>
                  <p className="section-kicker">Training & Evaluation</p>
                  <h2>Visual Analytics</h2>
                </div>
              </div>
              <div className="callout compact" style={{ marginTop: 12 }}>
                <span>Plots are generated automatically at the end of the training pipeline. Click <strong>Generate Visualizations</strong> if the pipeline has completed and no plots appear yet.</span>
              </div>
            </section>
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
        </>
      )}
    </div>
  );
}


export default VisualizationPage;
