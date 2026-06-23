import { useCallback, useEffect, useMemo, useState } from 'react';

import AgentAvatar from '../components/AgentAvatar.jsx';
import HBars from '../components/HBars.jsx';
import StatusPill from '../components/StatusPill.jsx';
import Stepper from '../components/Stepper.jsx';
import {
  fetchLeaderboard,
  fetchShap,
  fetchTokens,
  fetchVerdict,
  fetchHpt,
  runHpt,
  modelDownloadUrl,
  modelsDownloadAllUrl,
} from '../api/client.js';
import { streamEvaluationEvents } from '../api/events.js';
import { useBoundedPoll } from '../hooks/useBoundedPoll.js';
import { AGENTS, LEADERBOARD, SHAP } from '../data.js';
import { Icons } from '../icons.jsx';

const judgeAgent = AGENTS.find((agent) => agent.id === 'judge');
const featureAgent = AGENTS.find((agent) => agent.id === 'feature');

// Bounded polling so the leaderboard syncs as post-training artifacts land.
const LEADERBOARD_POLL_MS = 3000;
const LEADERBOARD_MAX_POLLS = 80;

// Ordered metric keys for classification and regression, used to dynamically
// render all available metrics from the leaderboard payload.
const CLASSIFICATION_METRIC_KEYS = [
  { key: 'accuracy',       label: 'Accuracy'   },
  { key: 'f1_macro',       label: 'F1 Macro'   },
  { key: 'f1_weighted',    label: 'F1 Weighted'},
  { key: 'precision_macro',label: 'Precision'  },
  { key: 'recall_macro',   label: 'Recall'     },
];
const REGRESSION_METRIC_KEYS = [
  { key: 'r2',   label: 'R2'   },
  { key: 'mae',  label: 'MAE'  },
  { key: 'mse',  label: 'MSE'  },
  { key: 'rmse', label: 'RMSE' },
];

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '--';
  }
  return Number(value).toFixed(digits);
}

// Map a Judge finding status to its marker icon + colour. Keeps unicode glyphs
// out of the data layer (backend emits 'pass'/'fail'/'info' only).
const FINDING_STATUS_MARKERS = {
  pass: { Icon: Icons.checkCircle, color: 'var(--ok)' },
  fail: { Icon: Icons.x, color: 'var(--err)' },
  info: { Icon: Icons.info, color: 'var(--muted, #888)' },
};

// Map a model decision to a pill style.
const DECISION_PILL_STYLES = {
  APPROVED: { background: 'rgba(34,197,94,0.14)', color: 'var(--ok)', border: '1px solid rgba(34,197,94,0.3)' },
  RANKED:   { background: 'rgba(234,179,8,0.14)', color: '#eab308', border: '1px solid rgba(234,179,8,0.3)' },
  REJECTED: { background: 'rgba(239,68,68,0.14)', color: 'var(--err)', border: '1px solid rgba(239,68,68,0.3)' },
  PENDING:  { background: 'rgba(148,163,184,0.14)', color: 'var(--muted, #888)', border: '1px solid rgba(148,163,184,0.3)' },
};

// One model governance card: decision pill + per-dimension findings + optional
// ranking explanation. Renders the SPEC's "Model Decision Card".
function ModelDecisionCard({ row }) {
  const findings = row.findings || [];
  if (findings.length === 0) return null;
  const decision = row.decision || (row.winner ? 'APPROVED' : 'PENDING');
  const pillStyle = DECISION_PILL_STYLES[decision] || DECISION_PILL_STYLES.PENDING;

  return (
    <div
      className="decision-card"
      style={{
        background: 'rgba(255,255,255,0.02)',
        // Border colour mirrors the decision: green = APPROVED, red = REJECTED,
        // neutral = PENDING (reuses the pill's per-decision border colour).
        border: pillStyle.border || '1px solid rgba(255,255,255,0.06)',
        borderRadius: 8,
        padding: 16,
        display: 'flex',
        flexDirection: 'column',
        gap: 10,
      }}
    >
      {/* Top row: rank on the left, decision pill on the right. Keeping these on
          their own line stops a long model name from squeezing them off-card. */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
        <span className="mono" style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--ink)' }}>Rank #{row.rank}</span>
        <span
          className="pill"
          style={{ ...pillStyle, fontWeight: 700, fontSize: '0.72rem', padding: '2px 10px', borderRadius: 5, whiteSpace: 'nowrap', flexShrink: 0 }}
        >
          {decision}
        </span>
      </div>
      {/* Model name on its own full-width line so long names wrap cleanly. */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, minWidth: 0 }}>
        {row.winner && row.rank === 1 ? (
          <Icons.trophy size={16} />
        ) : row.winner ? (
          <Icons.checkCircle size={16} />
        ) : null}
        <strong style={{ fontSize: '1rem', overflowWrap: 'anywhere' }}>{row.model_name || row.model}</strong>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <p className="section-kicker" style={{ margin: 0, fontSize: '0.65rem' }}>Judge Findings</p>
        {findings.map((finding) => {
          const marker = FINDING_STATUS_MARKERS[finding.status] || FINDING_STATUS_MARKERS.info;
          const MarkerIcon = marker.Icon;
          return (
            <div key={finding.dimension} style={{ display: 'flex', alignItems: 'flex-start', gap: 8, fontSize: '0.82rem' }}>
              <span style={{ color: marker.color, flexShrink: 0, marginTop: 1 }} aria-hidden="true">
                {MarkerIcon ? <MarkerIcon size={15} /> : null}
              </span>
              {/* Status is otherwise conveyed only by icon + colour. */}
              <span className="sr-only">{finding.status}: </span>
              <span>
                <strong style={{ color: marker.color }}>{finding.label}:</strong>{' '}
                <span style={{ color: 'var(--ink)' }}>{finding.message}</span>
              </span>
            </div>
          );
        })}
      </div>

      {row.ranking_explanation ? (
        <details style={{ cursor: 'pointer', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 8 }}>
          <summary style={{ outline: 'none', fontWeight: 600, fontSize: '0.78rem', color: 'var(--ink)' }}>
            Ranking justification (rule-engine)
          </summary>
          <pre className="reasoning-block" style={{ marginTop: 8, background: 'rgba(0,0,0,0.2)', border: 'none', whiteSpace: 'pre-wrap' }}>
            {row.ranking_explanation}
          </pre>
        </details>
      ) : null}

      {row.llm_ranking_reasoning ? (
        <details style={{ cursor: 'pointer', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 8 }}>
          <summary style={{ outline: 'none', fontWeight: 600, fontSize: '0.78rem', color: 'var(--ink)' }}>
            LLM ranking rationale (incl. SHAP / domain correlation)
          </summary>
          <pre className="reasoning-block" style={{ marginTop: 8, background: 'rgba(0,0,0,0.2)', border: 'none', whiteSpace: 'pre-wrap' }}>
            {row.llm_ranking_reasoning}
          </pre>
        </details>
      ) : null}
    </div>
  );
}

// Detect whether the session uses classification or regression metrics by
// probing the first model's metrics dict. Falls back to a minimal schema
// for prototype data that stores metrics directly on the row object.
function detectMetricSchema(models) {
  // Probe the first row that actually has a populated metrics dict. The rank-1
  // row can be a gate-rejected model with empty metrics, which previously caused
  // a real run to fall back to the prototype acc/f1/auc columns.
  const rowWithMetrics =
    models.find((row) => row?.metrics && Object.keys(row.metrics).length > 0) || models[0] || {};
  const firstMetrics = rowWithMetrics.metrics || {};

  if (CLASSIFICATION_METRIC_KEYS.some(({ key }) => key in firstMetrics)) {
    return CLASSIFICATION_METRIC_KEYS;
  }
  if (REGRESSION_METRIC_KEYS.some(({ key }) => key in firstMetrics)) {
    return REGRESSION_METRIC_KEYS;
  }
  // Prototype data: pick two visible columns using top-level row keys.
  return [
    { key: 'acc',   label: 'Score'   },
    { key: 'f1',    label: 'F1'      },
    { key: 'auc',   label: 'AUC'     },
  ];
}

function formatTime(timestamp) {
  if (!timestamp) return '--:--:--';
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return '--:--:--';
  return date.toLocaleTimeString([], { hour12: false });
}

function LeaderboardScreen({ activeSessionId, go, startRun }) {
  const [leaderboardData, setLeaderboardData] = useState(null);
  const [shapData, setShapData] = useState(null);
  const [shapModelName, setShapModelName] = useState(null);
  const [verdictData, setVerdictData] = useState(null);
  const [tokenData, setTokenData] = useState(null);
  const [loadState, setLoadState] = useState('idle');
  const [hptData, setHptData] = useState(null);
  const [hptStatus, setHptStatus] = useState('idle'); // 'idle' | 'running' | 'complete' | 'failed'
  const [hptProgress, setHptProgress] = useState(0);
  const [hptMessage, setHptMessage] = useState('');
  const [hptLogs, setHptLogs] = useState([]);
  const [hptTrialNum, setHptTrialNum] = useState(0);
  const [hptTotalTrials, setHptTotalTrials] = useState(5);
  const [hptBestScore, setHptBestScore] = useState(null);
  const [hptError, setHptError] = useState(null);
  // True only during the POST /hpt/run round-trip, before hptStatus flips to
  // 'running'. Used to disable the Tune/Re-tune button and show a starting
  // spinner without prematurely opening the SSE subscription (see handleRunHpt).
  const [isStartingHpt, setIsStartingHpt] = useState(false);
  // User-configurable HPT run parameters. topN defaults to 3 per product
  // requirement; numTrials mirrors the backend default of 5.
  const [hptTopN, setHptTopN] = useState(3);
  const [hptNumTrials, setHptNumTrials] = useState(5);


  // Main page poll (bounded): fetches leaderboard + SHAP + verdict + tokens as
  // post-training artifacts land. `fetchLeaderboard` throwing is a real error
  // (counted toward give-up); the secondary reads degrade gracefully. HPT status
  // is NOT forced here -- it is owned by handleRunHpt + the SSE/checkHpt loop, so
  // the background poll can't reset a user-initiated 'running' (was a bug).
  const pollLeaderboard = useCallback(async () => {
    const leaderboard = await fetchLeaderboard(activeSessionId);
    const shap = await fetchShap(activeSessionId).catch(() => null);
    const verdict = await fetchVerdict(activeSessionId).catch(() => null);
    const tokens = await fetchTokens(activeSessionId).catch(() => null);
    const hpt = await fetchHpt(activeSessionId).catch(() => null);

    setLeaderboardData(leaderboard);
    const features = (shap?.features || []).map((item) => ({
      feature: item.feature,
      value: item.importance,
    }));
    setShapData(features.length ? features : null);
    setShapModelName(shap?.model_name || null);
    setVerdictData(verdict?.status && verdict.status !== 'pending' ? verdict : null);
    setTokenData(tokens?.status === 'complete' ? tokens : null);

    // Restore a previously-completed HPT result on (re)load without overriding a
    // tuning run the user just started.
    if (hpt?.status === 'complete' && hpt?.hpt_results?.length) {
      setHptData(hpt.hpt_results);
      setHptStatus((prev) => (prev === 'running' ? prev : 'complete'));
    }

    setLoadState('done');
    return { leaderboard, shap, verdict, tokens };
  }, [activeSessionId]);

  const { pollState: leaderboardPollState, restart: restartLeaderboardPoll } = useBoundedPoll(
    pollLeaderboard,
    {
      enabled: Boolean(activeSessionId),
      intervalMs: LEADERBOARD_POLL_MS,
      maxAttempts: LEADERBOARD_MAX_POLLS,
      maxErrorAttempts: 6,
      stopWhen: (res) => {
        const lbDone = res?.leaderboard?.status === 'complete';
        const shapDone = !res?.shap || res.shap.status === 'complete';
        const verdictDone = !res?.verdict || res.verdict.status === 'complete' || res.verdict.status === 'failed';
        const tokensDone = !res?.tokens || res.tokens.status === 'complete' || res.tokens.status === 'failed';
        return lbDone && shapDone && verdictDone && tokensDone;
      },
      resetKey: activeSessionId,
    },
  );

  // Surface load/error state from the bounded poll for the UI.
  useEffect(() => {
    if (!activeSessionId) {
      setLoadState('idle');
      return;
    }
    if (leaderboardPollState === 'gave_up') {
      setLoadState('error');
    } else if (leaderboardPollState === 'polling') {
      setLoadState((prev) => (prev === 'done' ? prev : 'loading'));
    }
  }, [activeSessionId, leaderboardPollState]);

  // Subscribe to SSE for real-time HPT stage progress updates ONLY while a tuning
  // run is active. The session is closed at the end of training, so subscribing
  // outside an HPT run yields an immediate end-of-stream and a native
  // EventSource reconnect storm. Gating on hptStatus==='running' opens the
  // stream when the user starts HPT and closes it (cleanup) when tuning ends.
  useEffect(() => {
    if (!activeSessionId || hptStatus !== 'running') return undefined;
    const source = streamEvaluationEvents(activeSessionId, {
      onEvent: (event) => {
        if (event?.stage !== 'hpt') return;
        setHptProgress(event.pct ?? 0);
        setHptMessage(event.msg ?? '');
        
        if (event.msg) {
          setHptLogs((prev) => {
            const exists = prev.some((log) => log.ts === event.ts && log.message === event.msg);
            if (exists) return prev;
            return [...prev, { ts: event.ts || new Date().toISOString(), message: event.msg }];
          });
        }

        if (event.details) {
          if (event.details.trial_number) setHptTrialNum(event.details.trial_number);
          if (event.details.total_trials) setHptTotalTrials(event.details.total_trials);
          if (event.details.best_score != null) setHptBestScore(event.details.best_score);
        }

        if (event.status === 'running') {
          setHptStatus('running');
        } else if (event.status === 'all_completed') {
          setHptStatus('complete');
          setHptProgress(100);
          // Re-fetch leaderboard so HPT best_params appear in winner row immediately
          fetchLeaderboard(activeSessionId)
            .then((lb) => setLeaderboardData(lb))
            .catch(() => {});
          // Fetch hpt results
          fetchHpt(activeSessionId)
            .then((data) => {
              if (data?.hpt_results?.length) setHptData(data.hpt_results);
            })
            .catch(() => {});
        } else if (event.status === 'failed') {
          setHptStatus('failed');
          setHptProgress(0);
        }
      },
    });
    return () => source?.close?.();
  }, [activeSessionId, hptStatus]);


  // HPT completion poll (bounded): a fallback to the SSE stream that resolves the
  // running -> complete/failed transition. Bounded so it never loops forever on
  // a persistently failing /hpt endpoint.
  const pollHpt = useCallback(async () => {
    const data = await fetchHpt(activeSessionId);
    if (data?.status === 'complete' && data?.hpt_results) {
      setHptData(data.hpt_results);
      setHptStatus('complete');
    } else if (data?.status === 'failed') {
      setHptStatus('failed');
    }
    return data;
  }, [activeSessionId]);

  useBoundedPoll(pollHpt, {
    enabled: Boolean(activeSessionId) && hptStatus === 'running',
    intervalMs: 2000,
    maxErrorAttempts: 6,
    stopWhen: (data) => ['complete', 'failed'].includes(data?.status),
    resetKey: activeSessionId,
  });

  const handleRunHpt = async () => {
    try {
      setHptError(null);
      setIsStartingHpt(true);
      setHptProgress(0);
      setHptMessage('Starting Optuna hyperparameter tuning...');
      setHptLogs([]);
      setHptTrialNum(0);
      setHptTotalTrials(hptNumTrials);
      setHptBestScore(null);
      // Clear prior results so a re-tune with different top-N/trials doesn't
      // show stale cards from the previous run while the new one is in flight.
      setHptData(null);
      await runHpt(activeSessionId, { topN: hptTopN, numTrials: hptNumTrials });
      // Only flip to 'running' (which triggers the SSE subscription effect)
      // AFTER the backend confirms the run started. The backend calls
      // event_bus.reset_session() synchronously before responding, so by now
      // the session is guaranteed to be un-marked as closed. Flipping this
      // earlier raced the SSE subscribe against that reset: on a re-tune the
      // prior run had already closed the session, so a subscribe attempt that
      // beat reset_session() got an immediate empty stream with no retry
      // until the native EventSource's ~3s auto-reconnect -- often after the
      // run had already finished, so the events viewer never reappeared.
      setHptStatus('running');
    } catch (err) {
      console.error(err);
      setHptStatus('failed');
      setHptProgress(0);
      setHptMessage('Failed to start tuning process.');
      // Surface the real backend error so the user can act on it.
      setHptError(err?.message || 'Failed to start hyperparameter tuning.');
    } finally {
      setIsStartingHpt(false);
    }
  };


  const models = leaderboardData?.models || [];
  // Distinguish a real session that simply has no models yet from the demo:
  // prototype data is shown ONLY when there is no active session at all. A real
  // session with zero models renders an explicit empty state instead of fake
  // XGBoost/LightGBM rows.
  const hasLiveModels = models.length > 0;
  const isRealSession = Boolean(activeSessionId);
  const usingLive = hasLiveModels;
  const showPrototype = !isRealSession && !hasLiveModels;
  const showEmptyState = isRealSession && !hasLiveModels;
  const displayRows = hasLiveModels ? models : showPrototype ? LEADERBOARD : [];
  const selectedModel = leaderboardData?.selected_model || null;
  // Plural selected set (top-N% judge selection). Falls back to the singular
  // field for older judge_decision.json payloads that predate this field.
  const selectedModels = leaderboardData?.selected_models?.length
    ? leaderboardData.selected_models
    : selectedModel
      ? [selectedModel]
      : [];
  const decisionTrace = leaderboardData?.decision_trace || verdictData?.decision_trace || null;
  const comparisonExplanation =
    leaderboardData?.comparison_explanation || verdictData?.comparison_explanation || null;
  // Rows that carry structured Judge findings (governance dashboard cards).
  const decisionCardRows = useMemo(
    () => displayRows.filter((row) => (row.findings || []).length > 0),
    [displayRows],
  );
  const metricSchema = useMemo(() => detectMetricSchema(displayRows), [displayRows]);
  // Build the grid column template so header + rows always have the same number
  // of columns as the dynamically rendered cells (Rank + Model + N metrics +
  // Overfit + Judge + optional Download). Supplied to the table via a CSS var.
  const leaderboardGridColumns = useMemo(() => {
    const columns = [
      '60px', // Rank
      'minmax(0, 1.4fr)', // Model
      ...metricSchema.map(() => 'minmax(64px, 0.6fr)'), // one per metric
      'minmax(80px, 0.7fr)', // Overfit
      'minmax(56px, 0.5fr)', // Judge
    ];
    if (usingLive) {
      columns.push('52px'); // Download
    }
    return columns.join(' ');
  }, [metricSchema, usingLive]);
  const winnerRow = displayRows.find((row) => row.winner) || displayRows[0];
  const winnerLabel = usingLive ? (selectedModel || winnerRow?.model_name || winnerRow?.model) : winnerRow?.model;
  // The full selected set, in display order, for the multi-model hero copy.
  const selectedRows = useMemo(
    () => displayRows.filter((row) => selectedModels.includes(row.model_name || row.model)),
    [displayRows, selectedModels],
  );

  // Build a list of reasons for the selected model from the judge verdict.
  const winnerReasons = useMemo(() => {
    if (!verdictData) return [];
    const ranked = verdictData.ranked_models || [];
    const winnerRecord = ranked.find((m) => m.model_name === selectedModel) || ranked[0];
    return winnerRecord?.reasons || [];
  }, [verdictData, selectedModel]);

  // Hero status pill reflects the real state instead of always claiming the
  // judge converged.
  const judgeConverged =
    verdictData?.status === 'complete' || leaderboardData?.status === 'complete';
  let heroPill;
  if (showPrototype) {
    heroPill = { status: 'idle', label: 'Demo data' };
  } else if (judgeConverged && hasLiveModels) {
    heroPill = { status: 'done', label: 'Judge converged' };
  } else if (loadState === 'error') {
    heroPill = { status: 'failed', label: 'Could not load results' };
  } else {
    heroPill = { status: 'running', label: 'Awaiting judge' };
  }

  // Workflow breadcrumb for moving back through the run lifecycle.
  const runSteps = [
    { key: 'upload', label: 'New Run', route: 'upload', status: 'done', enabled: true },
    { key: 'features', label: 'Features', route: 'features', status: 'done', enabled: true },
    { key: 'pipeline', label: 'Training', route: 'pipeline', status: 'done', enabled: true },
    { key: 'leaderboard', label: 'Leaderboard', route: 'leaderboard', status: 'active', enabled: true },
  ];

  return (
    <div className="screen-stack">
      <div className="card panel-section" style={{ padding: '10px 14px' }}>
        <Stepper steps={runSteps} onNavigate={(step) => go?.(step.route)} />
      </div>
      {/* Hero banner */}
      <section className="card hero-panel leaderboard-hero">
        <div className="winner-mark">
          <Icons.trophy size={28} />
        </div>
        <div>
          <StatusPill status={heroPill.status} label={heroPill.label} spin={heroPill.status === 'running'} />
          <h2>
            {!usingLive
              ? winnerLabel ? `${winnerLabel} is the recommended model` : 'Awaiting judge verdict'
              : selectedRows.length === 0
                ? 'Awaiting judge verdict'
                : selectedRows.length === 1
                  ? `${selectedRows[0].model_name || selectedRows[0].model} is the recommended model`
                  : `${selectedRows.length} models selected by the judge`}
          </h2>
          {usingLive && selectedRows.length > 1 ? (
            <p className="muted" style={{ margin: '2px 0 0 0' }}>
              Top pick: <strong>{selectedRows[0].model_name || selectedRows[0].model}</strong>
            </p>
          ) : null}
          <p className="muted">
            {showPrototype
              ? 'Sample leaderboard shown because no training run is active.'
              : hasLiveModels
                ? 'Live results for the current training run.'
                : 'No model results for this run yet.'}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {usingLive && activeSessionId ? (
            <a
              className="btn btn-secondary"
              download
              href={modelsDownloadAllUrl(activeSessionId)}
            >
              <Icons.download size={15} />
              Download all models
            </a>
          ) : null}
          <button className="btn btn-secondary" onClick={() => go?.('dashboard')} type="button">
            <Icons.grid size={16} />
            Dashboard
          </button>
          {/* Navigate back to the training view WITHOUT re-marking a finished run
              as running (the old bare startRun() did exactly that). */}
          <button className="btn btn-primary" onClick={() => go?.('pipeline')} type="button">
            <Icons.arrowLeft size={16} />
            View training
          </button>
        </div>
      </section>

      {/* Hyperparameter Tuning Section — moved above the leaderboard table so
          the user sees and can configure it first. */}
      {usingLive && activeSessionId && (
        <section className="card panel-section" style={{ borderLeft: '4px solid #ec4899', display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
              <div className="stage-card-icon" style={{ background: 'rgba(236, 72, 153, 0.15)', color: 'var(--hpt)', width: 36, height: 36, borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Icons.cpu size={20} />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '1.15rem' }}>Hyperparameter Tuning (Optuna HPT)</h3>
                <p className="muted" style={{ margin: '4px 0 0 0', fontSize: '0.85rem' }}>
                  {isStartingHpt && 'Starting Optuna hyperparameter tuning...'}
                  {!isStartingHpt && hptStatus === 'idle' && `Run Optuna HPT on the top-${hptTopN} Judge-selected model(s) (${hptNumTrials} trials each). Results appear in leaderboard.`}
                  {!isStartingHpt && hptStatus === 'running' && (hptMessage || `Tuning top-${hptTopN} model(s) (${hptNumTrials} Optuna trials each)...`)}
                  {!isStartingHpt && hptStatus === 'complete' && "HPT completed. Best hyperparameters and score are now in the leaderboard winner row."}
                  {!isStartingHpt && hptStatus === 'failed' && (hptError || "Hyperparameter tuning execution failed.")}
                </p>
              </div>
            </div>
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 12, flexWrap: 'wrap' }}>
              {/* Controls stay visible (and editable) after completion too, so the
                  user can change top-N / trials and re-tune without losing context. */}
              {hptStatus !== 'running' && (
                <>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '0.72rem', color: 'var(--ink-muted)' }}>
                    Top models
                    <select
                      className="input"
                      style={{ minWidth: 84, minHeight: 32, padding: '4px 8px' }}
                      value={hptTopN}
                      disabled={isStartingHpt}
                      onChange={(event) => setHptTopN(Number(event.target.value))}
                    >
                      {[1, 2, 3, 5, 10].map((count) => (
                        <option key={count} value={count}>{count}</option>
                      ))}
                    </select>
                  </label>
                  <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: '0.72rem', color: 'var(--ink-muted)' }}>
                    Trials per model
                    <input
                      type="number"
                      className="input"
                      style={{ width: 76, minHeight: 32, padding: '4px 8px' }}
                      min={1}
                      max={50}
                      value={hptNumTrials}
                      disabled={isStartingHpt}
                      onChange={(event) => {
                        const parsed = Number(event.target.value);
                        setHptNumTrials(Number.isFinite(parsed) ? Math.min(50, Math.max(1, parsed)) : 5);
                      }}
                    />
                  </label>
                </>
              )}
              {isStartingHpt && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div className="spinner small" />
                  <span className="mono" style={{ fontSize: '0.85rem', color: 'var(--hpt)', fontWeight: 600 }}>Starting...</span>
                </div>
              )}
              {!isStartingHpt && hptStatus === 'idle' && (
                <button className="btn btn-primary" onClick={handleRunHpt} style={{ background: '#ec4899', borderColor: '#ec4899' }} type="button">
                  <Icons.spark size={15} /> Tune Hyperparameters
                </button>
              )}
              {!isStartingHpt && hptStatus === 'running' && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div className="spinner small" />
                  <span className="mono" style={{ fontSize: '0.85rem', color: 'var(--hpt)', fontWeight: 600 }}>{hptProgress}%</span>
                </div>
              )}
              {!isStartingHpt && hptStatus === 'complete' && (
                <>
                  <span className="pill pill-done" style={{ background: 'rgba(236, 72, 153, 0.15)', color: 'var(--hpt)', border: '1px solid rgba(236, 72, 153, 0.3)', fontWeight: 600 }}>
                    Tuned
                  </span>
                  <button className="btn btn-secondary" onClick={handleRunHpt} type="button">
                    <Icons.spark size={15} /> Re-tune
                  </button>
                </>
              )}
              {!isStartingHpt && hptStatus === 'failed' && (
                <button className="btn btn-secondary" onClick={handleRunHpt} type="button">
                  Retry Tuning
                </button>
              )}
            </div>
          </div>

          {/* Live progress bar & trial stats - visible while HPT is running */}
          {hptStatus === 'running' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', color: 'var(--ink-muted)', fontWeight: 500 }}>
                <div>
                  Trial <span className="mono" style={{ color: 'var(--hpt)', fontWeight: 700 }}>{hptTrialNum}</span> / {hptTotalTrials}
                </div>
                {hptBestScore !== null && (
                  <div>
                    Best Score: <span className="mono" style={{ color: '#be185d', fontWeight: 700 }}>{formatNumber(hptBestScore, 4)}</span>
                  </div>
                )}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                <span className="muted mono" style={{ maxWidth: '80%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {hptMessage || 'Initialising Optuna study...'}
                </span>
                <strong className="mono" style={{ color: 'var(--hpt)' }}>{hptProgress}%</strong>
              </div>
              <div
                style={{ height: 6, background: 'var(--line-soft)', borderRadius: 3, overflow: 'hidden' }}
                role="progressbar"
                aria-valuenow={Math.round(hptProgress)}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-label="Hyperparameter tuning progress"
              >
                <div style={{ width: `${hptProgress}%`, height: '100%', background: 'linear-gradient(90deg, var(--hpt) 0%, #ec4899 100%)', borderRadius: 3, transition: 'width 0.4s ease' }} />
              </div>
            </div>
          )}

          {/* Scrollable event log viewer */}
          {hptLogs && hptLogs.length > 0 && (
            <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 6 }}>
              <p className="section-kicker" style={{ marginBottom: 0, fontSize: '0.75rem' }}>Optuna Trial Events Stream</p>
              <div
                className="terminal-body"
                style={{
                  height: 140,
                  overflowY: 'auto',
                  background: 'rgba(0,0,0,0.3)',
                  border: '1px solid rgba(255,255,255,0.05)',
                  borderRadius: 6,
                  padding: 10,
                  fontSize: '0.75rem',
                  lineHeight: '1.4',
                  fontFamily: 'monospace'
                }}
                ref={(el) => {
                  if (el) {
                    el.scrollTop = el.scrollHeight;
                  }
                }}
              >
                {hptLogs.map((log, index) => (
                  <div key={index} style={{ color: 'var(--ink-muted)', marginBottom: 4, display: 'flex', gap: 8 }}>
                    <span style={{ color: 'var(--ink-faint)' }}>{formatTime(log.ts)}</span>
                    <span style={{ color: 'var(--hpt)', fontWeight: 600 }}>[HPT]</span>
                    <span style={{ whiteSpace: 'pre-wrap' }}>{log.message}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {hptData && hptData.length > 0 && (
            <div className="hpt-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 15 }}>
              {hptData.map((model) => (
                <div className="hpt-card" key={model.name} style={{
                  background: 'rgba(255, 255, 255, 0.02)',
                  border: '1px solid rgba(255, 255, 255, 0.06)',
                  borderRadius: 8,
                  padding: 16,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 8
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h4 style={{ margin: 0, fontSize: '0.975rem' }}>{model.name}</h4>
                    <span className="mono text-xs muted">{model.n_trials} trials</span>
                  </div>
                  <div style={{ display: 'flex', gap: 12, fontSize: '0.8rem', color: 'var(--ink-muted)' }}>
                    <div>Tuning time: <span className="mono">{model.tuning_time_seconds ? model.tuning_time_seconds.toFixed(1) : 'N/A'}s</span></div>
                    <div>Best trial: <span className="mono">#{model.best_trial_number}</span></div>
                  </div>
                  <div style={{ marginTop: 4 }}>
                    <p className="section-kicker" style={{ marginBottom: 4, fontSize: '0.675rem' }}>Best Hyperparameters</p>
                    <div style={{
                      background: 'rgba(0,0,0,0.2)',
                      padding: 8,
                      borderRadius: 4,
                      maxHeight: 100,
                      overflowY: 'auto'
                    }}>
                      <pre className="mono" style={{ margin: 0, fontSize: '0.75rem', whiteSpace: 'pre-wrap', color: 'var(--ink-muted)' }}>
                        {JSON.stringify(model.best_hyperparameters, null, 2)}
                      </pre>
                    </div>
                  </div>
                  {model.val_metrics ? (
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: '0.8rem' }}>
                      <span>Validation Score:</span>
                      <strong className="mono" style={{ color: 'var(--hpt)' }}>
                        {Object.entries(model.val_metrics).map(([metric, val]) => `${metric}=${val.toFixed(4)}`).join(', ')}
                      </strong>
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {/* Empty state: a real run that has produced no ranked models yet. Shown
          instead of fabricated prototype rows so users aren't misled. */}
      {showEmptyState ? (
        <section className="card empty-card" style={{ textAlign: 'center' }}>
          <Icons.trophy size={30} />
          <h2>No model results yet</h2>
          <p className="muted">
            {leaderboardPollState === 'gave_up'
              ? 'We could not load results for this run. Check that the run completed, then refresh.'
              : leaderboardPollState === 'capped'
                ? 'Still waiting on results. Training and the judge may still be running.'
                : 'This run has not produced ranked models yet. Results appear here once training and the judge finish.'}
          </p>
          <button className="btn btn-secondary" onClick={restartLeaderboardPoll} type="button">
            <Icons.activity size={15} /> Refresh
          </button>
        </section>
      ) : null}

      {/* Main leaderboard table */}
      {showEmptyState ? null : (
      <section className="card panel-section">
        <div className="section-head">
          <div>
            <p className="section-kicker">Models</p>
            <h2>Leaderboard</h2>
          </div>
          {loadState === 'loading' ? <StatusPill status="running" spin /> : null}
        </div>

        {/* Sync stopped before the run reached a terminal state -- let the user
            re-sync instead of leaving stale data with no recourse. */}
        {hasLiveModels && ['gave_up', 'capped'].includes(leaderboardPollState) ? (
          <div className="inline-banner inline-banner-warn" role="alert" style={{ marginBottom: 12 }}>
            <Icons.alert size={15} />
            <span>
              {leaderboardPollState === 'gave_up'
                ? 'Live sync stopped after repeated errors.'
                : 'Live sync paused (results may still be updating).'}
            </span>
            <span className="inline-banner-actions">
              <button className="btn btn-secondary" onClick={restartLeaderboardPoll} type="button">
                Refresh
              </button>
            </span>
          </div>
        ) : null}

        <div
          className="leaderboard-table leaderboard-scroll"
          style={{ '--lb-grid-cols': leaderboardGridColumns }}
          role="table"
          aria-label="Model leaderboard"
        >
          <div className="leaderboard-head" role="row">
            <span role="columnheader">Rank</span>
            <span role="columnheader">Model</span>
            {metricSchema.map(({ key, label }) => (
              <span role="columnheader" key={key}>{label}</span>
            ))}
            <span role="columnheader">Overfit</span>
            <span role="columnheader">Judge</span>
            {usingLive ? <span role="columnheader">Download</span> : null}
          </div>

          {displayRows.map((row) => {
            const modelName = row.model_name || row.model;
            const overfitting = row.overfitting;
            const isOverfitted = overfitting?.is_overfitted;
            const overfitGap = overfitting?.gap;
            // HPT data merged directly from leaderboard endpoint (used in winner badge + params panel below)

            return (
              <div
                className={row.winner ? 'leaderboard-row winner' : 'leaderboard-row'}
                key={modelName}
                role="row"
              >
                <span className="rank mono" role="cell">{row.rank}</span>
                <span role="cell">
                  <strong>{modelName}</strong>
                  {(row.reasons || []).length > 0 ? (
                    <small className="muted">
                      {row.reasons[0]}
                    </small>
                  ) : null}
                  {/* HPT best score badge shown inline on any tuned row (top_n can
                      tune more than just the winner). */}
                  {row.hpt_best_score != null && (
                    <small className="hpt-score-badge">
                      HPT {row.hpt_primary_metric ?? 'score'}: {typeof row.hpt_best_score === 'number' ? row.hpt_best_score.toFixed(4) : row.hpt_best_score}
                    </small>
                  )}
                </span>
                {metricSchema.map(({ key }) => (
                  <span className="mono" role="cell" key={key}>
                    {formatNumber(row.metrics?.[key] ?? row[key])}
                  </span>
                ))}
                <span
                  className="mono"
                  role="cell"
                  style={{ color: isOverfitted ? 'var(--err)' : isOverfitted === false ? 'var(--ok)' : undefined }}
                >
                  {overfitGap !== null && overfitGap !== undefined
                    ? `${isOverfitted ? 'HIGH' : 'OK'} ${formatNumber(overfitGap, 3)}`
                    : '--'}
                </span>
                <span className="mono" role="cell">{formatNumber(row.score ?? row.judge, 2)}</span>
                {usingLive && activeSessionId ? (
                  <span role="cell">
                    <a
                      className="btn-icon"
                      download
                      href={modelDownloadUrl(activeSessionId, modelName)}
                      title={`Download ${modelName}`}
                    >
                      <Icons.download size={15} />
                    </a>
                  </span>
                ) : null}
              </div>
            );
          })}

          {/* HPT best params inline panel — one per tuned model, not just the
              winner, since "Top models" can tune more than one. */}
          {displayRows.filter((r) => r.hpt_best_params && Object.keys(r.hpt_best_params).length > 0).map((row) => (
            <div
              key={`hpt-params-${row.model_name || row.model}`}
              style={{
                borderTop: '1px solid rgba(236,72,153,0.2)',
                padding: '12px 16px',
                background: 'rgba(236,72,153,0.04)',
              }}
            >
              <p className="section-kicker" style={{ margin: '0 0 6px 0', fontSize: '0.65rem', color: 'var(--hpt)' }}>
                BEST HYPERPARAMETERS ({row.model_name || row.model}) &mdash; {row.hpt_n_trials ?? '?'} Optuna trials
              </p>
              <pre className="mono" style={{ margin: 0, fontSize: '0.75rem', color: 'var(--ink-muted)', whiteSpace: 'pre-wrap', maxHeight: 120, overflowY: 'auto' }}>
                {JSON.stringify(row.hpt_best_params, null, 2)}
              </pre>
            </div>
          ))}
        </div>
      </section>
      )}

      {/* Model Decision Cards — per-model Judge governance dashboard (SPEC) */}
      {decisionCardRows.length > 0 ? (
        <section className="card panel-section">
          <div className="agent-reasoning-header">
            {judgeAgent ? <AgentAvatar agent={judgeAgent} size={30} state="done" /> : null}
            <div>
              <p className="section-kicker">Model Governance</p>
              <h2>Judge Decision Cards</h2>
            </div>
            <StatusPill status="done" label={`${decisionCardRows.length} models judged`} />
          </div>

          {/* Why one model beat another (top-two comparison) */}
          {comparisonExplanation ? (
            <div style={{ marginBottom: 14 }}>
              <p className="section-kicker" style={{ marginBottom: 6 }}>Comparison</p>
              <pre className="reasoning-block" style={{ whiteSpace: 'pre-wrap' }}>{comparisonExplanation}</pre>
            </div>
          ) : null}

          <div
            className="decision-card-grid"
            style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 14 }}
          >
            {decisionCardRows.map((row) => (
              <ModelDecisionCard key={row.model_name || row.model} row={row} />
            ))}
          </div>
        </section>
      ) : null}

      {/* Bottom row: SHAP + Judge Reasoning + Token Usage */}
      <div className="leaderboard-bottom-grid">
        {/* SHAP panel */}
        <section className="card panel-section">
          <div className="agent-reasoning-header">
            {featureAgent ? <AgentAvatar agent={featureAgent} size={30} state={shapData ? 'done' : 'idle'} /> : null}
            <div>
              <p className="section-kicker">Explainability</p>
              <h2>SHAP Feature Importance {shapModelName ? `(${shapModelName})` : ''}</h2>
            </div>
          </div>
          <HBars data={shapData || SHAP} />
        </section>

        {/* Judge Reasoning panel — VERY IMPORTANT: full text, never truncated */}
        <section className="card panel-section reasoning-panel">
          <div className="agent-reasoning-header">
            {judgeAgent ? <AgentAvatar agent={judgeAgent} size={30} state={verdictData ? 'done' : 'idle'} /> : null}
            <div>
              <p className="section-kicker">Judge</p>
              <h2>Agent Reasoning</h2>
            </div>
            {verdictData ? <StatusPill status="done" label="Judge converged" /> : null}
          </div>

          {/* Visible failure state: llm_ranking_status is always set by the
              backend (applied/failed/skipped), so a 'failed' ranking is shown
              explicitly instead of silently rendering an empty panel. */}
          {decisionTrace?.llm_ranking_status === 'failed' ? (
            <div className="inline-banner inline-banner-warn" role="alert" style={{ marginBottom: 10 }}>
              <Icons.alert size={15} />
              <span>LLM ranking unavailable -- showing rule-based order.</span>
              {decisionTrace.llm_ranking_error ? (
                <details style={{ marginLeft: 8, cursor: 'pointer' }}>
                  <summary style={{ outline: 'none', fontSize: '0.78rem' }}>Error details</summary>
                  <pre className="reasoning-block" style={{ marginTop: 6 }}>{decisionTrace.llm_ranking_error}</pre>
                </details>
              ) : null}
            </div>
          ) : null}

          {/* Full LLM commentary */}
          {decisionTrace?.llm_commentary ? (
            <div>
              <p className="section-kicker" style={{ marginBottom: 6 }}>LLM Commentary</p>
              <pre className="reasoning-block">{decisionTrace.llm_commentary}</pre>
            </div>
          ) : (
            <p className="muted">
              {verdictData ? 'Rule-based decision -- no LLM commentary.' : 'Awaiting judge verdict.'}
            </p>
          )}

          {decisionTrace?.transcript ? (
            <div style={{ marginTop: 14 }}>
              <p className="section-kicker" style={{ marginBottom: 6 }}>LLM Audit Trail</p>
              <details style={{ cursor: 'pointer', background: 'var(--panel-2)', border: '1px solid var(--line)', borderRadius: 'var(--radius)', padding: '10px 14px' }}>
                <summary style={{ outline: 'none', fontWeight: 500, fontSize: '13px', color: 'var(--ink)' }}>
                  View Raw LLM Prompt & Response Transcript
                </summary>
                <pre className="reasoning-block" style={{ marginTop: 10, background: 'rgba(0, 0, 0, 0.25)', border: 'none', maxHeight: '350px', overflowY: 'auto' }}>
                  {decisionTrace.transcript}
                </pre>
              </details>
            </div>
          ) : null}

          {/* Rule outcomes table -- gate_outcomes (model -> rejection reason) and
              scores (model -> composite score) are dicts, not scalars, so they
              get a per-model sub-list instead of a raw JSON.stringify dump. */}
          {decisionTrace?.rule_outcomes && Object.keys(decisionTrace.rule_outcomes).length > 0 ? (
            <div style={{ marginTop: 14 }}>
              <p className="section-kicker" style={{ marginBottom: 6 }}>Rule Outcomes</p>
              <div className="rule-outcomes-table">
                {Object.entries(decisionTrace.rule_outcomes).map(([ruleName, outcome]) => {
                  if (ruleName === 'gate_outcomes' && outcome && typeof outcome === 'object') {
                    const entries = Object.entries(outcome);
                    if (entries.length === 0) return null;
                    return (
                      <div className="rule-row rule-row-stacked" key={ruleName}>
                        <span className="rule-name mono">Gate Outcomes</span>
                        <div className="rule-sublist">
                          {entries.map(([modelName, reason]) => (
                            <div className="rule-subrow" key={modelName}>
                              <span className="rule-submodel mono">{modelName}</span>
                              <span className="rule-subreason">{reason}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  }
                  if (ruleName === 'scores' && outcome && typeof outcome === 'object') {
                    const entries = Object.entries(outcome).sort(
                      ([, scoreA], [, scoreB]) => (Number(scoreB) || 0) - (Number(scoreA) || 0)
                    );
                    if (entries.length === 0) return null;
                    return (
                      <div className="rule-row rule-row-stacked" key={ruleName}>
                        <span className="rule-name mono">Scores</span>
                        <div className="rule-sublist">
                          {entries.map(([modelName, score]) => (
                            <div className="rule-subrow" key={modelName}>
                              <span className="rule-submodel mono">{modelName}</span>
                              <span className="rule-subscore mono">{Number(score).toFixed(4)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  }
                  return (
                    <div className="rule-row" key={ruleName}>
                      <span className="rule-name mono">{ruleName}</span>
                      <span className="rule-value">{JSON.stringify(outcome)}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : null}

          {/* Winner model reasons -- only when the per-model decision cards are
              not rendered, to avoid showing the same reasoning twice. */}
          {winnerReasons.length > 0 && decisionCardRows.length === 0 ? (
            <div style={{ marginTop: 14 }}>
              <p className="section-kicker" style={{ marginBottom: 6 }}>
                Reasons for {winnerLabel}
              </p>
              <div className="reason-list">
                {winnerReasons.map((reason, reasonIndex) => (
                  <div className="reason-row" key={reasonIndex}>
                    <Icons.checkCircle size={15} />
                    <span>{reason}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}
        </section>
      </div>
    </div>
  );
}

export default LeaderboardScreen;
