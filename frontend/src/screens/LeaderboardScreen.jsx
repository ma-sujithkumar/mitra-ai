import { useEffect, useMemo, useState } from 'react';

import AgentAvatar from '../components/AgentAvatar.jsx';
import HBars from '../components/HBars.jsx';
import StatusPill from '../components/StatusPill.jsx';
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
import { streamTrainingEvents } from '../api/events.js';
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

// Detect whether the session uses classification or regression metrics by
// probing the first model's metrics dict. Falls back to a minimal schema
// for prototype data that stores metrics directly on the row object.
function detectMetricSchema(models) {
  const firstRow = models[0] || {};
  const firstMetrics = firstRow.metrics || {};

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

function LeaderboardScreen({ activeSessionId, startRun }) {
  const [leaderboardData, setLeaderboardData] = useState(null);
  const [shapData, setShapData] = useState(null);
  const [verdictData, setVerdictData] = useState(null);
  const [tokenData, setTokenData] = useState(null);
  const [loadState, setLoadState] = useState('idle');
  const [hptData, setHptData] = useState(null);
  const [hptStatus, setHptStatus] = useState('idle'); // 'idle' | 'running' | 'complete' | 'failed'

  useEffect(() => {
    if (!activeSessionId) {
      setLoadState('idle');
      return undefined;
    }
    let cancelled = false;
    let timeoutId = null;
    let attempts = 0;
    setLoadState('loading');

    // Bounded poll so the page syncs as the pipeline produces artifacts
    // (judge/SHAP/tokens land after training). Stop when the judge has
    // converged (status === 'complete') or after the attempt cap.
    async function pollOnce() {
      attempts += 1;
      try {
        // Sequential reads (parallelization = 1).
        const leaderboard = await fetchLeaderboard(activeSessionId);
        const shap = await fetchShap(activeSessionId).catch(() => null);
        const verdict = await fetchVerdict(activeSessionId).catch(() => null);
        const tokens = await fetchTokens(activeSessionId).catch(() => null);
        const hpt = await fetchHpt(activeSessionId).catch(() => null);
        
        if (cancelled) return;

        setLeaderboardData(leaderboard);
        const features = (shap?.features || []).map((item) => ({
          feature: item.feature,
          value: item.importance,
        }));
        setShapData(features.length ? features : null);
        setVerdictData(verdict?.status && verdict.status !== 'pending' ? verdict : null);
        setTokenData(tokens?.status === 'complete' ? tokens : null);
        
        if (hpt?.status === 'complete' && hpt?.hpt_results?.length) {
          setHptData(hpt.hpt_results);
          setHptStatus('complete');
        } else if (hpt?.status === 'running') {
          setHptStatus('running');
        } else {
          setHptStatus('idle');
          setHptData(null);
        }

        setLoadState('done');

        const terminal = leaderboard?.status === 'complete';
        if (!terminal && attempts < LEADERBOARD_MAX_POLLS) {
          timeoutId = setTimeout(pollOnce, LEADERBOARD_POLL_MS);
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

  useEffect(() => {
    if (!activeSessionId || hptStatus !== 'running') return undefined;
    let timerId = null;
    let stopped = false;
    async function checkHpt() {
      try {
        const data = await fetchHpt(activeSessionId);
        if (stopped) return;
        if (data?.status === 'complete' && data?.hpt_results) {
          setHptData(data.hpt_results);
          setHptStatus('complete');
        } else if (data?.status === 'failed') {
          setHptStatus('failed');
        } else {
          timerId = setTimeout(checkHpt, 2000);
        }
      } catch (err) {
        if (!stopped) {
          timerId = setTimeout(checkHpt, 5000);
        }
      }
    }
    checkHpt();
    return () => {
      stopped = true;
      if (timerId) clearTimeout(timerId);
    };
  }, [activeSessionId, hptStatus]);

  const handleRunHpt = async () => {
    try {
      setHptStatus('running');
      await runHpt(activeSessionId);
    } catch (err) {
      console.error(err);
      setHptStatus('failed');
    }
  };

  const models = leaderboardData?.models || [];
  const usingLive = models.length > 0;
  const displayRows = usingLive ? models : LEADERBOARD;
  const selectedModel = leaderboardData?.selected_model || null;
  const decisionTrace = leaderboardData?.decision_trace || verdictData?.decision_trace || null;
  const metricSchema = useMemo(() => detectMetricSchema(displayRows), [displayRows]);
  const winnerRow = displayRows.find((row) => row.winner) || displayRows[0];
  const winnerLabel = usingLive ? (selectedModel || winnerRow?.model_name || winnerRow?.model) : winnerRow?.model;

  // Build a list of reasons for the selected model from the judge verdict.
  const winnerReasons = useMemo(() => {
    if (!verdictData) return [];
    const ranked = verdictData.ranked_models || [];
    const winnerRecord = ranked.find((m) => m.model_name === selectedModel) || ranked[0];
    return winnerRecord?.reasons || [];
  }, [verdictData, selectedModel]);

  // Per-agent token totals for the token usage panel.
  const agentTokenRows = useMemo(() => {
    if (!tokenData?.agents) return [];
    return Object.entries(tokenData.agents).map(([agentName, usage]) => ({
      name: agentName,
      input: usage?.input_tokens ?? 0,
      output: usage?.output_tokens ?? 0,
    }));
  }, [tokenData]);

  return (
    <div className="screen-stack">
      {/* Hero banner */}
      <section className="card hero-panel leaderboard-hero">
        <div className="winner-mark">
          <Icons.trophy size={28} />
        </div>
        <div>
          <StatusPill status="done" label={usingLive ? 'Judge converged' : 'Prototype data'} />
          <h2>{winnerLabel ? `${winnerLabel} is the recommended model` : 'Awaiting judge'}</h2>
          <p className="muted">
            {usingLive
              ? `Live results for session ${activeSessionId}.`
              : 'Prototype leaderboard using the Claude handoff model results.'}
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
          <button className="btn btn-primary" onClick={() => startRun()} type="button">
            <Icons.play size={16} />
            Back to training
          </button>
        </div>
      </section>

      {/* Main leaderboard table */}
      <section className="card panel-section">
        <div className="section-head">
          <div>
            <p className="section-kicker">Models</p>
            <h2>Leaderboard</h2>
          </div>
          {loadState === 'loading' ? <StatusPill status="running" spin /> : null}
        </div>

        <div className="leaderboard-table leaderboard-scroll">
          <div className="leaderboard-head">
            <span>Rank</span>
            <span>Model</span>
            {metricSchema.map(({ key, label }) => (
              <span key={key}>{label}</span>
            ))}
            <span>Overfit</span>
            <span>Judge</span>
            {usingLive ? <span>Download</span> : null}
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
              >
                <span className="rank mono">{row.rank}</span>
                <span>
                  <strong>{modelName}</strong>
                  {(row.reasons || []).length > 0 ? (
                    <small className="muted">
                      {row.reasons[0]}
                    </small>
                  ) : null}
                  {/* HPT best score badge shown inline on winner row */}
                  {row.winner && row.hpt_best_score != null && (
                    <small style={{ display: 'inline-flex', alignItems: 'center', gap: 4, marginLeft: 8, background: 'rgba(236,72,153,0.12)', color: '#ec4899', borderRadius: 4, padding: '1px 6px', fontSize: '0.7rem', fontWeight: 700 }}>
                      HPT {row.hpt_primary_metric ?? 'score'}: {typeof row.hpt_best_score === 'number' ? row.hpt_best_score.toFixed(4) : row.hpt_best_score}
                    </small>
                  )}
                </span>
                {metricSchema.map(({ key }) => (
                  <span className="mono" key={key}>
                    {formatNumber(row.metrics?.[key] ?? row[key])}
                  </span>
                ))}
                <span
                  className="mono"
                  style={{ color: isOverfitted ? 'var(--err)' : isOverfitted === false ? 'var(--ok)' : undefined }}
                >
                  {overfitGap !== null && overfitGap !== undefined
                    ? `${isOverfitted ? 'HIGH' : 'OK'} ${formatNumber(overfitGap, 3)}`
                    : '--'}
                </span>
                <span className="mono">{formatNumber(row.score ?? row.judge, 1)}</span>
                {usingLive && activeSessionId ? (
                  <span>
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

          {/* HPT best params inline panel — only for winner row when tuned */}
          {displayRows.filter((r) => r.winner && r.hpt_best_params && Object.keys(r.hpt_best_params).length > 0).map((row) => (
            <div
              key={`hpt-params-${row.model_name || row.model}`}
              style={{
                borderTop: '1px solid rgba(236,72,153,0.2)',
                padding: '12px 16px',
                background: 'rgba(236,72,153,0.04)',
              }}
            >
              <p className="section-kicker" style={{ margin: '0 0 6px 0', fontSize: '0.65rem', color: '#ec4899' }}>
                BEST HYPERPARAMETERS ({row.model_name || row.model}) &mdash; {row.hpt_n_trials ?? '?'} Optuna trials
              </p>
              <pre className="mono" style={{ margin: 0, fontSize: '0.75rem', color: '#ccc', whiteSpace: 'pre-wrap', maxHeight: 120, overflowY: 'auto' }}>
                {JSON.stringify(row.hpt_best_params, null, 2)}
              </pre>
            </div>
          ))}
        </div>
      </section>

      {/* Hyperparameter Tuning Section */}
      {usingLive && activeSessionId && (
        <section className="card panel-section" style={{ borderLeft: '4px solid #ec4899', display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
            <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
              <div className="stage-card-icon" style={{ background: 'rgba(236, 72, 153, 0.15)', color: '#ec4899', width: 36, height: 36, borderRadius: 6, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Icons.cpu size={20} />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '1.15rem' }}>Hyperparameter Tuning (Optuna HPT)</h3>
                <p className="muted" style={{ margin: '4px 0 0 0', fontSize: '0.85rem' }}>
                  {hptStatus === 'idle' && "Run Optuna HPT on the top-1 Judge-selected model (5 trials). Results appear in leaderboard."}
                  {hptStatus === 'running' && "Tuning the top-1 model (5 Optuna trials)... Results will appear in the leaderboard winner row."}
                  {hptStatus === 'complete' && "HPT completed. Best hyperparameters and score are now in the leaderboard winner row."}
                  {hptStatus === 'failed' && "Hyperparameter tuning execution failed."}
                </p>
              </div>
            </div>
            <div>
              {hptStatus === 'idle' && (
                <button className="btn btn-primary" onClick={handleRunHpt} style={{ background: '#ec4899', borderColor: '#ec4899' }} type="button">
                  <Icons.spark size={15} /> Tune Hyperparameters
                </button>
              )}
              {hptStatus === 'running' && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div className="spinner small" />
                  <span className="muted" style={{ fontSize: '0.9rem' }}>Tuning...</span>
                </div>
              )}
              {hptStatus === 'complete' && (
                <span className="pill pill-done" style={{ background: 'rgba(236, 72, 153, 0.15)', color: '#ec4899', border: '1px solid rgba(236, 72, 153, 0.3)', fontWeight: 600 }}>
                  Tuned
                </span>
              )}
              {hptStatus === 'failed' && (
                <button className="btn btn-secondary" onClick={handleRunHpt} type="button">
                  Retry Tuning
                </button>
              )}
            </div>
          </div>

          {hptData && hptData.length > 0 && (
            <div className="hpt-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 15 }}>
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
                  <div style={{ display: 'flex', gap: 12, fontSize: '0.8rem', color: '#aaa' }}>
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
                      <pre className="mono" style={{ margin: 0, fontSize: '0.75rem', whiteSpace: 'pre-wrap', color: '#ccc' }}>
                        {JSON.stringify(model.best_hyperparameters, null, 2)}
                      </pre>
                    </div>
                  </div>
                  {model.val_metrics ? (
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4, fontSize: '0.8rem' }}>
                      <span>Validation Score:</span>
                      <strong className="mono" style={{ color: '#ec4899' }}>
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

      {/* Bottom row: SHAP + Judge Reasoning + Token Usage */}
      <div className="leaderboard-grid">
        {/* SHAP panel */}
        <section className="card panel-section">
          <div className="agent-reasoning-header">
            {featureAgent ? <AgentAvatar agent={featureAgent} size={30} state={shapData ? 'done' : 'idle'} /> : null}
            <div>
              <p className="section-kicker">Explainability</p>
              <h2>SHAP Feature Importance</h2>
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

          {/* Rule outcomes table */}
          {decisionTrace?.rule_outcomes && Object.keys(decisionTrace.rule_outcomes).length > 0 ? (
            <div style={{ marginTop: 14 }}>
              <p className="section-kicker" style={{ marginBottom: 6 }}>Rule Outcomes</p>
              <div className="rule-outcomes-table">
                {Object.entries(decisionTrace.rule_outcomes).map(([ruleName, outcome]) => (
                  <div className="rule-row" key={ruleName}>
                    <span className="rule-name mono">{ruleName}</span>
                    <span className="rule-value">{JSON.stringify(outcome)}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}

          {/* Winner model reasons */}
          {winnerReasons.length > 0 ? (
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

        {/* Token usage panel */}
        {tokenData || loadState === 'done' ? (
          <section className="card panel-section">
            <div className="section-head">
              <div>
                <p className="section-kicker">Resources</p>
                <h2>Token Usage</h2>
              </div>
              <Icons.spark size={18} />
            </div>
            {agentTokenRows.length > 0 ? (
              <div className="rule-outcomes-table">
                {agentTokenRows.map(({ name, input, output }) => (
                  <div className="rule-row" key={name}>
                    <span className="rule-name mono">{name}</span>
                    <span className="rule-value muted">
                      {input.toLocaleString()} in / {output.toLocaleString()} out
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="muted">No token usage data available.</p>
            )}
          </section>
        ) : null}
      </div>
    </div>
  );
}

export default LeaderboardScreen;
