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
        if (cancelled) return;

        setLeaderboardData(leaderboard);
        const features = (shap?.features || []).map((item) => ({
          feature: item.feature,
          value: item.importance,
        }));
        setShapData(features.length ? features : null);
        setVerdictData(verdict?.status && verdict.status !== 'pending' ? verdict : null);
        setTokenData(tokens?.status === 'complete' ? tokens : null);
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
        </div>
      </section>

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
