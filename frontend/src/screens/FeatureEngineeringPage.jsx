import { useEffect, useState } from 'react';

import AgentAvatar from '../components/AgentAvatar.jsx';
import StatusPill from '../components/StatusPill.jsx';
import {
  fetchD2VPrior,
  fetchFeatureEngineering,
  fetchFeatureEngineeringJobStatus,
  fetchFeatureLeaderboard,
} from '../api/client.js';
import { fetchTrainingStatus, resetTraining, startTraining } from '../api/training.js';
import { AGENTS } from '../data.js';
import { Icons } from '../icons.jsx';

const featureAgent = AGENTS.find((agent) => agent.id === 'feature');
const judgeAgent = AGENTS.find((agent) => agent.id === 'judge');

const POLL_INTERVAL_MS = 2000;
// If no FE job is registered for the session and no artifacts exist, stop
// polling after this many idle attempts and show an "absent" message rather
// than spinning forever.
const MAX_POLL_ATTEMPTS = 15;
// dataset_prior.json is written by a D2V query step that runs *after*
// feature_run.json already reports "done" (see pipeline_prep.py: _run_d2v_query
// fires after _write_feature_run_status), so a single fetch right when the FE
// job flips to done can still race the file and return status="pending"
// forever. Keep retrying on its own short loop until it resolves.
const D2V_MAX_POLL_ATTEMPTS = 30;

function agentStateFromId(agentId, agentsArray) {
  const found = (agentsArray || []).find((agent) => agent.id === agentId);
  return found?.state || 'idle';
}

function StepStatusIcon({ status }) {
  if (status === 'ok') return <Icons.checkCircle size={16} style={{ color: 'var(--ok)' }} />;
  if (status === 'error') return <Icons.alert size={16} style={{ color: 'var(--err)' }} />;
  if (status === 'running') return <span className="spinner small" />;
  return <Icons.dot size={12} />;
}

function FeatureStepList({ steps, agents }) {
  if (!steps || steps.length === 0) {
    return <p className="muted">No step data available yet.</p>;
  }

  return (
    <div className="fe-step-list">
      {steps.map((step) => {
        const isLlmStep = step.agent_type === 'llm';
        const elapsedLabel = step.elapsed_sec !== null && step.elapsed_sec !== undefined
          ? `${Number(step.elapsed_sec).toFixed(1)}s`
          : '';

        return (
          <div className={`fe-step ${step.status}`} key={step.name}>
            <span className="fe-step-icon">
              <StepStatusIcon status={step.status} />
            </span>
            <span className="fe-step-label">
              {step.label}
              {isLlmStep && featureAgent ? (
                <AgentAvatar agent={featureAgent} size={20} state={step.status === 'ok' ? 'done' : 'idle'} />
              ) : null}
            </span>
            <span className="fe-step-meta">
              {step.llm_source ? (
                <span className="pill pill-queued" style={{ fontSize: 10 }}>{step.llm_source}</span>
              ) : null}
              {elapsedLabel ? <span className="mono muted" style={{ fontSize: 11 }}>{elapsedLabel}</span> : null}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function AgentReasoningPanel({ reasoning, summaryData }) {
  const llmReasoning = reasoning?.llm_reasoning;
  const selectionMethod = reasoning?.selection_method || summaryData?.selection_method;
  const warnings = summaryData?.warnings || [];

  return (
    <section className="card panel-section reasoning-panel">
      <div className="agent-reasoning-header">
        {featureAgent ? (
          <AgentAvatar agent={featureAgent} size={34} state={llmReasoning ? 'done' : 'idle'} />
        ) : null}
        <div>
          <p className="section-kicker">Feature Selection Agent</p>
          <h2>Agent Reasoning</h2>
        </div>
        {selectionMethod ? (
          <span className="pill pill-done" style={{ fontSize: 11 }}>{selectionMethod}</span>
        ) : null}
      </div>

      {llmReasoning ? (
        <div style={{ marginTop: 12 }}>
          <p className="section-kicker" style={{ marginBottom: 6 }}>LLM Rationale</p>
          {/* Full rationale text -- never truncated, scrollable */}
          <pre className="reasoning-block">{llmReasoning}</pre>
        </div>
      ) : (
        <p className="muted" style={{ marginTop: 10 }}>
          {selectionMethod
            ? `Selection method: ${selectionMethod}. No LLM rationale recorded.`
            : 'Agent reasoning will appear here after the feature engineering step completes.'}
        </p>
      )}

      {warnings.length > 0 ? (
        <div className="callout error compact" style={{ marginTop: 12 }}>
          <strong>Warnings</strong>
          {warnings.map((warning, warningIndex) => (
            <small key={warningIndex}>{warning}</small>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function SummaryPanel({ summary, agents }) {
  if (!summary) return null;
  const featureState = agentStateFromId('feature', agents);
  const judgeState = agentStateFromId('judge', agents);

  return (
    <section className="card panel-section">
      <div className="section-head">
        <div>
          <p className="section-kicker">Pipeline</p>
          <h2>Feature Engineering Summary</h2>
        </div>
      </div>

      <div className="fe-summary-grid">
        {summary.task ? (
          <div className="fe-summary-item">
            <span className="muted">Task</span>
            <strong className="pill pill-done">{summary.task}</strong>
          </div>
        ) : null}
        {summary.target_column ? (
          <div className="fe-summary-item">
            <span className="muted">Target</span>
            <strong className="mono">{summary.target_column}</strong>
          </div>
        ) : null}
        <div className="fe-summary-item">
          <span className="muted">Selected</span>
          <strong>{(summary.selected_columns || []).length} columns</strong>
        </div>
        <div className="fe-summary-item">
          <span className="muted">Dropped</span>
          <strong>{(summary.dropped_columns || []).length} columns</strong>
        </div>
        <div className="fe-summary-item">
          <span className="muted">Created</span>
          <strong>{(summary.created_columns || []).length} engineered</strong>
        </div>
      </div>

      {/* Agent status row */}
      <div style={{ marginTop: 16 }}>
        <p className="section-kicker" style={{ marginBottom: 8 }}>Agents</p>
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          {featureAgent ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <AgentAvatar agent={featureAgent} size={32} state={featureState} />
              <div>
                <div style={{ fontWeight: 600, fontSize: 13 }}>{featureAgent.name}</div>
                <div className="muted" style={{ fontSize: 11 }}>{featureAgent.type} | {featureAgent.role}</div>
              </div>
            </div>
          ) : null}
          {judgeAgent ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <AgentAvatar agent={judgeAgent} size={32} state={judgeState} />
              <div>
                <div style={{ fontWeight: 600, fontSize: 13 }}>{judgeAgent.name}</div>
                <div className="muted" style={{ fontSize: 11 }}>{judgeAgent.type} | {judgeAgent.role}</div>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function D2VPanel({ d2vData }) {
  if (!d2vData || d2vData.status === 'pending') return null;

  if (d2vData.cold_start) {
    return (
      <section className="card panel-section">
        <div className="section-head">
          <div>
            <p className="section-kicker">Dataset2Vec</p>
            <h2>Similar Datasets</h2>
          </div>
          <Icons.database size={18} />
        </div>
        <div className="callout compact">
          <strong>Cold start</strong>
          <span>No similar past datasets found in the meta-knowledge base yet.</span>
        </div>
      </section>
    );
  }

  const neighbors = d2vData.neighbors || [];
  const rankedModels = d2vData.ranked_models || [];
  const caveats = d2vData.caveats || [];

  return (
    <section className="card panel-section">
      <div className="section-head">
        <div>
          <p className="section-kicker">Dataset2Vec</p>
          <h2>Similar Datasets &amp; Model Recommendations</h2>
        </div>
        <Icons.database size={18} />
      </div>

      {caveats.length > 0 ? (
        <div className="callout compact" style={{ marginBottom: 12 }}>
          {caveats.map((caveat, caveatIndex) => (
            <small key={caveatIndex}>{caveat}</small>
          ))}
        </div>
      ) : null}

      {neighbors.length > 0 ? (
        <div>
          <p className="section-kicker" style={{ marginBottom: 6 }}>Nearest Neighbors</p>
          <div className="rule-outcomes-table">
            {neighbors.map((neighbor) => (
              <div className="rule-row" key={neighbor.dataset_id}>
                <span className="rule-name mono">{neighbor.dataset_id}</span>
                <span className="rule-value">
                  <div className="bar" style={{ width: 80, display: 'inline-flex' }}>
                    <i style={{ width: `${(neighbor.similarity || 0) * 100}%` }} />
                  </div>
                  <span className="muted" style={{ fontSize: 11, marginLeft: 6 }}>
                    {Number(neighbor.similarity || 0).toFixed(3)} sim
                  </span>
                  {neighbor.best_model ? (
                    <span className="pill pill-done" style={{ marginLeft: 6, fontSize: 10 }}>
                      {neighbor.best_model}
                    </span>
                  ) : null}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {rankedModels.length > 0 ? (
        <div style={{ marginTop: 14 }}>
          <p className="section-kicker" style={{ marginBottom: 6 }}>Recommended Models</p>
          <div className="rule-outcomes-table">
            {rankedModels.map((rec, recIndex) => (
              <div className="rule-row" key={rec.model_name || recIndex}>
                <span className="rule-name mono">#{recIndex + 1} {rec.model_name}</span>
                <span className="rule-value muted" style={{ fontSize: 11 }}>
                  score: {Number(rec.score || 0).toFixed(3)}
                  {rec.expected_metric !== undefined ? ` | expected: ${Number(rec.expected_metric).toFixed(3)}` : ''}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </section>
  );
}

const FE_ALGO_META = [
  { key: 'mi',        label: 'Mutual Info',  color: '#6366f1' },
  { key: 'ig',        label: 'Info Gain',    color: '#22c55e' },
  { key: 'mrmr',      label: 'mRMR',         color: '#f59e0b' },
  { key: 'laplacian', label: 'Laplacian',    color: '#06b6d4' },
  { key: 'variance',  label: 'Variance',     color: '#a855f7' },
];

function FeatureLeaderboard({ leaderboardData }) {
  if (!leaderboardData || !leaderboardData.features || leaderboardData.features.length === 0) {
    return null;
  }

  const features = leaderboardData.features;

  return (
    <section className="card panel-section">
      <div className="section-head">
        <div>
          <p className="section-kicker">Explainability</p>
          <h2>Feature Leaderboard</h2>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <span className="pill pill-done">{leaderboardData.total_selected} selected</span>
          <span className="pill pill-err">{leaderboardData.total_dropped} dropped</span>
        </div>
      </div>
      <p className="muted" style={{ marginTop: 4, marginBottom: 10, fontSize: '0.82rem' }}>
        Combined score is the mean of normalized MI, IG, mRMR, Laplacian, and Variance. Green = selected; red = dropped.
      </p>

      {/* Algorithm legend */}
      <div style={{ display: 'flex', gap: 14, marginBottom: 14, flexWrap: 'wrap' }}>
        {FE_ALGO_META.map((algo) => (
          <span key={algo.key} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.75rem', color: 'var(--ink-muted)' }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: algo.color, display: 'inline-block', flexShrink: 0 }} />
            {algo.label}
          </span>
        ))}
      </div>

      <div className="hbars">
        {features.map((item, index) => {
          const isSelected = item.status === 'selected';
          const barColor = isSelected ? 'var(--ok)' : 'var(--error)';
          const barWidth = (item.combined_score || 0) * 100;
          return (
            <div className="hbar-row" key={item.feature} style={{ marginBottom: 6 }}>
              <div className="hbar-labels">
                <span className="mono" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{
                    width: 7, height: 7, borderRadius: '50%',
                    background: barColor, flexShrink: 0, display: 'inline-block',
                  }} />
                  {item.feature}
                </span>
                <span className="mono">{(item.combined_score || 0).toFixed(3)}</span>
              </div>
              {/* Combined score bar */}
              <div className="bar">
                <i style={{
                  '--bar-color': barColor,
                  '--bar-delay': `${index * 0.04}s`,
                  width: `${barWidth}%`,
                }} />
              </div>
              {/* Per-algorithm mini bars */}
              {item.algo_scores && Object.keys(item.algo_scores).length > 0 ? (
                <div style={{ display: 'flex', gap: 4, marginTop: 3 }}>
                  {FE_ALGO_META.map((algo) => {
                    const score = item.algo_scores[algo.key];
                    if (score === undefined) return null;
                    return (
                      <div key={algo.key} title={`${algo.label}: ${score.toFixed(3)}`}
                        style={{ flex: 1, height: 4, background: 'var(--surface-2)', borderRadius: 2, overflow: 'hidden' }}>
                        <div style={{ width: `${score * 100}%`, height: '100%', background: algo.color, borderRadius: 2 }} />
                      </div>
                    );
                  })}
                </div>
              ) : null}
            </div>
          );
        })}
      </div>

      <div style={{ display: 'flex', gap: 16, marginTop: 14, fontSize: '0.78rem', color: 'var(--ink-muted)' }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--ok)', display: 'inline-block' }} />
          Selected by pipeline
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--error)', display: 'inline-block' }} />
          Dropped
        </span>
      </div>
    </section>
  );
}

function FeatureEngineeringPage({ activeSessionId, startRun }) {
  const [feData, setFeData] = useState(null);
  const [d2vData, setD2vData] = useState(null);
  const [leaderboardData, setLeaderboardData] = useState(null);
  // jobStatus is the lifecycle source of truth (running/done/error/idle), read
  // from the FE job registry; feData is the detailed 11-step view from files.
  const [jobStatus, setJobStatus] = useState('idle');
  const [jobError, setJobError] = useState(null);
  const [polling, setPolling] = useState(false);
  const [startingTraining, setStartingTraining] = useState(false);
  const [trainingError, setTrainingError] = useState(null);
  // Whether a training run already exists for this session. Drives the CTA
  // label ("Continue to Training" vs "Re-run Training") and whether we must
  // wipe prior artifacts before starting, so revisiting a completed session no
  // longer hits the backend 409 "Training already exists" error.
  const [trainingExists, setTrainingExists] = useState(false);

  useEffect(() => {
    if (!activeSessionId) {
      setFeData(null);
      setJobStatus('idle');
      setLeaderboardData(null);
      return undefined;
    }

    let timeoutId = null;
    let cancelled = false;
    let idleAttempts = 0;

    async function pollOnce() {
      try {
        // Sequential reads (one run/job at a time): lifecycle, then the
        // detailed 11-step view from files.
        const status = await fetchFeatureEngineeringJobStatus(activeSessionId).catch(() => ({ status: 'idle' }));
        const detail = await fetchFeatureEngineering(activeSessionId).catch(() => null);
        if (cancelled) return;

        if (detail) setFeData(detail);
        const lifecycle = status?.status || 'idle';
        setJobStatus(lifecycle);
        setJobError(status?.message || null);

        const detailDone = detail?.status === 'done';
        const detailFailed = detail?.status === 'partial_failure';
        const isDone = lifecycle === 'done' || detailDone;
        const isError = lifecycle === 'error' || detailFailed;

        if (isError) {
          setPolling(false);
          return;
        }
        if (isDone) {
          setPolling(false);
          fetchFeatureLeaderboard(activeSessionId)
            .then((data) => { if (!cancelled) setLeaderboardData(data); })
            .catch(() => {});
          return;
        }
        // Still running, OR no FE job ever started for this session (idle).
        if (lifecycle === 'idle' && (!detail || detail.status === 'pending')) {
          idleAttempts += 1;
          if (idleAttempts >= MAX_POLL_ATTEMPTS) {
            setPolling(false);
            return;
          }
        } else {
          idleAttempts = 0;
        }
        setPolling(true);
        timeoutId = setTimeout(pollOnce, POLL_INTERVAL_MS);
      } catch (pollError) {
        if (!cancelled) setPolling(false);
      }
    }

    setPolling(true);
    pollOnce();

    return () => {
      cancelled = true;
      setPolling(false);
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [activeSessionId]);

  // Dataset2Vec prior poll: independent of the main FE poll above because
  // dataset_prior.json can still be mid-write when feature_run.json already
  // reports "done" (backend writes the D2V prior after the status file).
  // Keeps retrying until the endpoint reports a non-pending status.
  useEffect(() => {
    if (!activeSessionId) {
      setD2vData(null);
      return undefined;
    }

    let timeoutId = null;
    let cancelled = false;
    let attempts = 0;

    async function pollD2vOnce() {
      const prior = await fetchD2VPrior(activeSessionId).catch(() => null);
      if (cancelled) return;

      if (prior) setD2vData(prior);

      if (prior && prior.status !== 'pending') {
        return;
      }
      attempts += 1;
      if (attempts >= D2V_MAX_POLL_ATTEMPTS) {
        return;
      }
      timeoutId = setTimeout(pollD2vOnce, POLL_INTERVAL_MS);
    }

    pollD2vOnce();

    return () => {
      cancelled = true;
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [activeSessionId]);

  // Detect whether a training run already exists for this session. A 404 means
  // no run yet (first-time "Continue to Training"); any successful status means
  // a run exists and the CTA becomes "Re-run Training".
  useEffect(() => {
    const sessionId = String(activeSessionId || '').trim();
    if (!sessionId) {
      setTrainingExists(false);
      return undefined;
    }
    let cancelled = false;
    fetchTrainingStatus(sessionId)
      .then(() => { if (!cancelled) setTrainingExists(true); })
      .catch(() => { if (!cancelled) setTrainingExists(false); });
    return () => { cancelled = true; };
  }, [activeSessionId]);

  async function handleContinueToTraining() {
    const sessionId = String(activeSessionId || '').trim();
    if (!sessionId) return;
    setTrainingError(null);
    setStartingTraining(true);
    try {
      const summary = feData?.summary || {};
      const startPayload = {
        sessionId,
        targetColumn: summary.target_column || null,
        problemType: summary.task === 'clustering' ? 'unsupervised' : (summary.task || null),
        executionMode: 'ray',
        // Hard-fail: FE must have produced model_config.json; never fall back.
        allowFallbackArtifacts: false,
      };
      // Re-run: wipe prior training artifacts first so the pipeline starts clean
      // (the backend rejects /start with 409 while a run already exists).
      if (trainingExists) {
        await resetTraining(sessionId);
      }
      try {
        await startTraining(startPayload);
      } catch (startError) {
        // Safety net: if a run still exists (e.g. trainingExists was stale),
        // reset once and retry instead of surfacing the 409 to the user.
        if (startError.status === 409) {
          await resetTraining(sessionId);
          await startTraining(startPayload);
        } else {
          throw startError;
        }
      }
      startRun?.(sessionId);
    } catch (continueError) {
      setTrainingError(continueError.message);
      setStartingTraining(false);
    }
  }

  if (!activeSessionId) {
    return (
      <div className="screen-stack">
        <div className="callout compact">
          <strong>No active session</strong>
          <span>Start a new run to see feature engineering details.</span>
        </div>
      </div>
    );
  }

  const overallStatus = feData?.status || 'pending';
  const steps = feData?.steps || [];
  const agents = feData?.agents || [];
  const summary = feData?.summary || null;
  const reasoning = feData?.reasoning || null;

  const isError = jobStatus === 'error' || overallStatus === 'partial_failure';
  const isDone = jobStatus === 'done' || overallStatus === 'done';
  const isRunning = !isError && !isDone && (polling || jobStatus === 'running');
  // No FE job ever ran for this session and polling has stopped.
  const isAbsent = !isError && !isDone && !isRunning && jobStatus === 'idle'
    && overallStatus === 'pending';

  const headerStatus = isError ? 'error' : isDone ? 'done' : 'running';
  const headerLabel = isError ? 'Failed' : isDone ? 'Complete' : isRunning ? 'Running' : 'Waiting';

  return (
    <div className="screen-stack">
      {/* Status header */}
      <section className="card hero-panel" style={{ paddingBottom: 16 }}>
        <div>
          <StatusPill status={headerStatus} spin={isRunning} label={headerLabel} />
          <h2>Feature Engineering Pipeline</h2>
          <p className="muted">Session: {activeSessionId}</p>
        </div>
        {isDone ? (
          <button
            className="btn btn-primary"
            disabled={startingTraining}
            onClick={handleContinueToTraining}
            type="button"
            title={trainingExists ? 'Deletes existing training artifacts and restarts the pipeline' : undefined}
          >
            {startingTraining ? <span className="spinner" /> : trainingExists ? <Icons.activity size={16} /> : <Icons.play size={16} />}
            {startingTraining
              ? (trainingExists ? 'Restarting training...' : 'Starting training...')
              : (trainingExists ? 'Re-run Training' : 'Continue to Training')}
          </button>
        ) : null}
      </section>

      {/* Hard-fail banner: pipeline halted, no Continue button */}
      {isError ? (
        <div className="callout error compact">
          <strong>Feature engineering failed -- pipeline halted.</strong>
          <span>{jobError || 'Check the LLM settings and dataset, then re-run from New Run.'}</span>
        </div>
      ) : null}

      {trainingError ? (
        <div className="callout error compact">
          <strong>Could not start training.</strong>
          <span>{trainingError}</span>
        </div>
      ) : null}

      {isAbsent ? (
        <div className="callout compact">
          <strong>No feature engineering run for this session</strong>
          <span>Start a run from New Run and click "Continue to Feature Engineering" after metadata.</span>
        </div>
      ) : null}

      <div className="fe-main-grid">
        {/* Left: step list */}
        <div className="screen-stack">
          <section className="card panel-section">
            <div className="section-head">
              <div>
                <p className="section-kicker">Pipeline Steps</p>
                <h2>11-Step Feature Pipeline</h2>
              </div>
              {isRunning ? <StatusPill status="running" spin label="Live" /> : null}
            </div>
            <FeatureStepList agents={agents} steps={steps} />
          </section>

          <SummaryPanel agents={agents} summary={summary} />
        </div>

        {/* Right: Agent Reasoning panel (VERY IMPORTANT -- prominent, full text) */}
        <div className="screen-stack">
          <AgentReasoningPanel reasoning={reasoning} summaryData={summary} />
          <D2VPanel d2vData={d2vData} />
        </div>
      </div>

      {/* Feature leaderboard: ranked by MI score, colored by selection status */}
      {isDone ? <FeatureLeaderboard leaderboardData={leaderboardData} /> : null}

    </div>
  );
}

export default FeatureEngineeringPage;
