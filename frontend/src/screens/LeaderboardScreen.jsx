import { useEffect, useMemo, useState } from 'react';

import AgentAvatar from '../components/AgentAvatar.jsx';
import HBars from '../components/HBars.jsx';
import StatusPill from '../components/StatusPill.jsx';
import { fetchLeaderboard, fetchShap, fetchVerdict } from '../api/client.js';
import { AGENTS, LEADERBOARD, SHAP } from '../data.js';
import { Icons } from '../icons.jsx';

const judgeAgent = AGENTS.find((agent) => agent.id === 'judge');

// Metric keys to probe for the two numeric columns, covering both
// classification (accuracy/f1) and regression (r2/rmse) result dicts.
const PRIMARY_METRIC_KEYS = ['accuracy', 'r2', 'roc_auc', 'auc'];
const SECONDARY_METRIC_KEYS = ['f1', 'f1_score', 'rmse', 'mae'];

const DEFAULT_REASONS = [
  ['Lowest overfit gap', '0.018 against the holdout split'],
  ['Accuracy floor met', 'All top candidates clear the threshold'],
  ['SHAP stable', 'Feature importance consistent across folds'],
];

function pickMetric(metrics, keys) {
  if (!metrics) {
    return null;
  }
  for (const key of keys) {
    if (metrics[key] !== undefined && metrics[key] !== null) {
      return metrics[key];
    }
  }
  return null;
}

function formatNumber(value, digits) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '--';
  }
  return Number(value).toFixed(digits);
}

// Map the live /leaderboard payload onto the row shape the table renders.
function toLeaderboardRows(apiModels) {
  return apiModels.map((model) => {
    const primary = pickMetric(model.metrics, PRIMARY_METRIC_KEYS) ?? model.validation_score;
    const secondary = pickMetric(model.metrics, SECONDARY_METRIC_KEYS);
    return {
      rank: model.rank,
      model: model.model_name,
      acc: primary,
      f1: secondary,
      judge: model.score,
      verdict: model.verdict,
      reasons: model.reasons || [],
      winner: Boolean(model.winner),
      hp: model.verdict ? `judge: ${model.verdict}` : '',
    };
  });
}

// Turn the judge verdict payload into [title, detail] reason rows. Prefers the
// winning model's own reasons, then falls back to the LLM commentary trace.
function buildReasons(verdict, selectedModel) {
  if (!verdict || verdict.status === 'pending') {
    return null;
  }
  const ranked = verdict.ranked_models || [];
  const winnerRecord = ranked.find((model) => model.model_name === selectedModel) || ranked[0];
  if (winnerRecord && (winnerRecord.reasons || []).length) {
    return winnerRecord.reasons.map((reason) => [winnerRecord.model_name, reason]);
  }
  const commentary = verdict.decision_trace?.llm_commentary;
  if (commentary) {
    return [['Judge commentary', commentary]];
  }
  return null;
}

function LeaderboardScreen({ activeSessionId, startRun }) {
  const [liveModels, setLiveModels] = useState(null);
  const [liveShap, setLiveShap] = useState(null);
  const [liveReasons, setLiveReasons] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [loadState, setLoadState] = useState('idle');

  useEffect(() => {
    if (!activeSessionId) {
      setLoadState('idle');
      return undefined;
    }
    let cancelled = false;
    setLoadState('loading');

    Promise.all([
      fetchLeaderboard(activeSessionId),
      fetchShap(activeSessionId),
      fetchVerdict(activeSessionId),
    ])
      .then(([leaderboard, shap, verdict]) => {
        if (cancelled) {
          return;
        }
        const models = leaderboard?.models || [];
        setLiveModels(models.length ? toLeaderboardRows(models) : null);
        setSelectedModel(leaderboard?.selected_model || null);
        const features = (shap?.features || []).map((item) => ({
          feature: item.feature,
          value: item.importance,
        }));
        setLiveShap(features.length ? features : null);
        setLiveReasons(buildReasons(verdict, leaderboard?.selected_model));
        setLoadState('done');
      })
      .catch(() => {
        if (!cancelled) {
          setLoadState('error');
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeSessionId]);

  // Fall back to the prototype mock until a real session has produced results.
  const usingLive = Boolean(liveModels);
  const rows = usingLive ? liveModels : LEADERBOARD;
  const shapData = liveShap || SHAP;
  const winner = rows.find((model) => model.winner) || rows[0];
  const maxAccuracy = useMemo(
    () => Math.max(...rows.map((model) => Number(model.acc) || 0), 0.0001),
    [rows],
  );
  const reasons = liveReasons || DEFAULT_REASONS;
  const winnerLabel = usingLive ? (selectedModel || winner?.model) : winner?.model;

  return (
    <div className="screen-stack">
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
        <button className="btn btn-primary" onClick={() => startRun()} type="button">
          <Icons.play size={16} />
          Back to training
        </button>
      </section>

      <div className="leaderboard-grid">
        <section className="card panel-section">
          <div className="section-head">
            <div>
              <p className="section-kicker">Models</p>
              <h2>Leaderboard</h2>
            </div>
            {loadState === 'loading' ? <StatusPill status="running" spin /> : null}
          </div>
          <div className="leaderboard-table">
            <div className="leaderboard-head">
              <span>Rank</span>
              <span>Model</span>
              <span>Score</span>
              <span>Metric</span>
              <span>Judge</span>
            </div>
            {rows.map((model) => (
              <div className={model.winner ? 'leaderboard-row winner' : 'leaderboard-row'} key={model.model}>
                <span className="rank mono">{model.rank}</span>
                <span>
                  <strong>{model.model}</strong>
                  <small>{model.hp}</small>
                </span>
                <span className="metric-bar">
                  <div className="bar">
                    <i style={{ width: `${((Number(model.acc) || 0) / maxAccuracy) * 100}%` }} />
                  </div>
                  <em className="mono">{formatNumber(model.acc, 3)}</em>
                </span>
                <span className="mono">{formatNumber(model.f1, 3)}</span>
                <span className="mono">{formatNumber(model.judge, 1)}</span>
              </div>
            ))}
          </div>
        </section>

        <aside className="screen-stack">
          <section className="card panel-section">
            <div className="section-head">
              <div>
                <p className="section-kicker">Explainability</p>
                <h2>SHAP</h2>
              </div>
              <Icons.spark size={18} />
            </div>
            <HBars data={shapData} />
          </section>

          <section className="card panel-section">
            <div className="judge-head">
              {judgeAgent ? <AgentAvatar agent={judgeAgent} state="done" /> : null}
              <div>
                <p className="section-kicker">Judge</p>
                <h2>Reasoning</h2>
              </div>
            </div>
            <div className="reason-list">
              {reasons.map(([title, detail]) => (
                <div className="reason-row" key={`${title}-${detail}`}>
                  <Icons.checkCircle size={17} />
                  <span>
                    <strong>{title}</strong>
                    <small>{detail}</small>
                  </span>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}

export default LeaderboardScreen;
