import { useEffect, useMemo, useReducer, useRef, useState } from 'react';

import AgentAvatar from '../components/AgentAvatar.jsx';
import HBars from '../components/HBars.jsx';
import ModelTrainingCard from '../components/training/ModelTrainingCard.jsx';
import TrainingLogs from '../components/training/TrainingLogs.jsx';
import TrainingProgress from '../components/training/TrainingProgress.jsx';
import TrainingSummary from '../components/training/TrainingSummary.jsx';
import { fetchModelConfig, fetchPlots, fetchShap, fetchVerdict, plotUrl } from '../api/client.js';
import { streamTrainingEvents } from '../api/events.js';
import { cancelTraining, fetchTrainingStatus } from '../api/training.js';
import { AGENTS } from '../data.js';
import { Icons } from '../icons.jsx';
import {
  applyTrainingEvent,
  applyTrainingStatus,
  createTrainingState,
  overallTrainingProgress,
  selectTrainingModels,
  trainingCounts,
} from '../trainingState.js';

const judgeAgent = AGENTS.find((agent) => agent.id === 'judge');
const hptAgent = AGENTS.find((agent) => agent.id === 'hpt');
const featureAgent = AGENTS.find((agent) => agent.id === 'feature');

function TrainingAnalyticsSection({ sessionId }) {
  const [shapData, setShapData] = useState(null);
  const [verdictData, setVerdictData] = useState(null);
  const [modelConfigData, setModelConfigData] = useState(null);
  const [plots, setPlots] = useState([]);

  useEffect(() => {
    if (!sessionId) return undefined;
    let cancelled = false;

    Promise.all([
      fetchShap(sessionId),
      fetchVerdict(sessionId),
      fetchModelConfig(sessionId),
      fetchPlots(sessionId),
    ]).then(([shap, verdict, config, plotsResp]) => {
      if (cancelled) return;
      const shapFeatures = (shap?.features || []).map((item) => ({
        feature: item.feature,
        value: item.importance,
      }));
      setShapData(shapFeatures.length ? shapFeatures : null);
      setVerdictData(verdict?.status !== 'pending' ? verdict : null);
      setModelConfigData(config?.status === 'complete' ? config : null);
      setPlots(plotsResp?.plots || []);
    }).catch(() => {});

    return () => { cancelled = true; };
  }, [sessionId]);

  const decisionTrace = verdictData?.decision_trace || null;
  const llmCommentary = decisionTrace?.llm_commentary || null;
  const ruleOutcomes = decisionTrace?.rule_outcomes || {};
  const selectedModel = verdictData?.selected_model || null;
  const winnerReasons = useMemo(() => {
    if (!verdictData) return [];
    const ranked = verdictData.ranked_models || [];
    const winnerRecord = ranked.find((model) => model.model_name === selectedModel) || ranked[0];
    return winnerRecord?.reasons || [];
  }, [verdictData, selectedModel]);

  // Plots filtered to training/hpt/overfitting stages for display here.
  const analyticsPlots = plots.filter((plot) =>
    plot.stage && /training|hpt|overfitting/.test(plot.stage)
  );

  // Model families from model_config for the chip list.
  const modelFamilies = useMemo(() => {
    if (!modelConfigData) return [];
    const models = modelConfigData.models || modelConfigData.candidates || [];
    return models.map((modelEntry) => modelEntry.family || modelEntry.model_name || modelEntry.name).filter(Boolean);
  }, [modelConfigData]);

  return (
    <section className="screen-stack">
      {/* Model config chip list */}
      {modelFamilies.length > 0 ? (
        <section className="card panel-section">
          <div className="section-head">
            <div>
              <p className="section-kicker">Configuration</p>
              <h2>Selected Model Families</h2>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 8 }}>
            {modelFamilies.map((family) => (
              <span className="pill pill-queued" key={family}>{family}</span>
            ))}
          </div>
        </section>
      ) : null}

      <div className="training-analytics-grid">
        {/* Left column: SHAP + HPT plots */}
        <div className="screen-stack">
          {shapData ? (
            <section className="card panel-section">
              <div className="agent-reasoning-header">
                {featureAgent ? <AgentAvatar agent={featureAgent} size={28} state="done" /> : null}
                <div>
                  <p className="section-kicker">Explainability</p>
                  <h2>SHAP Feature Importance</h2>
                </div>
              </div>
              <HBars data={shapData} />
            </section>
          ) : null}

          {analyticsPlots.length > 0 ? (
            <section className="card panel-section">
              <div className="agent-reasoning-header">
                {hptAgent ? <AgentAvatar agent={hptAgent} size={28} state="done" /> : null}
                <div>
                  <p className="section-kicker">Training Plots</p>
                  <h2>HPT / Overfitting / Training</h2>
                </div>
              </div>
              <div className="plot-gallery">
                {analyticsPlots.slice(0, 6).map((plot) => (
                  <div className="plot-card" key={plot.path}>
                    <img
                      alt={plot.name}
                      className="plot-thumb"
                      loading="lazy"
                      src={plotUrl(sessionId, plot.path)}
                    />
                    <p className="plot-name muted">{plot.name.replace(/_/g, ' ')}</p>
                  </div>
                ))}
              </div>
            </section>
          ) : null}
        </div>

        {/* Right column: Judge Reasoning -- VERY IMPORTANT: full text, not truncated */}
        <div className="screen-stack">
          <section className="card panel-section reasoning-panel">
            <div className="agent-reasoning-header">
              {judgeAgent ? <AgentAvatar agent={judgeAgent} size={30} state={verdictData ? 'done' : 'idle'} /> : null}
              <div>
                <p className="section-kicker">Judge</p>
                <h2>Agent Reasoning</h2>
              </div>
            </div>

            {llmCommentary ? (
              <div style={{ marginTop: 12 }}>
                <p className="section-kicker" style={{ marginBottom: 6 }}>LLM Commentary</p>
                <pre className="reasoning-block">{llmCommentary}</pre>
              </div>
            ) : (
              <p className="muted" style={{ marginTop: 10 }}>
                {verdictData
                  ? 'Rule-based decision -- no LLM commentary recorded.'
                  : 'Judge reasoning will appear after the evaluation phase completes.'}
              </p>
            )}

            {Object.keys(ruleOutcomes).length > 0 ? (
              <div style={{ marginTop: 14 }}>
                <p className="section-kicker" style={{ marginBottom: 6 }}>Rule Outcomes</p>
                <div className="rule-outcomes-table">
                  {Object.entries(ruleOutcomes).map(([ruleName, outcome]) => (
                    <div className="rule-row" key={ruleName}>
                      <span className="rule-name mono">{ruleName}</span>
                      <span className="rule-value">{JSON.stringify(outcome)}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            {winnerReasons.length > 0 ? (
              <div style={{ marginTop: 14 }}>
                <p className="section-kicker" style={{ marginBottom: 6 }}>
                  Reasons for {selectedModel}
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
    </section>
  );
}

const SESSION_STORAGE_KEY = 'mitra.activeTrainingSession';

function reducer(state, action) {
  if (action.type === 'reset') {
    return createTrainingState();
  }
  if (action.type === 'event') {
    return applyTrainingEvent(state, action.payload);
  }
  if (action.type === 'status') {
    return applyTrainingStatus(state, action.payload);
  }
  return state;
}

function TrainingPage({ activeSessionId, go, runState, setRunState, setActiveSessionId }) {
  const [state, dispatch] = useReducer(reducer, undefined, createTrainingState);
  const [sessionInput, setSessionInput] = useState(
    activeSessionId || window.localStorage.getItem(SESSION_STORAGE_KEY) || '',
  );
  const [connectedSessionId, setConnectedSessionId] = useState('');
  const [connectionStatus, setConnectionStatus] = useState('idle');
  const [connectionMessage, setConnectionMessage] = useState('');
  const [selectedModelId, setSelectedModelId] = useState(null);
  const [backendStatus, setBackendStatus] = useState(null);
  const [isCancelling, setIsCancelling] = useState(false);
  const sourceRef = useRef(null);

  const models = useMemo(() => selectTrainingModels(state), [state]);
  const counts = useMemo(() => trainingCounts(state), [state]);
  const progress = useMemo(() => overallTrainingProgress(state), [state]);

  function disconnect() {
    sourceRef.current?.close();
    sourceRef.current = null;
    setConnectionStatus('closed');
  }

  function connect(sessionId) {
    const normalized = String(sessionId || '').trim();
    if (!normalized) {
      setConnectionMessage('Enter a valid session ID.');
      return;
    }

    sourceRef.current?.close();
    dispatch({ type: 'reset' });
    setSelectedModelId(null);
    setConnectedSessionId(normalized);
    setSessionInput(normalized);
    setConnectionStatus('connecting');
    setBackendStatus(null);
    setConnectionMessage('Connecting to the training event stream…');
    setRunState('running');
    setActiveSessionId(normalized);
    window.localStorage.setItem(SESSION_STORAGE_KEY, normalized);

    sourceRef.current = streamTrainingEvents(normalized, {
      onOpen: () => {
        setConnectionStatus('open');
        setConnectionMessage('Live connection established.');
      },
      onEvent: (event) => {
        dispatch({ type: 'event', payload: event });
        setConnectionStatus('open');
        setConnectionMessage('Receiving live Ray training events.');
      },
      onDone: () => {
        setConnectionStatus('closed');
        setConnectionMessage('Training stream completed.');
        setRunState('done');
      },
      onError: () => {
        setConnectionStatus('reconnecting');
        setConnectionMessage('Connection interrupted. EventSource is reconnecting automatically.');
      },
    });
  }

  useEffect(() => {
    if (activeSessionId && activeSessionId !== connectedSessionId) {
      connect(activeSessionId);
    }
    return () => {
      sourceRef.current?.close();
    };
    // connect is intentionally event-driven; reconnect only when the active
    // session identity changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSessionId]);

  useEffect(() => {
    if (state.complete && runState !== 'done') {
      setRunState('done');
    }
  }, [runState, setRunState, state.complete]);

  useEffect(() => {
    if (!connectedSessionId || state.complete) {
      return undefined;
    }

    let stopped = false;
    async function pollStatus() {
      try {
        const statusPayload = await fetchTrainingStatus(connectedSessionId);
        if (!stopped) {
          setBackendStatus(statusPayload);
          dispatch({ type: 'status', payload: statusPayload });
          if (['completed', 'partial_failure', 'failed', 'cancelled'].includes(statusPayload.status)) {
            setRunState('done');
          }
        }
      } catch (statusError) {
        if (!stopped && statusError.status !== 404) {
          setConnectionMessage(statusError.message);
        }
      }
    }

    pollStatus();
    const intervalId = window.setInterval(pollStatus, 1500);
    return () => {
      stopped = true;
      window.clearInterval(intervalId);
    };
  }, [connectedSessionId, setRunState, state.complete]);

  async function handleCancel() {
    if (!connectedSessionId || isCancelling) {
      return;
    }
    setIsCancelling(true);
    try {
      const payload = await cancelTraining(connectedSessionId);
      setBackendStatus((currentStatus) => ({
        ...(currentStatus || {}),
        ...payload,
      }));
      setConnectionMessage('Training cancellation requested.');
    } catch (cancelError) {
      setConnectionMessage(cancelError.message);
    } finally {
      setIsCancelling(false);
    }
  }

  return (
    <div className="screen-stack">
      <section className="card training-session-bar">
        <div>
          <p className="section-kicker">Training run</p>
          <h2>
            {connectedSessionId ? 'Connected automatically' : 'Waiting to connect'}
          </h2>
          <p className="muted">
            {connectionMessage || 'This connects on its own using the run you just started.'}
          </p>
          {backendStatus?.status ? (
            <span className="mono muted">Backend status: {backendStatus.status}</span>
          ) : null}
        </div>
        <div className="training-session-controls">
          {['created', 'running'].includes(backendStatus?.status) ? (
            <button
              className="btn btn-secondary"
              disabled={isCancelling}
              onClick={handleCancel}
              type="button"
            >
              <Icons.pause size={16} />
              {isCancelling ? 'Cancelling...' : 'Cancel training'}
            </button>
          ) : null}
        </div>
      </section>

      {/* "Session ID" is an internal identifier, not something users should
          need to know or type. It is only exposed here, collapsed, as a
          fallback for re-attaching to a run when nothing auto-connected. */}
      <details className="advanced-disclosure" open={!activeSessionId && !connectedSessionId}>
        <summary>Advanced: connect to a specific run manually</summary>
        <div className="training-session-controls advanced-disclosure-body">
          <input
            aria-label="Training session ID"
            className="input mono"
            onChange={(event) => setSessionInput(event.target.value)}
            placeholder="session-id"
            value={sessionInput}
          />
          <button className="btn btn-primary" onClick={() => connect(sessionInput)} type="button">
            <Icons.play size={16} />
            {connectedSessionId ? 'Reconnect' : 'Connect'}
          </button>
          {sourceRef.current ? (
            <button className="btn btn-secondary" onClick={disconnect} type="button">
              <Icons.pause size={16} />
              Disconnect
            </button>
          ) : null}
        </div>
      </details>

      <TrainingProgress
        connectionStatus={connectionStatus}
        counts={counts}
        progress={progress}
      />

      {models.length ? (
        <div className="training-layout">
          <section className="card panel-section">
            <div className="section-head">
              <div>
                <p className="section-kicker">Selected Models</p>
                <h2>Training queue</h2>
              </div>
              <span className="mono muted">{models.length} candidates</span>
            </div>
            <div className="training-model-list">
              {models.map((model) => (
                <ModelTrainingCard
                  key={model.modelId}
                  model={model}
                  onSelect={setSelectedModelId}
                  selected={selectedModelId === model.modelId}
                />
              ))}
            </div>
          </section>

          <aside className="screen-stack">
            <TrainingLogs
              logs={state.logs}
              onClearFilter={() => setSelectedModelId(null)}
              selectedModelId={selectedModelId}
            />
            <TrainingSummary
              canContinue={state.complete}
              onContinue={() => go('leaderboard')}
              summary={state.summary}
            />
          </aside>
        </div>
      ) : (
        <section className="card empty-card training-empty">
          <Icons.cpu size={34} />
          <h2>{connectedSessionId ? 'Waiting for training jobs' : 'No session connected'}</h2>
          <p className="muted">
            {connectedSessionId
              ? 'Queued model events will appear here as soon as the orchestrator starts.'
              : 'Complete New Run and open Page 2, or connect with a known session ID.'}
          </p>
          {!connectedSessionId ? (
            <button className="btn btn-secondary" onClick={() => go('upload')} type="button">
              <Icons.upload size={16} />
              Go to New Run
            </button>
          ) : null}
        </section>
      )}

      {/* Analytics section: SHAP + Judge Reasoning + plots -- shown after training completes */}
      {state.complete && connectedSessionId ? (
        <TrainingAnalyticsSection sessionId={connectedSessionId} />
      ) : null}
    </div>
  );
}

export default TrainingPage;
