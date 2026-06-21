import { useEffect, useMemo, useReducer, useRef, useState } from 'react';

import AgentAvatar from '../components/AgentAvatar.jsx';
import HBars from '../components/HBars.jsx';
import ModelTrainingCard from '../components/training/ModelTrainingCard.jsx';
import TrainingLogs from '../components/training/TrainingLogs.jsx';
import TrainingProgress from '../components/training/TrainingProgress.jsx';
import TrainingSummary from '../components/training/TrainingSummary.jsx';
import { fetchHpt, fetchModelConfig, fetchPlots, fetchShap, fetchVerdict, plotUrl, fetchFeatureEngineering } from '../api/client.js';
import { streamTrainingEvents } from '../api/events.js';
import { cancelTraining, fetchTrainingStatus, startTraining, resetTraining } from '../api/training.js';
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

function getEvalStepStatus(stepKey, stageStatuses) {
  const stage = stageStatuses?.[stepKey];
  if (!stage) return 'queued';
  if (stage.status === 'complete') return 'done';
  if (stage.status === 'running') return 'active';
  if (stage.status === 'failed') return 'error';
  return 'queued';
}

function EvaluationProgressSteps({ stageStatuses }) {
  const steps = [
    { key: 'shap', label: 'SHAP Explainability Analysis' },
    { key: 'overfitting', label: 'Overfitting Analysis' },
    { key: 'judge', label: 'Judge Agent Verdict' },
  ];

  return (
    <div className="metadata-progress" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div className="metadata-steps">
        {steps.map((step) => {
          const status = getEvalStepStatus(step.key, stageStatuses);
          const stageInfo = stageStatuses?.[step.key];
          const progress = stageInfo?.progress ?? 0;
          const msg = stageInfo?.message ?? '';

          return (
            <div key={step.key} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div className={`metadata-step ${status}`} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <span className="metadata-step-icon">
                  {status === 'done' && <Icons.checkCircle size={16} style={{ color: 'var(--ok)' }} />}
                  {status === 'active' && <span className="spinner small" style={{ borderColor: 'var(--accent)' }} />}
                  {status === 'error' && <Icons.alert size={16} style={{ color: 'var(--error)' }} />}
                  {status === 'queued' && <Icons.dot size={12} />}
                </span>
                <span className="metadata-step-label" style={{ fontSize: '0.85rem' }}>
                  {step.label}
                  {status === 'active' && progress > 0 ? (
                    <small className="mono"> {progress}%</small>
                  ) : null}
                </span>
              </div>
              {status === 'active' && msg && (
                <p className="muted" style={{ margin: '2px 0 0 28px', fontSize: '0.78rem', fontStyle: 'italic', lineHeight: 1.25 }}>
                  {msg}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TrainingAnalyticsSection({ sessionId, verdictData, onRestartTraining, isRestarting, restartError, stageStatuses }) {
  const [shapData, setShapData] = useState(null);
  const [modelConfigData, setModelConfigData] = useState(null);
  const [plots, setPlots] = useState([]);

  useEffect(() => {
    if (!sessionId) return undefined;
    let cancelled = false;

    Promise.all([
      fetchShap(sessionId),
      fetchModelConfig(sessionId),
      fetchPlots(sessionId),
    ]).then(([shap, config, plotsResp]) => {
      if (cancelled) return;
      const shapFeatures = (shap?.features || []).map((item) => ({
        feature: item.feature,
        value: item.importance,
      }));
      setShapData(shapFeatures.length ? shapFeatures : null);
      setModelConfigData(config?.status === 'complete' ? config : null);
      setPlots(plotsResp?.plots || []);
    }).catch(() => {});

    return () => { cancelled = true; };
  }, [sessionId]);

  if (!verdictData || verdictData.status !== 'complete') {
    return null;
  }

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

  // Plots filtered to training/overfitting stages for display here.
  const analyticsPlots = plots.filter((plot) =>
    plot.stage && /training|overfitting/.test(plot.stage)
  );

  // Model families from model_config for the chip list.
  const modelFamilies = useMemo(() => {
    if (!modelConfigData) return [];
    const models = modelConfigData.models || modelConfigData.candidates || [];
    return models.map((modelEntry) => modelEntry.family || modelEntry.model_name || modelEntry.name).filter(Boolean);
  }, [modelConfigData]);

  const hasLeftContent = Boolean(shapData || analyticsPlots.length > 0);

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

      <div className="training-analytics-grid" style={hasLeftContent ? {} : { gridTemplateColumns: '1fr' }}>
        {/* Left column: SHAP + HPT plots */}
        {hasLeftContent && (
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
                  {judgeAgent ? <AgentAvatar agent={judgeAgent} size={28} state="done" /> : null}
                  <div>
                    <p className="section-kicker">Training Plots</p>
                    <h2>Training & Overfitting</h2>
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
        )}

        {/* Right column: Judge Reasoning -- VERY IMPORTANT: full text, not truncated */}
        <div className="screen-stack">
          <section className="card panel-section reasoning-panel">
            <div className="agent-reasoning-header">
              {judgeAgent ? <AgentAvatar agent={judgeAgent} size={30} state={verdictData?.status === 'complete' ? 'done' : 'idle'} /> : null}
              <div>
                <p className="section-kicker">Judge</p>
                <h2>Agent Reasoning</h2>
              </div>
            </div>

            {verdictData?.status === 'complete' ? (
              <>
                {llmCommentary ? (
                  <div style={{ marginTop: 12 }}>
                    <p className="section-kicker" style={{ marginBottom: 6 }}>LLM Commentary</p>
                    <pre className="reasoning-block">{llmCommentary}</pre>
                  </div>
                ) : (
                  <p className="muted" style={{ marginTop: 10 }}>
                    Rule-based decision -- no LLM commentary recorded.
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

                <div style={{ marginTop: 20, paddingTop: 15, borderTop: '1px solid rgba(255, 255, 255, 0.08)' }}>
                  <button
                    className="btn btn-secondary full-width"
                    disabled={isRestarting}
                    onClick={onRestartTraining}
                    type="button"
                  >
                    <Icons.play size={16} />
                    {isRestarting ? 'Restarting Training...' : 'Re-run Training with Judge Feedback'}
                  </button>
                  {restartError ? (
                    <p className="error-text" style={{ marginTop: 8, color: 'var(--color-warn, #ef4444)', fontSize: '0.85rem' }}>
                      {restartError}
                    </p>
                  ) : null}
                </div>
              </>
            ) : (
              <div style={{ padding: '15px 0', display: 'flex', flexDirection: 'column', gap: 16 }}>
                <EvaluationProgressSteps stageStatuses={stageStatuses} />
                <div className="progress-bar indeterminate">
                  <span />
                </div>
                <p className="muted" style={{ textAlign: 'center', fontSize: '0.82rem', margin: 0 }}>
                  Evaluating trained candidates against complexity and accuracy constraints...
                </p>
              </div>
            )}
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

function formatTime(timestamp) {
  if (!timestamp) return '--:--:--';
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return '--:--:--';
  return date.toLocaleTimeString([], { hour12: false });
}

function EvaluationLogs({ logs }) {
  const logRef = useRef(null);
  const evaluationLogs = logs.filter(
    (entry) =>
      entry.stage === 'evaluation' ||
      entry.stage === 'judge' ||
      entry.stage === 'shap' ||
      entry.stage === 'overfitting'
  );

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [evaluationLogs.length]);

  let status = 'pending';
  if (evaluationLogs.length > 0) {
    const lastLog = evaluationLogs[evaluationLogs.length - 1];
    if (lastLog.level === 'error' || lastLog.status === 'failed') {
      status = 'failed';
    } else if (
      lastLog.status === 'completed' ||
      lastLog.message.toLowerCase().includes('completed') ||
      lastLog.message.toLowerCase().includes('complete')
    ) {
      status = 'complete';
    } else {
      status = 'running';
    }
  }

  return (
    <section className="card terminal-panel evaluation-log-panel" style={{ marginTop: 15 }}>
      <div className="terminal-head" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>SSE evaluation event stream</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {status === 'running' && (
            <span className="pill pill-running" style={{ background: 'rgba(59, 130, 246, 0.15)', color: '#3b82f6', border: '1px solid rgba(59, 130, 246, 0.3)', fontWeight: 600 }}>
              Running
            </span>
          )}
          {status === 'complete' && (
            <span className="pill pill-done" style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#10b981', border: '1px solid rgba(16, 185, 129, 0.3)', fontWeight: 600 }}>
              Completed
            </span>
          )}
          {status === 'failed' && (
            <span className="pill pill-failed" style={{ background: 'rgba(239, 68, 68, 0.15)', color: '#ef4444', border: '1px solid rgba(239, 68, 68, 0.3)', fontWeight: 600 }}>
              Failed
            </span>
          )}
          {status === 'pending' && <span className="pill pill-queued">Pending</span>}
        </div>
      </div>
      <div className="terminal-body evaluation-log-body" ref={logRef} style={{ height: 180, overflowY: 'auto' }}>
        {evaluationLogs.length ? (
          evaluationLogs.map((entry, index) => (
            <div
              className="terminal-line evaluation-log-line"
              key={`${entry.sequence}-${index}`}
              style={{
                color: entry.level === 'error' ? 'var(--error)' : entry.level === 'warn' ? 'var(--warning)' : 'inherit',
              }}
            >
              <span>{formatTime(entry.ts)}</span>
              <strong
                className={`level-${entry.level}`}
                style={{
                  color: entry.level === 'error' ? 'var(--error)' : entry.level === 'warn' ? 'var(--warning)' : 'var(--accent)',
                  marginRight: 8,
                }}
              >
                {entry.status}
              </strong>
              <em style={{ fontStyle: 'normal' }}>{entry.message}</em>
            </div>
          ))
        ) : (
          <span className="terminal-empty">awaiting evaluation pipeline events</span>
        )}
      </div>
    </section>
  );
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
  const [verdictData, setVerdictData] = useState(null);
  const [isRestarting, setIsRestarting] = useState(false);
  const [restartError, setRestartError] = useState(null);
  const [stageStatuses, setStageStatuses] = useState({
    d2v: { status: 'pending', progress: 0, message: '' },
    model_selection: { status: 'pending', progress: 0, message: '' },
    training: { status: 'pending', progress: 0, message: '' },
    shap: { status: 'pending', progress: 0, message: '' },
    overfitting: { status: 'pending', progress: 0, message: '' },
    evaluation: { status: 'pending', progress: 0, message: '' },
    judge: { status: 'pending', progress: 0, message: '' },
    hpt: { status: 'pending', progress: 0, message: '' },
  });

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
    setStageStatuses({
      d2v: { status: 'pending', progress: 0, message: '' },
      model_selection: { status: 'pending', progress: 0, message: '' },
      training: { status: 'pending', progress: 0, message: '' },
      shap: { status: 'pending', progress: 0, message: '' },
      overfitting: { status: 'pending', progress: 0, message: '' },
      evaluation: { status: 'pending', progress: 0, message: '' },
      judge: { status: 'pending', progress: 0, message: '' },
      hpt: { status: 'pending', progress: 0, message: '' },
    });

    sourceRef.current = streamTrainingEvents(normalized, {
      onOpen: () => {
        setConnectionStatus('open');
        setConnectionMessage('Live connection established.');
      },
      onEvent: (event) => {
        dispatch({ type: 'event', payload: event });
        if (event && event.stage) {
          setStageStatuses((prev) => {
            const next = { ...prev };
            let statusVal = 'pending';
            if (event.status === 'running') statusVal = 'running';
            else if (event.status === 'completed' || event.status === 'all_completed') statusVal = 'complete';
            else if (event.status === 'failed') statusVal = 'failed';
            
            next[event.stage] = {
              status: statusVal,
              progress: event.pct ?? 0,
              message: event.msg ?? '',
            };
            return next;
          });
        }
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




  useEffect(() => {
    if (!connectedSessionId) {
      setVerdictData(null);
      return undefined;
    }
    let stopped = false;
    let timerId = null;

    async function pollVerdict() {
      try {
        const data = await fetchVerdict(connectedSessionId);
        if (stopped) return;
        setVerdictData(data);
        if (data?.status !== 'complete') {
          timerId = window.setTimeout(pollVerdict, 2000);
        }
      } catch (err) {
        if (!stopped) {
          timerId = window.setTimeout(pollVerdict, 5000);
        }
      }
    }

    pollVerdict();

    return () => {
      stopped = true;
      if (timerId) window.clearTimeout(timerId);
    };
  }, [connectedSessionId]);

  useEffect(() => {
    if (verdictData?.status === 'complete') {
      setStageStatuses((prev) => ({
        ...prev,
        shap: prev.shap.status === 'complete' ? prev.shap : { status: 'complete', progress: 100, message: 'SHAP analysis completed.' },
        overfitting: prev.overfitting.status === 'complete' ? prev.overfitting : { status: 'complete', progress: 100, message: 'Overfitting analysis completed.' },
        judge: prev.judge.status === 'complete' ? prev.judge : { status: 'complete', progress: 100, message: 'Verdict complete.' },
        evaluation: prev.evaluation.status === 'complete' ? prev.evaluation : { status: 'complete', progress: 100, message: 'Evaluation completed.' },
      }));
    }
  }, [verdictData]);

  // Terminal-close: once the pipeline truly ends (judge verdict converged, or the
  // run failed/cancelled) the backend closes the SSE session. Native EventSource
  // would otherwise auto-reconnect forever against the closed session and
  // re-replay history. Close it explicitly so the stream stops cleanly.
  useEffect(() => {
    const verdictDone = verdictData?.status === 'complete';
    const runEnded = ['failed', 'cancelled'].includes(backendStatus?.status);
    if ((verdictDone || runEnded) && sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
      setConnectionStatus('closed');
      setConnectionMessage(
        verdictDone
          ? 'Pipeline complete. Live stream closed.'
          : 'Run ended. Live stream closed.',
      );
    }
  }, [verdictData, backendStatus]);

  async function handleRestartTraining() {
    if (!connectedSessionId || isRestarting) {
      return;
    }
    setIsRestarting(true);
    setRestartError(null);
    try {
      const feData = await fetchFeatureEngineering(connectedSessionId);
      const summary = feData?.summary || {};

      await resetTraining(connectedSessionId);

      setVerdictData(null);
      setBackendStatus(null);
      setSelectedModelId(null);

      await startTraining({
        sessionId: connectedSessionId,
        targetColumn: summary.target_column || null,
        problemType: summary.task || null,
        executionMode: 'ray',
        allowFallbackArtifacts: false,
      });

      connect(connectedSessionId);
    } catch (err) {
      setRestartError(err.message || 'Failed to restart training run.');
    } finally {
      setIsRestarting(false);
    }
  }

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

  const d2vStatus = stageStatuses.d2v.status;
  const d2vProgress = stageStatuses.d2v.progress;

  const modelSelectionStatus = stageStatuses.model_selection.status;
  const modelSelectionProgress = stageStatuses.model_selection.progress;

  const isTrainingFinished = counts.total > 0 && counts.completed + counts.failed === counts.total;
  const isTrainingStarted = counts.running > 0 || counts.completed > 0;
  const trainingStatus = isTrainingFinished ? 'complete' : isTrainingStarted ? 'running' : stageStatuses.training.status;
  const trainingProgress = progress;

  const shapStatus = stageStatuses.shap.status;
  const shapProgress = stageStatuses.shap.progress;

  const overfittingStatus = stageStatuses.overfitting.status;
  const overfittingProgress = stageStatuses.overfitting.progress;

  const judgeStageStatus = stageStatuses.judge.status;
  // Derive a unified judgeStatus: once verdict is final always 'complete',
  // otherwise track the SSE judge stage events in real time.
  const judgeStatus = verdictData?.status === 'complete' ? 'complete' : judgeStageStatus;
  const judgeProgress = stageStatuses.judge.progress;
  const judgeMessage = stageStatuses.judge.message;

  const isModelSelectionComplete = modelSelectionStatus === 'complete' || models.length > 0;

  const stagesList = [
    { id: 'd2v', label: 'Dataset2Vec Matcher', status: d2vStatus, progress: d2vProgress, icon: <Icons.layers size={18} />, desc: 'Query database for recommended models' },
    { id: 'model_selection', label: 'Model Selection Agent', status: modelSelectionStatus, progress: modelSelectionProgress, icon: <Icons.spark size={18} />, desc: 'Identify and rank candidate model types' },
    { id: 'training', label: 'Model Parallel Training', status: trainingStatus, progress: trainingProgress, icon: <Icons.cpu size={18} />, desc: 'Train short-listed models in parallel on Ray' },
    { id: 'shap', label: 'SHAP Explainability', status: shapStatus, progress: shapProgress, icon: <Icons.chart size={18} />, desc: 'Generate SHAP feature importance values' },
    { id: 'overfitting', label: 'Overfitting Analysis', status: overfittingStatus, progress: overfittingProgress, icon: <Icons.alert size={18} />, desc: 'Detect train/val score generalization gaps' },
    { id: 'evaluation', label: 'Eval Orchestration', status: stageStatuses.evaluation.status, progress: stageStatuses.evaluation.progress, icon: <Icons.cpu size={18} />, desc: 'Coordinate SHAP, overfitting, judge pipeline' },
    { id: 'judge', label: 'Judge Multi-turn Loop', status: judgeStatus, progress: judgeProgress, icon: <Icons.trophy size={18} />, desc: 'Evaluate constraints and converge on winner' },
    { id: 'hpt', label: 'HPT Optimization', status: stageStatuses.hpt.status, progress: stageStatuses.hpt.progress, icon: <Icons.spark size={18} />, desc: 'Optuna hyperparameter optimization for top-1 model' },
  ];

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

      {/* Pipeline execution stages dashboard */}
      {connectedSessionId ? (
        <section className="card panel-section" style={{ marginTop: 20 }}>
          <div className="section-head">
            <div>
              <p className="section-kicker">Execution flow</p>
              <h2>Pipeline execution stages</h2>
            </div>
          </div>
          <div className="pipeline-stages-grid">
            {stagesList.map((stage) => {
              const isRunning = stage.status === 'running';
              const isComplete = stage.status === 'complete';
              const isFailed = stage.status === 'failed';
              
              return (
                <div 
                  className={`pipeline-stage-card ${stage.status}`} 
                  key={stage.id}
                  title={stage.desc}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div className="stage-card-icon">
                      {stage.icon}
                    </div>
                    <div>
                      {isRunning && <span className="pulse-indicator" />}
                      {isComplete && <Icons.checkCircle size={16} style={{ color: 'var(--ok)' }} />}
                      {isFailed && <Icons.alert size={16} style={{ color: 'var(--err)' }} />}
                      {stage.status === 'pending' && <Icons.dot size={10} style={{ color: 'rgba(255,255,255,0.2)' }} />}
                    </div>
                  </div>
                  <div style={{ marginTop: 4 }}>
                    <strong style={{ fontSize: '0.95rem', display: 'block', color: 'var(--ink)' }}>{stage.label}</strong>
                    <span className="muted" style={{ fontSize: '0.75rem', display: 'block', marginTop: 2, height: 32, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {stageStatuses[stage.id]?.message || stage.desc}
                    </span>
                  </div>
                  <div 
                    className="stage-micro-progress" 
                    style={{ width: `${stage.progress}%` }} 
                  />
                </div>
              );
            })}
          </div>

          {/* Judge live progress panel - shown when judge is actively evaluating */}
          {(judgeStatus === 'running' || judgeStatus === 'complete') && (
            <div style={{
              marginTop: 16,
              padding: '14px 18px',
              background: 'rgba(251, 191, 36, 0.05)',
              border: '1px solid rgba(251, 191, 36, 0.2)',
              borderRadius: 8,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                {judgeAgent ? <AgentAvatar agent={judgeAgent} size={24} state={judgeStatus === 'complete' ? 'done' : 'running'} /> : null}
                <div>
                  <p className="section-kicker" style={{ margin: 0, fontSize: '0.65rem', color: 'rgba(251, 191, 36, 0.9)' }}>JUDGE AGENT</p>
                  <strong style={{ fontSize: '0.9rem' }}>
                    {judgeStatus === 'complete' ? 'Evaluation complete' : 'Evaluating candidates...'}
                  </strong>
                </div>
                {judgeStatus === 'running' && <div className="spinner small" style={{ marginLeft: 'auto' }} />}
                {judgeStatus === 'complete' && <Icons.checkCircle size={18} style={{ marginLeft: 'auto', color: 'var(--ok)' }} />}
              </div>
              {judgeMessage && (
                <p className="mono muted" style={{ margin: 0, fontSize: '0.78rem', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                  {judgeMessage}
                </p>
              )}
              {judgeProgress > 0 && judgeStatus === 'running' && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', marginBottom: 4 }}>
                    <span className="muted">Progress</span>
                    <strong className="mono">{judgeProgress}%</strong>
                  </div>
                  <div style={{ height: 4, background: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{ width: `${judgeProgress}%`, height: '100%', background: 'linear-gradient(90deg, #f59e0b 0%, #fbbf24 100%)', borderRadius: 2, transition: 'width 0.5s ease' }} />
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      ) : null}

      {!connectedSessionId ? (
        <section className="card empty-card training-empty" style={{ marginTop: 20 }}>
          <Icons.cpu size={34} />
          <h2>No session connected</h2>
          <p className="muted">
            Complete New Run and open Page 2, or connect with a known session ID.
          </p>
          <button className="btn btn-secondary" onClick={() => go('upload')} type="button">
            <Icons.upload size={16} />
            Go to New Run
          </button>
        </section>
      ) : !isModelSelectionComplete ? (
        <section className="card empty-card training-empty" style={{ marginTop: 20 }}>
          <Icons.spark size={34} style={{ color: 'var(--accent)' }} />
          <h2>Model Selection running</h2>
          <p className="muted">
            The Model Selection agent is currently matching data schemas and selecting optimal architectures...
          </p>
          <div style={{ width: '100%', maxWidth: 400, marginTop: 15 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, fontSize: '0.85rem' }}>
              <span className="muted">Model selection progress</span>
              <strong className="mono">{modelSelectionProgress}%</strong>
            </div>
            <div className="bar" style={{ height: 6, background: 'rgba(255, 255, 255, 0.05)', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{ width: `${modelSelectionProgress}%`, height: '100%', background: 'linear-gradient(90deg, #3b82f6 0%, #ec4899 100%)', borderRadius: 3 }} />
            </div>
          </div>
          <p className="mono muted" style={{ fontSize: '0.8rem', marginTop: 10, textAlign: 'center', maxWidth: 500 }}>
            {stageStatuses.model_selection.message || 'Selecting candidate models...'}
          </p>
        </section>
      ) : models.length ? (
        <div className="training-layout" style={{ marginTop: 20 }}>
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
            <EvaluationLogs logs={state.logs} />
            {/* Show evaluation pipeline steps as soon as any eval stage is active */}
            {(stageStatuses.shap.status !== 'pending' ||
              stageStatuses.overfitting.status !== 'pending' ||
              stageStatuses.judge.status !== 'pending') && (
              <section className="card panel-section" style={{ padding: 15 }}>
                <div className="section-head" style={{ marginBottom: 12 }}>
                  <div>
                    <p className="section-kicker">Evaluation Status</p>
                    <h2>Pipeline Progress</h2>
                  </div>
                </div>
                <EvaluationProgressSteps stageStatuses={stageStatuses} />
              </section>
            )}
            <TrainingSummary
              canContinue={state.complete && verdictData?.status === 'complete'}
              onContinue={() => go('leaderboard')}
              summary={state.summary}
              judgePending={state.summary != null && verdictData?.status !== 'complete'}
            />
          </aside>

        </div>
      ) : (
        <section className="card empty-card training-empty" style={{ marginTop: 20 }}>
          <Icons.cpu size={34} />
          <h2>Waiting for training jobs</h2>
          <p className="muted">
            Queued model events will appear here as soon as the orchestrator starts.
          </p>
        </section>
      )}




      {/* Analytics section: SHAP + Judge Reasoning + plots -- shown after training completes */}
      {state.complete && connectedSessionId ? (
        <TrainingAnalyticsSection
          sessionId={connectedSessionId}
          verdictData={verdictData}
          onRestartTraining={handleRestartTraining}
          isRestarting={isRestarting}
          restartError={restartError}
          stageStatuses={stageStatuses}
        />
      ) : null}
    </div>
  );
}

export default TrainingPage;
