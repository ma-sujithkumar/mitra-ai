import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from 'react';

import AgentAvatar from '../components/AgentAvatar.jsx';
import HBars from '../components/HBars.jsx';
import StatusPill from '../components/StatusPill.jsx';
import Stepper from '../components/Stepper.jsx';
import ConfirmDialog from '../components/ConfirmDialog.jsx';
import ModelTrainingCard from '../components/training/ModelTrainingCard.jsx';
import TrainingLogs from '../components/training/TrainingLogs.jsx';
import TrainingProgress from '../components/training/TrainingProgress.jsx';
import TrainingSummary from '../components/training/TrainingSummary.jsx';
import { fetchHpt, fetchModelConfig, fetchPlots, fetchShap, fetchVerdict, plotUrl, fetchFeatureEngineering, fetchShapStatus, fetchOverfittingStatus, fetchJudgeStatus } from '../api/client.js';
import { streamTrainingEvents } from '../api/events.js';
import { cancelTraining, fetchTrainingStatus, startTraining, resetTraining } from '../api/training.js';
import { useBoundedPoll } from '../hooks/useBoundedPoll.js';
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

// The pipeline stages this page renders a card for. SSE events for any other
// stage (feature_engineering / metadata / validate / upload) are ignored here
// so they aren't silently written into state with no matching card.
const PIPELINE_STAGES = ['d2v', 'model_selection', 'training', 'shap', 'overfitting', 'evaluation', 'judge', 'hpt'];

function createStageStatuses() {
  return PIPELINE_STAGES.reduce((statuses, stage) => {
    statuses[stage] = { status: 'pending', progress: 0, message: '' };
    return statuses;
  }, {});
}

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

// TrainingAnalyticsSection renders SHAP + plot visualizations after training completes.
// Judge reasoning is displayed in the aside panel (not here) so it appears before
// the Continue button.
function TrainingAnalyticsSection({ sessionId }) {
  const [shapData, setShapData] = useState(null);
  const [modelConfigData, setModelConfigData] = useState(null);
  const [plots, setPlots] = useState([]);
  const [analyticsError, setAnalyticsError] = useState(null);

  useEffect(() => {
    if (!sessionId) return undefined;
    let cancelled = false;

    Promise.all([
      fetchShap(sessionId),
      fetchModelConfig(sessionId),
      fetchPlots(sessionId),
    ]).then(([shap, config, plotsResp]) => {
      if (cancelled) return;
      setAnalyticsError(null);
      const shapFeatures = (shap?.features || []).map((item) => ({
        feature: item.feature,
        value: item.importance,
      }));
      setShapData(shapFeatures.length ? shapFeatures : null);
      setModelConfigData(config?.status === 'complete' ? config : null);
      setPlots(plotsResp?.plots || []);
    }).catch((analyticsFetchError) => {
      if (!cancelled) {
        setAnalyticsError(
          analyticsFetchError?.message || 'Could not load SHAP / model-config / plots for this run.',
        );
      }
    });

    return () => { cancelled = true; };
  }, [sessionId]);

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

  const hasContent = Boolean(shapData || analyticsPlots.length > 0 || modelFamilies.length > 0);

  if (!hasContent && !analyticsError) return null;

  return (
    <section className="screen-stack">
      {analyticsError ? (
        <div className="inline-banner inline-banner-error" role="alert" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Icons.alert size={16} />
          <span>{analyticsError}</span>
        </div>
      ) : null}

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

      {/* Grid for SHAP and Plots -- Side by side or full width depending on availability */}
      {hasContent && (
        <div className="training-analytics-grid" style={{ gridTemplateColumns: shapData && analyticsPlots.length > 0 ? '1fr 1fr' : '1fr' }}>
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
                  <h2>Training & Overfitting Gaps</h2>
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

function EvaluationLogs({ logs, polledLogs = [] }) {
  const logRef = useRef(null);
  const sseEvalLogs = logs.filter(
    (entry) =>
      entry.stage === 'evaluation' ||
      entry.stage === 'judge' ||
      entry.stage === 'shap' ||
      entry.stage === 'overfitting'
  );

  // Combine SSE logs and polled logs, deduplicating by message + stage + status
  const combinedLogs = [...sseEvalLogs];
  polledLogs.forEach((polled) => {
    const isDuplicate = sseEvalLogs.some(
      (sse) => sse.stage === polled.stage && sse.message === polled.message
    );
    if (!isDuplicate) {
      combinedLogs.push(polled);
    }
  });

  // Sort by timestamp or sequence to keep them chronological
  combinedLogs.sort((a, b) => new Date(a.ts) - new Date(b.ts));

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [combinedLogs.length]);

  // Derive panel status from structured event fields only -- not by substring
  // matching the message text (which misclassified e.g. "incomplete").
  let status = 'pending';
  if (combinedLogs.length > 0) {
    const lastLog = combinedLogs[combinedLogs.length - 1];
    if (lastLog.level === 'error' || lastLog.status === 'failed') {
      status = 'failed';
    } else if (['completed', 'all_completed'].includes(lastLog.status)) {
      status = 'complete';
    } else {
      status = 'running';
    }
  }

  return (
    <section className="card terminal-panel evaluation-log-panel" style={{ marginTop: 15 }}>
      <div className="terminal-head" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>Evaluation event stream</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <StatusPill status={status} spin={status === 'running'} />
        </div>
      </div>
      <div className="terminal-body evaluation-log-body" ref={logRef} style={{ height: 180, overflowY: 'auto' }}>
        {combinedLogs.length ? (
          combinedLogs.map((entry, index) => (
            <div
              className="terminal-line evaluation-log-line"
              key={`${entry.sequence}-${index}`}
              style={{
                color: entry.level === 'error' ? 'var(--error)' : entry.level === 'warn' ? 'var(--warning)' : 'inherit',
              }}
            >
              <span>{formatTime(entry.ts)}</span>
              {entry.stage ? (
                <span
                  className="mono"
                  style={{
                    textTransform: 'uppercase',
                    fontSize: '0.68rem',
                    color: 'var(--ink-muted)',
                    margin: '0 8px',
                    letterSpacing: '0.04em',
                  }}
                >
                  {entry.stage}
                </span>
              ) : null}
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
  const [pendingConfirm, setPendingConfirm] = useState(null); // 'cancel' | 'restart' | null
  const [sessionInputError, setSessionInputError] = useState(null);
  const [stageStatuses, setStageStatuses] = useState(createStageStatuses);
  const [runToken, setRunToken] = useState(0);
  const [judgeStatusData, setJudgeStatusData] = useState(null);
  const [polledEvalLogs, setPolledEvalLogs] = useState([]);

  const sourceRef = useRef(null);

  const baseModels = useMemo(() => selectTrainingModels(state), [state]);
  // Push judge-rejected models to the end of the list so accepted models stay visible at top.
  const models = useMemo(() => {
    if (!verdictData?.ranked_models) return baseModels;
    const rejectedNames = new Set(
      verdictData.ranked_models
        .filter((ranked) => ranked.verdict === 'reject')
        .map((ranked) => ranked.model_name)
    );
    const accepted = baseModels.filter((model) => !rejectedNames.has(model.modelName));
    const rejected = baseModels.filter((model) => rejectedNames.has(model.modelName));
    return [...accepted, ...rejected];
  }, [baseModels, verdictData]);
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
    setVerdictData(null);
    setJudgeStatusData(null);
    setPolledEvalLogs([]);
    setConnectionMessage('Connecting to the training event stream…');
    setRunState('running');
    setActiveSessionId(normalized);
    window.localStorage.setItem(SESSION_STORAGE_KEY, normalized);
    setStageStatuses(createStageStatuses());
    setRunToken((prev) => prev + 1);

    sourceRef.current = streamTrainingEvents(normalized, {
      onOpen: () => {
        setConnectionStatus('open');
        setConnectionMessage('Live connection established.');
      },
      onEvent: (event) => {
        dispatch({ type: 'event', payload: event });
        // Only track stages this page renders a card for; ignore upstream
        // stages (feature_engineering/metadata/validate) that share the bus.
        if (event && PIPELINE_STAGES.includes(event.stage)) {
          setStageStatuses((prev) => {
            const next = { ...prev };
            let statusVal = 'pending';
            if (event.status === 'running') statusVal = 'running';
            else if (event.status === 'completed' || event.status === 'all_completed') statusVal = 'complete';
            else if (event.status === 'failed') statusVal = 'failed';

            const previousStage = prev[event.stage] || { progress: 0 };
            next[event.stage] = {
              status: statusVal,
              // Keep the last known progress when an event omits pct so the bar
              // doesn't flicker back to 0.
              progress: event.pct ?? previousStage.progress ?? 0,
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

  // Backend status poll (bounded): stops on terminal status/completion, and
  // gives up after repeated non-404 errors instead of looping forever. A 404
  // means the session is not registered yet -> keep polling without counting
  // it as an error. Like the other secondary polls (verdict/shap/overfitting/
  // judge), this is NOT stopped by the manual "Disconnect" button -- that only
  // closes the live SSE stream, not background status polling.
  const pollStatus = useCallback(async () => {
    try {
      const statusPayload = await fetchTrainingStatus(connectedSessionId);
      setBackendStatus(statusPayload);
      dispatch({ type: 'status', payload: statusPayload });
      if (['completed', 'partial_failure', 'failed', 'cancelled'].includes(statusPayload.status)) {
        setRunState('done');
      }
      return statusPayload;
    } catch (statusError) {
      if (statusError.status === 404) {
        return { status: 'pending' };
      }
      setConnectionMessage(statusError.message);
      throw statusError;
    }
  }, [connectedSessionId, setRunState]);

  // NOTE: deliberately NOT gated on connectionStatus !== 'closed'. On reconnect
  // to an already-finished session the SSE replay can be incomplete (only the
  // last judge-turn's events may remain in the in-memory history) and the
  // session is often already closed server-side, so the stream flips to
  // 'closed' within milliseconds of connecting -- before the top-level
  // all_completed event (which sets state.complete) was ever received. Gating
  // this disk-backed fallback on connectionStatus created a race where BOTH
  // the SSE path and this poll could fail to ever set state.complete, leaving
  // "Continue to leaderboard" permanently disabled. !state.complete alone is
  // sufficient to stop polling once we have a true answer.
  useBoundedPoll(pollStatus, {
    enabled: Boolean(connectedSessionId) && !state.complete,
    intervalMs: 1500,
    maxErrorAttempts: 8,
    stopWhen: (statusPayload) =>
      ['completed', 'partial_failure', 'failed', 'cancelled'].includes(statusPayload?.status),
    resetKey: `${connectedSessionId}-${runToken}`,
  });

  // Clear the verdict when the connected session changes (mirrors the old
  // effect's reset-on-disconnect behaviour).
  useEffect(() => {
    if (!connectedSessionId) {
      setVerdictData(null);
      setJudgeStatusData(null);
    }
  }, [connectedSessionId]);

  // Judge verdict poll (bounded): stops once the judge converges, caps total
  // attempts (~5 min) so it never polls forever if a verdict never lands, and
  // gives up after repeated errors. `verdictPollState` drives the retry UI.
  const pollVerdict = useCallback(async () => {
    const data = await fetchVerdict(connectedSessionId);
    setVerdictData(data);
    return data;
  }, [connectedSessionId]);

  const { pollState: verdictPollState, restart: restartVerdictPoll } = useBoundedPoll(pollVerdict, {
    // Keep polling even after the SSE stream closes: the backend closes the stream right after
    // emitting the top-level all_completed event, which may arrive before the verdict poll
    // has seen status='complete'. Without this the canContinue button never shows.
    // maxAttempts is 0 (unlimited) because judge turns can take longer than a fixed cap
    // (training + 3 judge LLM calls can exceed 5 min). stopWhen handles exit.
    enabled: Boolean(connectedSessionId),
    intervalMs: 2000,
    maxAttempts: 0,
    maxErrorAttempts: 6,
    stopWhen: (data) => data?.status === 'complete',
    resetKey: `${connectedSessionId}-${runToken}`,
  });

  // New live progress status polling for SHAP, Overfitting, and Judge Agent.
  const pollShapStatus = useCallback(async () => {
    try {
      const data = await fetchShapStatus(connectedSessionId);
      setStageStatuses((prev) => ({
        ...prev,
        shap: {
          status: data.status === 'completed' ? 'complete' : data.status,
          progress: data.progress ?? 0,
          message: data.message || '',
        }
      }));
      return data;
    } catch {
      return null;
    }
  }, [connectedSessionId]);

  const pollOverfittingStatus = useCallback(async () => {
    try {
      const data = await fetchOverfittingStatus(connectedSessionId);
      setStageStatuses((prev) => ({
        ...prev,
        overfitting: {
          status: data.status === 'completed' ? 'complete' : data.status,
          progress: data.progress ?? 0,
          message: data.message || '',
        }
      }));
      return data;
    } catch {
      return null;
    }
  }, [connectedSessionId]);

  const pollJudgeStatus = useCallback(async () => {
    try {
      const data = await fetchJudgeStatus(connectedSessionId);
      setJudgeStatusData(data);
      setStageStatuses((prev) => ({
        ...prev,
        judge: {
          status: (data.status === 'completed' || data.status === 'all_completed') ? 'complete' : data.status,
          progress: data.progress ?? 0,
          message: data.message || '',
        }
      }));
      return data;
    } catch {
      return null;
    }
  }, [connectedSessionId]);

  useBoundedPoll(pollShapStatus, {
    enabled: Boolean(connectedSessionId) && verdictData?.status !== 'complete' && connectionStatus !== 'closed',
    intervalMs: 2000,
    resetKey: `${connectedSessionId}-${runToken}`,
  });

  useBoundedPoll(pollOverfittingStatus, {
    enabled: Boolean(connectedSessionId) && verdictData?.status !== 'complete' && connectionStatus !== 'closed',
    intervalMs: 2000,
    resetKey: `${connectedSessionId}-${runToken}`,
  });

  useBoundedPoll(pollJudgeStatus, {
    enabled: Boolean(connectedSessionId) && verdictData?.status !== 'complete' && connectionStatus !== 'closed',
    intervalMs: 2000,
    resetKey: `${connectedSessionId}-${runToken}`,
  });

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

  // When the pipeline marks itself complete via SSE, ensure the verdict poll is
  // still running. If it already capped (maxAttempts=0 now, but kept as safety)
  // or gave up, restart it so the canContinue button can appear.
  useEffect(() => {
    if (state.complete && verdictData?.status !== 'complete') {
      restartVerdictPoll();
    }
  // restartVerdictPoll is stable (useCallback); state.complete is the signal.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.complete]);

  useEffect(() => {
    setPolledEvalLogs((prev) => {
      let updated = [...prev];
      let changed = false;

      const addIfNew = (stage, status, message) => {
        if (!message || status === 'pending') return;
        const lastOfStage = [...updated].reverse().find((entry) => entry.stage === stage);
        if (!lastOfStage || lastOfStage.message !== message || lastOfStage.status !== status) {
          updated.push({
            sequence: `polled-${stage}-${updated.length}`,
            ts: new Date().toISOString(),
            level: status === 'failed' ? 'error' : 'info',
            status: status,
            stage: stage,
            message: message,
          });
          changed = true;
        }
      };

      addIfNew('shap', stageStatuses.shap.status, stageStatuses.shap.message);
      addIfNew('overfitting', stageStatuses.overfitting.status, stageStatuses.overfitting.message);

      if (judgeStatusData?.logs && Array.isArray(judgeStatusData.logs)) {
        judgeStatusData.logs.forEach((logMsg) => {
          const exists = updated.some((entry) => entry.stage === 'judge' && entry.message === logMsg);
          if (!exists) {
            updated.push({
              sequence: `polled-judge-${updated.length}`,
              ts: new Date().toISOString(),
              level: judgeStatusData.status === 'failed' ? 'error' : 'info',
              status: judgeStatusData.status || 'running',
              stage: 'judge',
              message: logMsg,
            });
            changed = true;
          }
        });
      }

      return changed ? updated : prev;
    });
  }, [
    stageStatuses.shap.message,
    stageStatuses.shap.status,
    stageStatuses.overfitting.message,
    stageStatuses.overfitting.status,
    judgeStatusData?.logs,
    judgeStatusData?.status,
  ]);

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

      // Validate required inputs before the destructive reset so we don't wipe
      // results and then fail the re-train with a null target.
      const targetColumn = summary.target_column || null;
      const problemType = summary.task || null;
      if (!targetColumn) {
        setRestartError(
          'Cannot restart: the target column for this run is unavailable (feature-engineering summary missing). Start a fresh run from New Run instead.',
        );
        return;
      }

      await resetTraining(connectedSessionId);

      setVerdictData(null);
      setBackendStatus(null);
      setSelectedModelId(null);
      setJudgeStatusData(null);
      setPolledEvalLogs([]);
      setRunToken((prev) => prev + 1);

      await startTraining({
        sessionId: connectedSessionId,
        targetColumn,
        problemType,
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

  // Confirmation gating for destructive actions (cancel a run / wipe + re-run).
  const CONFIRM_DETAILS = {
    cancel: {
      title: 'Cancel this training run?',
      body: 'The run will stop and partial results may be lost. This cannot be undone.',
      confirmLabel: 'Cancel run',
      tone: 'danger',
      run: handleCancel,
    },
    restart: {
      title: 'Re-run training with judge feedback?',
      body: 'This resets the current run and retrains from scratch. Existing leaderboard and judge results for this run will be discarded.',
      confirmLabel: 'Reset and re-run',
      tone: 'danger',
      run: handleRestartTraining,
    },
  };
  const activeConfirm = pendingConfirm ? CONFIRM_DETAILS[pendingConfirm] : null;

  function handleConfirm() {
    const action = activeConfirm;
    setPendingConfirm(null);
    action?.run?.();
  }

  // Validate the manual session id before attempting to connect, so a bad value
  // gives immediate, actionable feedback instead of a silent no-op.
  function handleManualConnect() {
    const trimmed = String(sessionInput || '').trim();
    if (!trimmed) {
      setSessionInputError('Enter a session ID to connect.');
      return;
    }
    if (!/^[A-Za-z0-9._-]+$/.test(trimmed)) {
      setSessionInputError('Session IDs contain only letters, numbers, dots, dashes, and underscores.');
      return;
    }
    setSessionInputError(null);
    connect(trimmed);
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
  // Derive a unified judgeStatus: mark complete when verdict landed, pipeline fully done
  // (state.complete), or the SSE judge stage event said all_completed. The state.complete
  // guard prevents the spinner from persisting after SSE closes but before the verdict
  // poll returns its first 'complete' response.
  const judgeStatus =
    verdictData?.status === 'complete' || state.complete ? 'complete' : judgeStageStatus;
  const judgeProgress = stageStatuses.judge.progress;
  const judgeMessage = stageStatuses.judge.message;

  const isModelSelectionComplete = modelSelectionStatus === 'complete' || models.length > 0;

  // Cancel should be available whenever a run is still active -- not only when
  // the last polled backend status happens to be created/running (which goes
  // stale if the SSE stream drops).
  const runIsTerminal =
    state.complete ||
    ['completed', 'partial_failure', 'failed', 'cancelled'].includes(backendStatus?.status);
  const canCancel = Boolean(connectedSessionId) && !runIsTerminal;
  // Offer a manual reconnect when the live stream dropped and the run is not
  // yet finished (instead of an indefinite silent "reconnecting" state).
  const showReconnect =
    Boolean(connectedSessionId) &&
    !runIsTerminal &&
    ['reconnecting', 'closed', 'error'].includes(connectionStatus);
  // The judge verdict never landed within the polling budget.
  const verdictStalled =
    state.complete &&
    verdictData?.status !== 'complete' &&
    ['gave_up', 'capped'].includes(verdictPollState);

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

  // Workflow breadcrumb: lets the user move between run stages instead of being
  // trapped on this page with only the sidebar as an escape.
  const runSteps = [
    { key: 'upload', label: 'New Run', route: 'upload', status: 'done', enabled: true },
    { key: 'features', label: 'Features', route: 'features', status: 'done', enabled: true },
    { key: 'pipeline', label: 'Training', route: 'pipeline', status: 'active', enabled: true },
    {
      key: 'leaderboard',
      label: 'Leaderboard',
      route: 'leaderboard',
      status: 'queued',
      enabled: state.complete,
    },
  ];

  return (
    <div className="screen-stack">
      <div className="card panel-section" style={{ padding: '10px 14px' }}>
        <Stepper steps={runSteps} onNavigate={(step) => go(step.route)} />
      </div>
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
          {showReconnect ? (
            <button
              className="btn btn-secondary"
              onClick={() => connect(connectedSessionId)}
              type="button"
            >
              <Icons.play size={16} />
              Reconnect
            </button>
          ) : null}
          {canCancel ? (
            <button
              className="btn btn-secondary"
              disabled={isCancelling}
              onClick={() => setPendingConfirm('cancel')}
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
            aria-invalid={sessionInputError ? 'true' : undefined}
            className="input mono"
            onChange={(event) => {
              setSessionInput(event.target.value);
              if (sessionInputError) setSessionInputError(null);
            }}
            onKeyDown={(event) => {
              if (event.key === 'Enter') handleManualConnect();
            }}
            placeholder="e.g. 20260621_014807_matches_9747960c"
            value={sessionInput}
          />
          <button className="btn btn-primary" onClick={handleManualConnect} type="button">
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
        {sessionInputError ? (
          <p className="inline-banner inline-banner-error" role="alert" style={{ marginTop: 8 }}>
            <Icons.alert size={14} />
            <span>{sessionInputError}</span>
          </p>
        ) : null}
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
                  role="group"
                  aria-label={`${stage.label}: ${stage.status}`}
                  aria-live={isRunning ? 'polite' : undefined}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div className="stage-card-icon" aria-hidden="true">
                      {stage.icon}
                    </div>
                    <div aria-hidden="true">
                      {isRunning && <span className="pulse-indicator" />}
                      {isComplete && <Icons.checkCircle size={16} style={{ color: 'var(--ok)' }} />}
                      {isFailed && <Icons.alert size={16} style={{ color: 'var(--err)' }} />}
                      {stage.status === 'pending' && <Icons.dot size={10} style={{ color: 'var(--ink-faint)' }} />}
                    </div>
                  </div>
                  <div style={{ marginTop: 4 }}>
                    <strong style={{ fontSize: '0.95rem', display: 'block', color: 'var(--ink)' }}>{stage.label}</strong>
                    {/* Visually-hidden status text so screen readers convey state
                        that is otherwise only shown via colour/icon. */}
                    <span className="sr-only">Status: {stage.status}</span>
                    <span className="muted stage-card-message" style={{ fontSize: '0.75rem', display: 'block', marginTop: 2 }}>
                      {stageStatuses[stage.id]?.message || stage.desc}
                    </span>
                  </div>
                  <div
                    className="stage-micro-progress"
                    style={{ width: `${stage.progress}%` }}
                    role="progressbar"
                    aria-valuenow={Math.round(stage.progress)}
                    aria-valuemin={0}
                    aria-valuemax={100}
                    aria-label={`${stage.label} progress`}
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
            Start a run from New Run, or connect to an existing run with its session ID.
          </p>
          <button className="btn btn-secondary" onClick={() => go('upload')} type="button">
            <Icons.upload size={16} />
            Go to New Run
          </button>
        </section>
      ) : !isModelSelectionComplete ? (
        runIsTerminal ? (
          /* The run ended before any models were selected -> failure, not progress. */
          <section className="card empty-card training-empty" style={{ marginTop: 20 }}>
            <Icons.alert size={34} style={{ color: 'var(--err)' }} />
            <h2>Run ended before models were selected</h2>
            <p className="muted">
              {backendStatus?.status === 'cancelled'
                ? 'This run was cancelled before model selection completed.'
                : 'The run failed before any candidate models were produced.'}
            </p>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center', flexWrap: 'wrap', marginTop: 12 }}>
              <button className="btn btn-secondary" onClick={() => go('upload')} type="button">
                <Icons.upload size={16} /> Back to New Run
              </button>
            </div>
          </section>
        ) : (
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
            <div
              className="bar"
              style={{ height: 6, background: 'var(--line-soft)', borderRadius: 3, overflow: 'hidden' }}
              role="progressbar"
              aria-valuenow={Math.round(modelSelectionProgress)}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label="Model selection progress"
            >
              <div style={{ width: `${modelSelectionProgress}%`, height: '100%', background: 'linear-gradient(90deg, var(--info) 0%, #ec4899 100%)', borderRadius: 3 }} />
            </div>
          </div>
          <p className="mono muted" style={{ fontSize: '0.8rem', marginTop: 10, textAlign: 'center', maxWidth: 500 }}>
            {stageStatuses.model_selection.message || 'Selecting candidate models...'}
          </p>
        </section>
        )
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
            <EvaluationLogs logs={state.logs} polledLogs={polledEvalLogs} />
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

            {/* Full Judge Agent Reasoning and Verdict in the aside panel.
                This is intentionally placed BEFORE TrainingSummary so the user
                sees the judge verdict before reaching the Continue button. */}
            {(stageStatuses.judge.status !== 'pending' || judgeStatusData || verdictData) && (
              <section className="card panel-section reasoning-panel" style={{ padding: 15 }}>
                <div className="agent-reasoning-header" style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
                  {judgeAgent ? <AgentAvatar agent={judgeAgent} size={28} state={verdictData?.status === 'complete' ? 'done' : 'running'} /> : null}
                  <div style={{ flex: 1 }}>
                    <p className="section-kicker" style={{ margin: 0 }}>Judge Agent</p>
                    <h3 style={{ margin: 0, fontSize: '1.1rem' }}>Reasoning & Verdict</h3>
                  </div>
                  {/* Multi-turn indicator badge */}
                  {judgeStatusData?.max_turns > 1 ? (
                    <span className="pill pill-run" style={{ fontSize: '0.72rem' }}>
                      Turn {judgeStatusData.turn || 1}/{judgeStatusData.max_turns}
                    </span>
                  ) : null}
                </div>

                {/* Live judge logs shown while running and after completion */}
                {judgeStatusData?.logs && judgeStatusData.logs.length > 0 ? (
                  <div style={{ marginBottom: 10 }}>
                    <p className="section-kicker" style={{ marginBottom: 6 }}>Logs</p>
                    <div style={{ background: 'var(--panel-2)', border: '1px solid var(--line)', borderRadius: 'var(--radius)', padding: 12, maxHeight: 130, overflowY: 'auto', fontSize: '0.78rem', fontFamily: 'monospace' }}>
                      {judgeStatusData.logs.map((log, logIndex) => (
                        <div key={logIndex} style={{ marginBottom: 3, color: 'var(--ink)' }}>{"=> "}{log}</div>
                      ))}
                    </div>
                  </div>
                ) : null}

                {/* Active tool calls while judge is still running */}
                {judgeStatusData?.tool_calls && judgeStatusData.tool_calls.length > 0 ? (
                  <div style={{ marginBottom: 10 }}>
                    <p className="section-kicker" style={{ marginBottom: 6 }}>Active Tool Calls</p>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                      {judgeStatusData.tool_calls.map((toolCall, callIndex) => (
                        <div key={callIndex} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.75rem', padding: '7px 10px', borderRadius: 4, background: 'var(--panel-2)', border: '1px solid var(--line)' }}>
                          <Icons.play size={12} className="spinner" style={{ color: 'var(--info)' }} />
                          <span>Calling <strong>{toolCall.name}</strong></span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}

                {/* Full verdict display once judge completes */}
                {verdictData?.status === 'complete' && (
                  <div>
                    {/* Winner / no-winner banner */}
                    {verdictData.selected_model ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, padding: '10px 14px', borderRadius: 'var(--radius)', border: '1px solid var(--ok)', background: 'rgba(19,132,95,0.08)' }}>
                        <Icons.checkCircle size={16} style={{ color: 'var(--ok)', flexShrink: 0 }} />
                        <span style={{ fontWeight: 600, fontSize: '0.88rem' }}>
                          Selected: {verdictData.selected_model}
                        </span>
                      </div>
                    ) : (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12, padding: '10px 14px', borderRadius: 'var(--radius)', border: '1px solid var(--warning)', background: 'var(--warning-bg)' }}>
                        <Icons.alert size={16} style={{ color: 'var(--warning)', flexShrink: 0 }} />
                        <span style={{ fontSize: '0.88rem' }}>No model selected -- all candidates rejected</span>
                      </div>
                    )}

                    {/* Ranked model list with accept/reject badges and top-2 findings */}
                    {verdictData.ranked_models && verdictData.ranked_models.length > 0 ? (
                      <div style={{ marginBottom: 12 }}>
                        <p className="section-kicker" style={{ marginBottom: 8 }}>Ranked Models</p>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                          {verdictData.ranked_models.map((rankedModel, modelIndex) => {
                            const isWinner = rankedModel.model_name === verdictData.selected_model;
                            const isApproved = rankedModel.verdict === 'select';
                            const topFindings = (rankedModel.findings || []).slice(0, 2);
                            return (
                              <div
                                key={modelIndex}
                                style={{
                                  border: `1px solid ${isWinner ? 'var(--ok)' : 'var(--line)'}`,
                                  borderRadius: 'var(--radius)',
                                  padding: '9px 11px',
                                  background: 'var(--panel-2)',
                                }}
                              >
                                <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: topFindings.length > 0 ? 5 : 0 }}>
                                  <span style={{ fontWeight: 600, fontSize: '0.82rem', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                    #{rankedModel.rank || modelIndex + 1} {rankedModel.model_name}
                                  </span>
                                  {rankedModel.score != null ? (
                                    <span className="pill pill-run" style={{ fontSize: '0.68rem', flexShrink: 0 }}>
                                      {typeof rankedModel.score === 'number' ? rankedModel.score.toFixed(3) : rankedModel.score}
                                    </span>
                                  ) : null}
                                  <span
                                    className={`pill ${isApproved ? 'pill-done' : 'pill-err'}`}
                                    style={{ fontSize: '0.68rem', flexShrink: 0 }}
                                  >
                                    {isApproved ? 'APPROVED' : 'REJECTED'}
                                  </span>
                                </div>
                                {topFindings.map((finding, findingIndex) => (
                                  <div
                                    key={findingIndex}
                                    style={{ display: 'flex', gap: 5, alignItems: 'flex-start', fontSize: '0.74rem', marginTop: 3, color: 'var(--ink-muted)' }}
                                  >
                                    <Icons.dot size={9} style={{ marginTop: 3, flexShrink: 0 }} />
                                    <span><strong>{finding.label}</strong>: {finding.message}</span>
                                  </div>
                                ))}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ) : null}

                    {/* LLM commentary in a collapsible block */}
                    {verdictData.decision_trace?.llm_commentary ? (
                      <details style={{ marginBottom: 8, cursor: 'pointer', background: 'var(--panel-2)', border: '1px solid var(--line)', borderRadius: 'var(--radius)', padding: '8px 11px' }}>
                        <summary style={{ outline: 'none', fontWeight: 500, fontSize: '0.82rem', color: 'var(--ink)' }}>
                          LLM Commentary
                        </summary>
                        <pre className="reasoning-block" style={{ marginTop: 8, whiteSpace: 'pre-wrap', fontSize: '0.77rem', maxHeight: 180, overflowY: 'auto', background: 'none', border: 'none', padding: 0 }}>
                          {verdictData.decision_trace.llm_commentary}
                        </pre>
                      </details>
                    ) : (
                      <p className="muted" style={{ fontSize: '0.8rem', marginBottom: 8 }}>
                        Rule-based decision -- no LLM commentary recorded.
                      </p>
                    )}
                  </div>
                )}
              </section>
            )}

            {/* Re-run Training button -- appears ABOVE Continue to Leaderboard so the
                user must engage with the judge verdict before they can continue. */}
            {verdictData?.status === 'complete' && state.complete ? (
              <div>
                <button
                  className="btn btn-secondary full-width"
                  disabled={isRestarting}
                  onClick={() => setPendingConfirm('restart')}
                  type="button"
                >
                  {isRestarting
                    ? <span className="spinner small" style={{ marginRight: 8 }} />
                    : <Icons.play size={16} />}
                  {isRestarting ? 'Restarting...' : 'Re-run Training with Judge Feedback'}
                </button>
                {restartError ? (
                  <p className="inline-banner inline-banner-error" role="alert" style={{ marginTop: 6 }}>
                    <Icons.alert size={15} />
                    <span>{restartError}</span>
                  </p>
                ) : null}
              </div>
            ) : null}

            <TrainingSummary
              canContinue={state.complete && verdictData?.status === 'complete'}
              onContinue={() => go('leaderboard')}
              summary={state.summary}
              judgePending={state.summary != null && verdictData?.status !== 'complete'}
            />
            {/* Judge verdict never landed within the polling budget: offer a
                retry and a non-blocking path to the leaderboard so the user
                isn't stuck behind a spinner forever. */}
            {verdictStalled ? (
              <div className="inline-banner inline-banner-warn" role="alert">
                <Icons.alert size={16} />
                <span>The judge verdict did not arrive in time.</span>
                <span className="inline-banner-actions">
                  <button className="btn btn-secondary" onClick={restartVerdictPoll} type="button">
                    Retry verdict
                  </button>
                  <button className="btn btn-secondary" onClick={() => go('leaderboard')} type="button">
                    View leaderboard
                  </button>
                </span>
              </div>
            ) : null}
          </aside>

        </div>
      ) : (
        <section className="card empty-card training-empty" style={{ marginTop: 20 }}>
          <Icons.cpu size={34} />
          <h2>Waiting for training jobs</h2>
          <p className="muted">
            Queued model events will appear here as soon as the orchestrator starts.
            If nothing appears, the run may have failed to start.
          </p>
          <div style={{ display: 'flex', gap: 8, justifyContent: 'center', flexWrap: 'wrap', marginTop: 12 }}>
            {showReconnect ? (
              <button className="btn btn-secondary" onClick={() => connect(connectedSessionId)} type="button">
                <Icons.play size={16} /> Reconnect
              </button>
            ) : null}
            <button className="btn btn-secondary" onClick={() => go('upload')} type="button">
              <Icons.upload size={16} /> Back to New Run
            </button>
          </div>
        </section>
      )}

      {/* Analytics section: SHAP + Judge Reasoning + plots -- shown after training completes */}
      {state.complete && connectedSessionId ? (
        <TrainingAnalyticsSection sessionId={connectedSessionId} />
      ) : null}

      <ConfirmDialog
        open={Boolean(activeConfirm)}
        title={activeConfirm?.title}
        body={activeConfirm?.body}
        confirmLabel={activeConfirm?.confirmLabel}
        tone={activeConfirm?.tone}
        onConfirm={handleConfirm}
        onCancel={() => setPendingConfirm(null)}
      />
    </div>
  );
}

export default TrainingPage;
