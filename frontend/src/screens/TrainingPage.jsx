import { useEffect, useMemo, useReducer, useRef, useState } from 'react';

import ModelTrainingCard from '../components/training/ModelTrainingCard.jsx';
import TrainingLogs from '../components/training/TrainingLogs.jsx';
import TrainingProgress from '../components/training/TrainingProgress.jsx';
import TrainingSummary from '../components/training/TrainingSummary.jsx';
import { streamTrainingEvents } from '../api/events.js';
import { cancelTraining, fetchTrainingStatus } from '../api/training.js';
import { Icons } from '../icons.jsx';
import {
  applyTrainingEvent,
  createTrainingState,
  overallTrainingProgress,
  selectTrainingModels,
  trainingCounts,
} from '../trainingState.js';

const SESSION_STORAGE_KEY = 'mitra.activeTrainingSession';

function reducer(state, action) {
  if (action.type === 'reset') {
    return createTrainingState();
  }
  if (action.type === 'event') {
    return applyTrainingEvent(state, action.payload);
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
          <p className="section-kicker">Training session</p>
          <h2>{connectedSessionId || 'Connect to a session'}</h2>
          <p className="muted">{connectionMessage || 'Use the session created on New Run, or paste a session ID.'}</p>
          {backendStatus?.status ? (
            <span className="mono muted">Backend status: {backendStatus.status}</span>
          ) : null}
        </div>
        <div className="training-session-controls">
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
    </div>
  );
}

export default TrainingPage;
