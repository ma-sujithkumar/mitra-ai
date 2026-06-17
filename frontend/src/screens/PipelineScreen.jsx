import { useEffect, useMemo, useRef, useState } from 'react';

import AgentAvatar from '../components/AgentAvatar.jsx';
import StatusPill from '../components/StatusPill.jsx';
import { AGENTS, STAGE_LOGS, STAGES } from '../data.js';
import { Icons } from '../icons.jsx';

const AGENT_BY_ID = Object.fromEntries(AGENTS.map((agent) => [agent.id, agent]));

function nowTimestamp() {
  const date = new Date();
  return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}:${String(date.getSeconds()).padStart(2, '0')}`;
}

function PipelineScreen({ go, runState, setRunState, startRun }) {
  const [cursor, setCursor] = useState(0);
  const [paused, setPaused] = useState(false);
  const [selectedStage, setSelectedStage] = useState('all');
  const logRef = useRef(null);
  const flatEvents = useMemo(
    () => STAGES.flatMap((stage, stageIndex) => (
      (STAGE_LOGS[stage.key] || []).map((line, lineIndex, lines) => ({
        stageIndex,
        stageKey: stage.key,
        level: line[0],
        message: line[1],
        lineIndex,
        lineCount: lines.length,
      }))
    )),
    [],
  );

  useEffect(() => {
    if (runState !== 'running' || paused) {
      return undefined;
    }
    if (cursor >= flatEvents.length) {
      setRunState('done');
      return undefined;
    }
    const timeout = setTimeout(() => setCursor((currentCursor) => currentCursor + 1), 420);
    return () => clearTimeout(timeout);
  }, [cursor, flatEvents.length, paused, runState, setRunState]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [cursor, selectedStage]);

  function restart() {
    setCursor(0);
    setPaused(false);
    startRun();
  }

  if (runState === 'idle') {
    return (
      <div className="empty-center">
        <div className="card empty-card">
          <Icons.flow size={34} />
          <h2>No active run</h2>
          <p className="muted">Start a preview or complete metadata generation from New Run.</p>
          <button className="btn btn-primary" onClick={restart} type="button">
            <Icons.play size={16} />
            Start preview
          </button>
        </div>
      </div>
    );
  }

  const visibleEvents = flatEvents.slice(0, cursor);
  const currentEvent = visibleEvents[visibleEvents.length - 1];
  const currentStageIndex = runState === 'done'
    ? STAGES.length - 1
    : currentEvent?.stageIndex || 0;
  const currentStageProgress = currentEvent
    ? (currentEvent.lineIndex + 1) / currentEvent.lineCount
    : 0;
  const overallProgress = runState === 'done'
    ? 100
    : Math.round(((currentStageIndex + currentStageProgress) / STAGES.length) * 100);
  const filteredEvents = selectedStage === 'all'
    ? visibleEvents
    : visibleEvents.filter((event) => event.stageKey === selectedStage);

  function stageState(stageIndex) {
    if (runState === 'done' || stageIndex < currentStageIndex) {
      return 'done';
    }
    if (stageIndex === currentStageIndex) {
      return 'running';
    }
    return 'queued';
  }

  return (
    <div className="screen-stack">
      <section className="card run-header">
        <div>
          <p className="section-kicker">Pipeline Preview</p>
          <h2>Session pipeline</h2>
          <p className="muted">Prototype stages after Epic 1 metadata generation.</p>
        </div>
        <div className="run-actions">
          <span className="mono progress-value">{overallProgress}%</span>
          <button className="btn btn-secondary" onClick={() => setPaused((value) => !value)} type="button">
            {paused ? <Icons.play size={16} /> : <Icons.pause size={16} />}
            {paused ? 'Resume' : 'Pause'}
          </button>
          <button className="btn btn-secondary" onClick={restart} type="button">
            <Icons.play size={16} />
            Restart
          </button>
          {runState === 'done' ? (
            <button className="btn btn-primary" onClick={() => go('leaderboard')} type="button">
              <Icons.trophy size={16} />
              Leaderboard
            </button>
          ) : null}
        </div>
        <div className="bar header-bar">
          <i style={{ width: `${overallProgress}%` }} />
        </div>
      </section>

      <div className="pipeline-grid">
        <section className="card panel-section">
          <div className="section-head">
            <div>
              <p className="section-kicker">Stages</p>
              <h2>Artifact Graph</h2>
            </div>
            <StatusPill status={runState === 'done' ? 'done' : 'running'} spin={runState === 'running' && !paused} />
          </div>
          <div className="pipeline-stage-list">
            {STAGES.map((stage, stageIndex) => {
              const agent = stage.agent ? AGENT_BY_ID[stage.agent] : null;
              const state = stageState(stageIndex);
              return (
                <button
                  className={`pipeline-stage ${selectedStage === stage.key ? 'active' : ''}`}
                  key={stage.key}
                  onClick={() => setSelectedStage(stage.key)}
                  type="button"
                >
                  {agent ? (
                    <AgentAvatar agent={agent} state={state} />
                  ) : (
                    <div className={`system-stage ${state}`}>
                      {state === 'done' ? <Icons.check size={18} /> : <Icons.cpu size={18} />}
                    </div>
                  )}
                  <span>
                    <strong>{stage.label}</strong>
                    <small>{stage.sub}</small>
                    <em className="mono">{stage.artifact}</em>
                  </span>
                  <StatusPill status={state} spin={state === 'running' && !paused} />
                </button>
              );
            })}
          </div>
        </section>

        <aside className="screen-stack">
          <section className="card terminal-panel">
            <div className="terminal-head">
              <span>SSE event stream</span>
              <button className="terminal-filter" onClick={() => setSelectedStage('all')} type="button">
                all stages
              </button>
            </div>
            <div className="terminal-body" ref={logRef}>
              {filteredEvents.length ? filteredEvents.map((event, index) => (
                <div className="terminal-line" key={`${event.stageKey}-${index}`}>
                  <span>{nowTimestamp()}</span>
                  <strong className={`level-${event.level}`}>{event.level}</strong>
                  <em>{event.message}</em>
                </div>
              )) : (
                <span className="terminal-empty">awaiting events</span>
              )}
            </div>
          </section>

          <section className="card panel-section">
            <div className="section-head">
              <div>
                <p className="section-kicker">Judge</p>
                <h2>Verdict</h2>
              </div>
              <StatusPill status={runState === 'done' ? 'done' : 'queued'} />
            </div>
            {runState === 'done' ? (
              <div className="winner-card">
                <Icons.trophy size={22} />
                <strong>XGBoost wins</strong>
                <span>Best F1 with the lowest overfit gap.</span>
              </div>
            ) : (
              <p className="muted">The Judge waits for evaluation artifacts.</p>
            )}
          </section>
        </aside>
      </div>
    </div>
  );
}

export default PipelineScreen;
