import { useEffect, useState } from 'react';

import AgentAvatar from '../components/AgentAvatar.jsx';
import Sparkline from '../components/Sparkline.jsx';
import Stat from '../components/Stat.jsx';
import StatusPill from '../components/StatusPill.jsx';
import { fetchRuns, fetchRunStats } from '../api/client.js';
import { AGENTS } from '../data.js';
import { Icons } from '../icons.jsx';

function Dashboard({ go, startRun }) {
  const [stats, setStats] = useState(null);
  const [runs, setRuns] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    let ignore = false;

    async function loadDashboard() {
      try {
        const [statsPayload, runsPayload] = await Promise.all([
          fetchRunStats(),
          fetchRuns(5),
        ]);
        if (!ignore) {
          setStats(statsPayload);
          setRuns(runsPayload.runs || []);
        }
      } catch (dashboardError) {
        if (!ignore) {
          setError(dashboardError.message);
        }
      }
    }

    loadDashboard();
    return () => {
      ignore = true;
    };
  }, []);

  return (
    <div className="screen-stack">
      <section className="card hero-panel">
        <div>
          <p className="section-kicker">MITRA AI</p>
          <h2>A team of agents, one optimized model.</h2>
          <p className="muted">
            Upload a dataset, validate it, generate metadata, then hand off to
            the multi-agent pipeline.
          </p>
        </div>
        <div className="hero-actions">
          <button className="btn btn-primary" onClick={() => go('upload')} type="button">
            <Icons.upload size={16} />
            New Run
          </button>
          <button className="btn btn-secondary" onClick={startRun} type="button">
            <Icons.play size={16} />
            Preview Pipeline
          </button>
        </div>
      </section>

      {error ? (
        <div className="card callout error">
          <strong>Dashboard unavailable</strong>
          <span>{error}</span>
        </div>
      ) : null}

      <div className="stats-grid">
        <Stat icon="layers" label="Total uploads" value={stats?.total_uploads ?? '-'} />
        <Stat accent icon="checkCircle" label="Validated runs" value={stats?.validated_runs ?? '-'} />
        <Stat icon="brain" label="Metadata runs" value={stats?.metadata_runs ?? '-'} />
      </div>

      <div className="dashboard-grid">
        <section className="card panel-section">
          <div className="section-head">
            <div>
              <p className="section-kicker">Recent</p>
              <h2>Uploads</h2>
            </div>
            <button className="btn btn-secondary" onClick={() => go('upload')} type="button">
              <Icons.plus size={16} />
              New
            </button>
          </div>

          {runs.length ? (
            <div className="run-table">
              <div className="run-table-head">
                <span>Dataset</span>
                <span>Rows</span>
                <span>Validation</span>
                <span>Metadata</span>
              </div>
              {runs.map((run) => (
                <button
                  className="run-row"
                  key={run.session_id}
                  onClick={() => go('upload')}
                  type="button"
                >
                  <span>
                    <strong>{run.original_filename}</strong>
                    <small className="mono">{run.session_id}</small>
                  </span>
                  <span className="mono">{run.row_count ?? '-'}</span>
                  <StatusPill status={run.validation_status} />
                  <StatusPill status={run.metadata_status} />
                </button>
              ))}
            </div>
          ) : (
            <div className="empty-state">
              <Icons.database size={28} />
              <strong>No uploads yet</strong>
              <span>Upload a CSV or Excel file to create the first session.</span>
            </div>
          )}
        </section>

        <aside className="screen-stack">
          <section className="card panel-section">
            <div className="section-head">
              <div>
                <p className="section-kicker">Trend</p>
                <h2>Run Activity</h2>
              </div>
              <StatusPill status="idle" label="Local" />
            </div>
            <div className="trend-row">
              <div>
                <span className="mono trend-value">{runs.length}</span>
                <p className="muted">latest sessions</p>
              </div>
              <Sparkline points={[0, 1, 1, 2, 3, Math.max(3, runs.length + 2)]} />
            </div>
          </section>

          <section className="card panel-section">
            <div className="section-head">
              <div>
                <p className="section-kicker">Agents</p>
                <h2>Roster</h2>
              </div>
            </div>
            <div className="agent-list-compact">
              {AGENTS.map((agent) => (
                <div className="agent-compact" key={agent.id}>
                  <AgentAvatar agent={agent} size={30} />
                  <div>
                    <strong>{agent.name}</strong>
                    <span>{agent.role}</span>
                  </div>
                  <span className="tag">{agent.type}</span>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}

export default Dashboard;
