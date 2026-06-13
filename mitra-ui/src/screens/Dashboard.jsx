import React, { useState, useEffect } from 'react';
import { Icons } from '../icons.jsx';
import { AGENTS } from '../data.js';
import { AgentAvatar } from '../components/AgentAvatar.jsx';
import { Stat } from '../components/Stat.jsx';
import { Sparkline } from '../components/Sparkline.jsx';
import { getRuns, getRunStats } from '../api.js';

function DriftDot({ drift }) {
  const colorMap = { stable: 'var(--ok)', watch: 'var(--warn)', '—': 'var(--ink-4)' };
  const labelMap = { stable: 'Stable', watch: 'Watch', '—': '—' };
  const color = colorMap[drift] || colorMap['—'];
  const label = labelMap[drift] || drift;
  return (
    <span className="row gap-6" style={{ fontSize: 12, color: 'var(--ink-2)' }}>
      <span style={{ width: 7, height: 7, borderRadius: 99, background: color }} />
      {label}
    </span>
  );
}

export function Dashboard({ go, startRun }) {
  const [runs, setRuns] = useState([]);
  const [stats, setStats] = useState({ total_runs: 0, models_trained: 0, best_accuracy: null, avg_run_time_min: null });
  const accSeries = [0.88, 0.91, 0.884, 0.94, 0.96, 0.973];

  useEffect(() => {
    getRuns(5).then(data => setRuns(data.runs || [])).catch(() => {});
    getRunStats().then(data => setStats(data)).catch(() => {});
  }, []);

  return (
    <div className="page page-in">
      {/* hero */}
      <div className="card" style={{
        padding: '26px 28px', marginBottom: 22, position: 'relative', overflow: 'hidden',
        background: 'linear-gradient(120deg, #fff 0%, #faf9ff 55%, #f3efff 100%)',
        border: '1px solid var(--accent-line)',
      }}>
        <div style={{
          position: 'absolute', right: -40, top: -50, width: 260, height: 260, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(108,71,255,.10), transparent 70%)',
        }} />
        <div className="row" style={{ justifyContent: 'space-between', alignItems: 'flex-start', gap: 20, position: 'relative' }}>
          <div className="col gap-10" style={{ maxWidth: 560 }}>
            <h1 style={{ fontSize: 25, fontWeight: 780 }}>A team of agents, one optimized model.</h1>
            <p className="muted" style={{ margin: 0, fontSize: 14, lineHeight: 1.55 }}>
              Upload a dataset and minimal metadata. MITRA's specialist agents profile, engineer,
              train, and tune in parallel — then the Judge converges on the best model and explains why.
            </p>
            <div className="row gap-10" style={{ marginTop: 6 }}>
              <button className="btn btn-primary" onClick={() => go('upload')}>
                <Icons.play size={15} />Start a new run
              </button>
              <button className="btn btn-secondary" onClick={() => go('leaderboard')}>
                <Icons.trophy size={16} />View last leaderboard
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* stats */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14, marginBottom: 22 }}>
        <Stat icon="layers" label="Total runs"     value={stats.total_runs}     />
        <Stat icon="cpu"    label="Models trained" value={stats.models_trained} accent />
        <Stat icon="target" label="Best accuracy"  value={stats.best_accuracy ? (stats.best_accuracy * 100).toFixed(1) : '—'} unit={stats.best_accuracy ? '%' : ''} accent />
        <Stat icon="gauge"  label="Avg run time"   value={stats.avg_run_time_min ?? '—'} unit={stats.avg_run_time_min ? 'min' : ''} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.55fr 1fr', gap: 18 }}>
        {/* recent runs table */}
        <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
          <div className="row" style={{ justifyContent: 'space-between', padding: '16px 20px', borderBottom: '1px solid var(--line)' }}>
            <div className="col" style={{ lineHeight: 1.3 }}>
              <h3 style={{ fontSize: 15, fontWeight: 700 }}>Recent runs</h3>
              <span className="faint" style={{ fontSize: 12 }}>Latest pipeline executions</span>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={() => go('upload')}>
              <Icons.plus size={15} />New
            </button>
          </div>

          {runs.length === 0 ? (
            <div style={{ padding: '32px 20px', textAlign: 'center', color: 'var(--ink-3)', fontSize: 13 }}>
              No runs yet. Start one above.
            </div>
          ) : (
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ color: 'var(--ink-3)', fontSize: 11, textTransform: 'uppercase', letterSpacing: '.04em' }}>
                  {['Run', 'Dataset', 'Task', 'Best model', 'Acc', 'Drift', ''].map((header, index) => (
                    <th key={index} style={{ textAlign: index >= 4 && index < 6 ? 'right' : 'left', fontWeight: 600, padding: '10px 20px', borderBottom: '1px solid var(--line)' }}>
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {runs.map((run, index) => (
                  <tr
                    key={run.session_id}
                    style={{ borderBottom: index < runs.length - 1 ? '1px solid var(--line-2)' : 'none', cursor: 'pointer' }}
                    onClick={() => go('leaderboard')}
                    onMouseEnter={evt => evt.currentTarget.style.background = 'var(--panel-2)'}
                    onMouseLeave={evt => evt.currentTarget.style.background = 'transparent'}
                  >
                    <td style={{ padding: '11px 20px' }}><span className="mono" style={{ fontSize: 12, color: 'var(--ink-2)' }}>{run.id}</span></td>
                    <td style={{ padding: '11px 20px', fontWeight: 600 }}>data.csv</td>
                    <td style={{ padding: '11px 20px', color: 'var(--ink-2)' }}><span className="tag">{run.task}</span></td>
                    <td style={{ padding: '11px 20px' }}>—</td>
                    <td style={{ padding: '11px 20px', textAlign: 'right' }} className="mono">{run.acc != null ? (run.acc * 100).toFixed(1) + '%' : '—'}</td>
                    <td style={{ padding: '11px 20px', textAlign: 'right' }}><div style={{ display: 'inline-flex' }}><DriftDot drift={run.drift} /></div></td>
                    <td style={{ padding: '11px 12px', textAlign: 'right', color: 'var(--ink-4)' }}><Icons.arrowR size={16} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* right column */}
        <div className="col gap-18">
          <div className="card" style={{ padding: '18px 20px' }}>
            <div className="row" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
              <h3 style={{ fontSize: 14.5, fontWeight: 700 }}>Accuracy trend</h3>
              <span className="pill pill-done"><span className="dot" />+9.3 pts</span>
            </div>
            <div className="row" style={{ justifyContent: 'space-between', alignItems: 'flex-end' }}>
              <div className="col gap-2">
                <span className="mono" style={{ fontSize: 26, fontWeight: 750 }}>
                  97.3<span style={{ fontSize: 14, color: 'var(--ink-3)' }}>%</span>
                </span>
                <span className="faint" style={{ fontSize: 11.5 }}>last 6 runs</span>
              </div>
              <Sparkline points={accSeries} w={150} h={48} />
            </div>
          </div>

          <div className="card" style={{ padding: '18px 20px', flex: 1 }}>
            <h3 style={{ fontSize: 14.5, fontWeight: 700, marginBottom: 4 }}>Agent roster</h3>
            <p className="faint" style={{ fontSize: 12, margin: '0 0 14px' }}>One agent per teammate · 8 owners</p>
            <div className="col gap-10">
              {AGENTS.map(agent => (
                <div key={agent.id} className="row gap-10">
                  <AgentAvatar agent={agent} size={30} />
                  <div className="col" style={{ lineHeight: 1.25, minWidth: 0 }}>
                    <span style={{ fontSize: 13, fontWeight: 600 }}>{agent.name}</span>
                    <span className="faint" style={{ fontSize: 11, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{agent.role}</span>
                  </div>
                  <div className="row gap-6" style={{ marginLeft: 'auto', flex: 'none' }}>
                    <span className="tag">{agent.type}</span>
                    <span style={{ width: 7, height: 7, borderRadius: 99, background: 'var(--ok)' }} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
