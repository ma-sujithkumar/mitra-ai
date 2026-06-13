import AgentAvatar from '../components/AgentAvatar.jsx';
import HBars from '../components/HBars.jsx';
import StatusPill from '../components/StatusPill.jsx';
import { AGENTS, LEADERBOARD, SHAP } from '../data.js';
import { Icons } from '../icons.jsx';

const judgeAgent = AGENTS.find((agent) => agent.id === 'judge');

function LeaderboardScreen({ startRun }) {
  const winner = LEADERBOARD[0];
  const maxAccuracy = Math.max(...LEADERBOARD.map((model) => model.acc));

  return (
    <div className="screen-stack">
      <section className="card hero-panel leaderboard-hero">
        <div className="winner-mark">
          <Icons.trophy size={28} />
        </div>
        <div>
          <StatusPill status="done" label="Judge converged" />
          <h2>{winner.model} is the recommended model</h2>
          <p className="muted">Prototype leaderboard using the Claude handoff model results.</p>
        </div>
        <button className="btn btn-primary" onClick={startRun} type="button">
          <Icons.play size={16} />
          New preview
        </button>
      </section>

      <div className="leaderboard-grid">
        <section className="card panel-section">
          <div className="section-head">
            <div>
              <p className="section-kicker">Models</p>
              <h2>Leaderboard</h2>
            </div>
          </div>
          <div className="leaderboard-table">
            <div className="leaderboard-head">
              <span>Rank</span>
              <span>Model</span>
              <span>Accuracy</span>
              <span>F1</span>
              <span>Judge</span>
            </div>
            {LEADERBOARD.map((model) => (
              <div className={model.winner ? 'leaderboard-row winner' : 'leaderboard-row'} key={model.model}>
                <span className="rank mono">{model.rank}</span>
                <span>
                  <strong>{model.model}</strong>
                  <small>{model.hp}</small>
                </span>
                <span className="metric-bar">
                  <div className="bar">
                    <i style={{ width: `${(model.acc / maxAccuracy) * 100}%` }} />
                  </div>
                  <em className="mono">{(model.acc * 100).toFixed(1)}</em>
                </span>
                <span className="mono">{model.f1.toFixed(3)}</span>
                <span className="mono">{model.judge.toFixed(1)}</span>
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
            <HBars data={SHAP} />
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
              {[
                ['Lowest overfit gap', '0.018 against the holdout split'],
                ['Accuracy floor met', 'All top candidates clear the threshold'],
                ['SHAP stable', 'Feature importance consistent across folds'],
              ].map(([title, detail]) => (
                <div className="reason-row" key={title}>
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
