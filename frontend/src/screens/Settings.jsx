import { useEffect, useState } from 'react';

import StatusPill from '../components/StatusPill.jsx';
import { fetchHealth, fetchPublicConfig } from '../api/client.js';

function Settings() {
  const [health, setHealth] = useState(null);
  const [publicConfig, setPublicConfig] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let ignore = false;

    async function loadSettings() {
      try {
        const [healthPayload, configPayload] = await Promise.all([
          fetchHealth(),
          fetchPublicConfig(),
        ]);
        if (!ignore) {
          setHealth(healthPayload);
          setPublicConfig(configPayload);
        }
      } catch (settingsError) {
        if (!ignore) {
          setError(settingsError.message);
        }
      }
    }

    loadSettings();
    return () => {
      ignore = true;
    };
  }, []);

  const providers = publicConfig?.llm?.providers || [];
  const baseModels = publicConfig?.llm?.base_models || {};

  return (
    <div className="screen-stack">
      {error ? (
        <div className="card callout error">
          <strong>Settings unavailable</strong>
          <span>{error}</span>
        </div>
      ) : null}

      <section className="settings-grid">
        <article className="card panel-section">
          <div className="section-head">
            <div>
              <p className="section-kicker">Server</p>
              <h2>Health</h2>
            </div>
            <StatusPill
              status={health?.status === 'ok' ? 'passed' : 'queued'}
              label={health?.status || 'Loading'}
            />
          </div>
          <dl className="detail-list">
            <div>
              <dt>Uptime</dt>
              <dd>{health ? `${health.uptime_seconds}s` : '-'}</dd>
            </div>
            <div>
              <dt>LLM Provider</dt>
              <dd>{health?.llm?.provider || 'Not configured'}</dd>
            </div>
            <div>
              <dt>Environment</dt>
              <dd>{health?.llm?.env_configured ? 'Configured' : 'Needs credentials'}</dd>
            </div>
          </dl>
        </article>

        <article className="card panel-section">
          <div className="section-head">
            <div>
              <p className="section-kicker">Uploads</p>
              <h2>Limits</h2>
            </div>
            <StatusPill status="idle" label="Public" />
          </div>
          <dl className="detail-list">
            <div>
              <dt>Extensions</dt>
              <dd>{publicConfig?.upload?.allowed_extensions?.join(', ') || '-'}</dd>
            </div>
            <div>
              <dt>Max Size</dt>
              <dd>
                {publicConfig ? `${publicConfig.upload.max_file_size_mb} MB` : '-'}
              </dd>
            </div>
            <div>
              <dt>Recent Uploads</dt>
              <dd>{publicConfig?.upload?.recent_upload_limit || '-'}</dd>
            </div>
          </dl>
        </article>
      </section>

      <section className="card panel-section">
        <div className="section-head">
          <div>
            <p className="section-kicker">Models</p>
            <h2>Provider Defaults</h2>
          </div>
          <StatusPill status="idle" label="No secrets" />
        </div>
        <div className="model-grid">
          {providers.map((provider) => (
            <div className="model-row" key={provider}>
              <span>{provider}</span>
              <code>{baseModels[provider]}</code>
            </div>
          ))}
          {!providers.length ? <p className="muted">Loading provider defaults.</p> : null}
        </div>
      </section>
    </div>
  );
}

export default Settings;
