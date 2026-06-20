import { useEffect, useState } from 'react';

import AdvancedSettings from '../components/AdvancedSettings.jsx';
import RunConfigurationPanel from '../components/RunConfigurationPanel.jsx';
import StatusPill from '../components/StatusPill.jsx';
import Toast from '../components/Toast.jsx';
import { fetchHealth, fetchPublicConfig } from '../api/client.js';
import { Icons } from '../icons.jsx';

function Settings({ activeSessionId, backRoute, go, llmSettings, llmSmokeStatus, setLlmSettings, setLlmSmokeStatus }) {
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
          // Default the active provider to the first server-supported one
          // only when the current selection is not available.
          const supportedProviders = configPayload.llm?.providers || [];
          setLlmSettings((currentSettings) => (
            supportedProviders.length && !supportedProviders.includes(currentSettings.provider)
              ? { ...currentSettings, provider: supportedProviders[0] }
              : currentSettings
          ));
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
  }, [setLlmSettings]);

  const providers = publicConfig?.llm?.providers || [];
  const baseModels = publicConfig?.llm?.base_models || {};

  return (
    <div className="screen-stack">
      <Toast message={error} onDismiss={() => setError(null)} tone="error" />

      {go ? (
        <button
          className="btn btn-secondary btn-sm back-button"
          onClick={() => go(backRoute || 'upload')}
          type="button"
        >
          <Icons.arrowLeft size={14} />
          Back
        </button>
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

      <RunConfigurationPanel
        llmSettings={llmSettings}
        llmSmokeStatus={llmSmokeStatus}
        publicConfig={publicConfig}
        setLlmSettings={setLlmSettings}
        setLlmSmokeStatus={setLlmSmokeStatus}
      />

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

      <AdvancedSettings activeSessionId={activeSessionId} />
    </div>
  );
}

export default Settings;
