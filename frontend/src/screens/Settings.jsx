import { useEffect, useState } from 'react';

import AdvancedSettings from '../components/AdvancedSettings.jsx';
import ByomFields from '../components/ByomFields.jsx';
import StatusPill from '../components/StatusPill.jsx';
import { fetchHealth, fetchPublicConfig, runLlmSmokeTest } from '../api/client.js';
import { llmConfigKey } from '../data.js';
import { Icons } from '../icons.jsx';

function Settings({ activeSessionId, llmSettings, llmSmokeStatus, setLlmSettings, setLlmSmokeStatus }) {
  const [health, setHealth] = useState(null);
  const [publicConfig, setPublicConfig] = useState(null);
  const [error, setError] = useState(null);

  async function handleSmokeTest() {
    setLlmSmokeStatus({ status: 'running', message: '', configKey: '' });
    try {
      const result = await runLlmSmokeTest(llmSettings);
      setLlmSmokeStatus({
        status: 'passed',
        message: `${result.provider} / ${result.model} responded in ${result.latency_ms} ms`,
        configKey: llmConfigKey(llmSettings),
      });
    } catch (smokeError) {
      setLlmSmokeStatus({
        status: 'failed',
        message: smokeError.message,
        configKey: '',
      });
    }
  }

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
  const SMOKE_PILL_BY_STATUS = {
    idle: { status: 'idle', label: 'Not tested' },
    running: { status: 'running', label: 'Testing' },
    passed: { status: 'passed', label: 'Verified' },
    failed: { status: 'failed', label: 'Failed' },
  };
  const smokeStatusPill = SMOKE_PILL_BY_STATUS[llmSmokeStatus.status] || SMOKE_PILL_BY_STATUS.idle;

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
            <p className="section-kicker">LLM</p>
            <h2>Run Configuration</h2>
          </div>
          <StatusPill
            status={smokeStatusPill.status}
            label={smokeStatusPill.label}
            spin={llmSmokeStatus.status === 'running'}
          />
        </div>
        <p className="muted">
          Provider, model, gateway, and key applied to every new run. A successful
          connection test is required before a run can start.
        </p>
        <ByomFields
          publicConfig={publicConfig}
          setSettings={setLlmSettings}
          settings={llmSettings}
        />
        <div className="smoke-test-row">
          <button
            className="btn btn-secondary"
            disabled={llmSmokeStatus.status === 'running'}
            onClick={handleSmokeTest}
            type="button"
          >
            {llmSmokeStatus.status === 'running' ? <span className="spinner" /> : <Icons.cpu size={16} />}
            Test connection
          </button>
          {llmSmokeStatus.message ? (
            <span className={`smoke-test-msg ${llmSmokeStatus.status}`}>
              {llmSmokeStatus.message}
            </span>
          ) : null}
        </div>
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

      <AdvancedSettings activeSessionId={activeSessionId} />
    </div>
  );
}

export default Settings;
