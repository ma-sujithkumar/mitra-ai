import ByomFields from './ByomFields.jsx';
import StatusPill from './StatusPill.jsx';
import { runLlmSmokeTest } from '../api/client.js';
import { llmConfigKey } from '../data.js';
import { Icons } from '../icons.jsx';

const SMOKE_PILL_BY_STATUS = {
  idle: { status: 'idle', label: 'Not tested' },
  running: { status: 'running', label: 'Testing' },
  passed: { status: 'passed', label: 'Verified' },
  failed: { status: 'failed', label: 'Failed' },
};

// Shared "Run Configuration" panel: provider/model/key/gateway fields plus the
// connection test. Rendered both on the New Run screen (mandatory, front and
// center) and on Settings (canonical place to revisit it), so there is exactly
// one implementation of the test/retry flow.
function RunConfigurationPanel({ llmSettings, llmSmokeStatus, setLlmSettings, setLlmSmokeStatus, publicConfig }) {
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

  const smokeStatusPill = SMOKE_PILL_BY_STATUS[llmSmokeStatus.status] || SMOKE_PILL_BY_STATUS.idle;
  const isRunning = llmSmokeStatus.status === 'running';
  const isFailed = llmSmokeStatus.status === 'failed';

  return (
    <section className="card panel-section run-config-panel">
      <div className="section-head">
        <div>
          <p className="section-kicker">Mandatory · LLM</p>
          <h2>Run Configuration</h2>
        </div>
        <StatusPill status={smokeStatusPill.status} label={smokeStatusPill.label} spin={isRunning} />
      </div>

      <div className="callout tutorial">
        <Icons.info size={16} />
        <span>
          Pick a provider and model, then click <strong>Test connection</strong> to verify it
          works. Once verified here, the <strong>Validate &amp; Review</strong> button on this
          page unlocks.
        </span>
      </div>

      <ByomFields publicConfig={publicConfig} setSettings={setLlmSettings} settings={llmSettings} />

      <div className="smoke-test-row">
        <button
          className="btn btn-secondary"
          disabled={isRunning}
          onClick={handleSmokeTest}
          type="button"
        >
          {isRunning ? <span className="spinner" /> : <Icons.cpu size={16} />}
          {isFailed ? 'Retry connection test' : 'Test connection'}
        </button>
        {llmSmokeStatus.message ? (
          <span className={`smoke-test-msg ${llmSmokeStatus.status}`}>
            {llmSmokeStatus.message}
          </span>
        ) : null}
      </div>
    </section>
  );
}

export default RunConfigurationPanel;
