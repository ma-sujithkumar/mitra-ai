import { useEffect, useState } from 'react';

import StatusPill from './StatusPill.jsx';
import { fetchAdvancedConfig, saveAdvancedConfig } from '../api/client.js';
import { Icons } from '../icons.jsx';

// Render one editable control per param type. Booleans become a checkbox,
// enums a select, numerics a number input. Keyed by spec.type so there is no
// branching ladder in the render path.
function renderControl(param, value, onChange) {
  const controlByType = {
    bool: (
      <input
        checked={Boolean(value)}
        onChange={(event) => onChange(event.target.checked)}
        type="checkbox"
      />
    ),
    enum: (
      <select onChange={(event) => onChange(event.target.value)} value={value}>
        {(param.choices || []).map((choice) => (
          <option key={choice} value={choice}>{choice}</option>
        ))}
      </select>
    ),
    int: (
      <input
        max={param.maximum}
        min={param.minimum}
        onChange={(event) => onChange(event.target.value)}
        step="1"
        type="number"
        value={value}
      />
    ),
    float: (
      <input
        max={param.maximum}
        min={param.minimum}
        onChange={(event) => onChange(event.target.value)}
        step="0.01"
        type="number"
        value={value}
      />
    ),
  };
  return controlByType[param.type] || controlByType.float;
}

// Coerce a control's string value back to the param's declared type before we
// send it to the backend (which also validates).
function coerceValue(param, rawValue) {
  const coercers = {
    bool: (value) => Boolean(value),
    enum: (value) => value,
    int: (value) => Number.parseInt(value, 10),
    float: (value) => Number.parseFloat(value),
  };
  return (coercers[param.type] || coercers.float)(rawValue);
}

function AdvancedSettings({ activeSessionId }) {
  const [params, setParams] = useState([]);
  const [values, setValues] = useState({});
  const [loadState, setLoadState] = useState('idle');
  const [saveState, setSaveState] = useState({ status: 'idle', message: '' });

  useEffect(() => {
    let ignore = false;
    setLoadState('loading');
    fetchAdvancedConfig(activeSessionId || null)
      .then((payload) => {
        if (ignore) {
          return;
        }
        const loadedParams = payload.params || [];
        setParams(loadedParams);
        const nextValues = {};
        loadedParams.forEach((param) => {
          nextValues[param.key] = param.value;
        });
        setValues(nextValues);
        setLoadState('done');
      })
      .catch(() => {
        if (!ignore) {
          setLoadState('error');
        }
      });
    return () => {
      ignore = true;
    };
  }, [activeSessionId]);

  function updateValue(key, rawValue) {
    setValues((current) => ({ ...current, [key]: rawValue }));
    setSaveState({ status: 'idle', message: '' });
  }

  async function handleSave() {
    if (!activeSessionId) {
      setSaveState({
        status: 'failed',
        message: 'Start or select a run first; overrides are saved per session.',
      });
      return;
    }
    setSaveState({ status: 'saving', message: '' });
    const overrides = {};
    params.forEach((param) => {
      overrides[param.key] = coerceValue(param, values[param.key]);
    });
    try {
      await saveAdvancedConfig(activeSessionId, overrides);
      setSaveState({ status: 'saved', message: 'Saved for the next run.' });
    } catch (saveError) {
      const rejected = saveError.payload?.detail?.rejected;
      const detail = rejected ? JSON.stringify(rejected) : saveError.message;
      setSaveState({ status: 'failed', message: detail });
    }
  }

  // Group params by their declared group for a tidy layout.
  const groups = params.reduce((accumulator, param) => {
    (accumulator[param.group] = accumulator[param.group] || []).push(param);
    return accumulator;
  }, {});

  const SAVE_PILL_BY_STATUS = {
    idle: { status: 'idle', label: 'Defaults' },
    saving: { status: 'running', label: 'Saving' },
    saved: { status: 'passed', label: 'Saved' },
    failed: { status: 'failed', label: 'Rejected' },
  };
  const savePill = SAVE_PILL_BY_STATUS[saveState.status] || SAVE_PILL_BY_STATUS.idle;

  return (
    <section className="card panel-section">
      <div className="section-head">
        <div>
          <p className="section-kicker">Pipeline</p>
          <h2>Advanced Settings</h2>
        </div>
        <StatusPill status={savePill.status} label={savePill.label} spin={saveState.status === 'saving'} />
      </div>
      <p className="muted">
        Tunable pipeline parameters applied to the next run. Overrides are saved
        into the active session and read by the pipeline at invoke time.
      </p>

      {loadState === 'loading' ? <p className="muted">Loading parameters.</p> : null}
      {loadState === 'error' ? <p className="muted">Could not load advanced parameters.</p> : null}

      {Object.entries(groups).map(([groupName, groupParams]) => (
        <div className="advanced-group" key={groupName}>
          <h3 className="section-kicker">{groupName}</h3>
          <div className="advanced-grid">
            {groupParams.map((param) => (
              <label className="advanced-row" key={param.key}>
                <span>{param.label}</span>
                {renderControl(param, values[param.key], (next) => updateValue(param.key, next))}
              </label>
            ))}
          </div>
        </div>
      ))}

      <div className="smoke-test-row">
        <button
          className="btn btn-secondary"
          disabled={saveState.status === 'saving' || loadState !== 'done'}
          onClick={handleSave}
          type="button"
        >
          {saveState.status === 'saving' ? <span className="spinner" /> : <Icons.gear size={16} />}
          Save overrides
        </button>
        {saveState.message ? (
          <span className={`smoke-test-msg ${saveState.status}`}>{saveState.message}</span>
        ) : null}
      </div>
    </section>
  );
}

export default AdvancedSettings;
