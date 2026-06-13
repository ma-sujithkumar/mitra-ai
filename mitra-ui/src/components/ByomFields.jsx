import React, { useState } from 'react';
import { Icons } from '../icons.jsx';
import { PROVIDERS } from '../data.js';
import { Segmented } from './Segmented.jsx';
import { FormField, FIELD_INPUT_STYLE } from './FormField.jsx';

export function ByomFields({ llm, setLlm }) {
  const [showKey, setShowKey] = useState(false);
  const setField = (key, value) => setLlm({ ...llm, [key]: value });
  const activeProv = PROVIDERS.find(prov => prov.value === llm.provider) || PROVIDERS[0];

  return (
    <div>
      <FormField label="Provider">
        <Segmented
          value={llm.provider}
          onChange={value => setField('provider', value)}
          options={PROVIDERS.map(prov => ({ value: prov.value, label: prov.label }))}
        />
      </FormField>

      <FormField label="API key" hint="required">
        <div style={{ position: 'relative' }}>
          <input
            type={showKey ? 'text' : 'password'}
            value={llm.apiKey}
            onChange={evt => setField('apiKey', evt.target.value)}
            placeholder={activeProv.placeholder}
            className="focusable"
            style={{
              ...FIELD_INPUT_STYLE,
              paddingRight: 60,
              fontFamily: 'var(--mono)',
              letterSpacing: showKey ? '0' : '.08em',
            }}
          />
          <button
            onClick={() => setShowKey(visible => !visible)}
            type="button"
            style={{
              position: 'absolute', right: 6, top: '50%', transform: 'translateY(-50%)',
              border: 'none', background: 'var(--panel-3)', color: 'var(--ink-2)',
              cursor: 'pointer', fontSize: 11, fontWeight: 600, padding: '4px 9px', borderRadius: 6,
            }}
          >
            {showKey ? 'Hide' : 'Show'}
          </button>
        </div>
      </FormField>

      <FormField label="Gateway server URL" hint="optional">
        <input
          value={llm.gateway}
          onChange={evt => setField('gateway', evt.target.value)}
          className="focusable"
          placeholder="https://litellm.local:4000"
          style={{ ...FIELD_INPUT_STYLE, fontFamily: 'var(--mono)' }}
        />
      </FormField>

      <div className="row gap-8" style={{ fontSize: 11.5, color: 'var(--ink-3)', lineHeight: 1.45 }}>
        <Icons.cpu size={14} style={{ flex: 'none', marginTop: 1 }} />
        <span>Every agent routes through the LiteLLM factory - no agent ever makes a direct LLM call.</span>
      </div>
    </div>
  );
}

export default ByomFields;
