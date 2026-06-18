import FormField from './FormField.jsx';
import Segmented from './Segmented.jsx';
import { PROVIDERS } from '../data.js';
import { Icons } from '../icons.jsx';

function ByomFields({ settings, setSettings, publicConfig }) {
  const Cpu = Icons.cpu;
  const selectedProvider = settings.provider || 'anthropic';
  const baseModels = publicConfig?.llm?.base_models || {};
  const baseModel = baseModels[selectedProvider] || '';

  function updateSetting(key, value) {
    setSettings((currentSettings) => ({
      ...currentSettings,
      [key]: value,
    }));
  }

  return (
    <div className="byom-panel">
      <FormField label="Provider">
        <Segmented
          label="LLM provider"
          onChange={(value) => updateSetting('provider', value)}
          options={PROVIDERS.map((provider) => ({
            value: provider.value,
            label: provider.label,
          }))}
          value={selectedProvider}
        />
      </FormField>

      <div className="byom-grid">
        <FormField label="Model" hint="blank uses provider base">
          <input
            className="input"
            onChange={(event) => updateSetting('model', event.target.value)}
            placeholder={baseModel || 'base model'}
            type="text"
            value={settings.model || ''}
          />
        </FormField>

        <FormField label="API Key" hint="kept in memory for this session">
          <input
            autoComplete="off"
            className="input mono"
            onChange={(event) => updateSetting('apiKey', event.target.value)}
            placeholder="paste provider key"
            type="password"
            value={settings.apiKey || ''}
          />
        </FormField>

        <FormField label="API Gateway URL" hint="optional">
          <input
            className="input mono"
            onChange={(event) => updateSetting('gatewayUrl', event.target.value)}
            placeholder="https://litellm.local:4000"
            type="url"
            value={settings.gatewayUrl || ''}
          />
        </FormField>
      </div>

      <div className="inline-note">
        <Cpu size={15} />
        <span>
          Custom hosted model endpoints should use provider-style model names so
          `LLM_TYPE` matches the LiteLLM endpoint family.
        </span>
      </div>
    </div>
  );
}

export default ByomFields;
