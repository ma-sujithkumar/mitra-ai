import { useState } from 'react';

import FormField from './FormField.jsx';
import Segmented from './Segmented.jsx';
import { PROVIDERS } from '../data.js';
import { Icons } from '../icons.jsx';

// Sentinel value for the dropdown's "Custom..." entry so users can still type a
// model name for self-hosted / BYOM endpoints.
const CUSTOM_MODEL_OPTION = '__custom__';

function ByomFields({ settings, setSettings, publicConfig }) {
  const Cpu = Icons.cpu;
  const selectedProvider = settings.provider || 'anthropic';
  const baseModels = publicConfig?.llm?.base_models || {};
  const baseModel = baseModels[selectedProvider] || '';
  const baseUrls = publicConfig?.llm?.base_urls || {};
  const baseUrl = baseUrls[selectedProvider] || '';
  const modelOptions = publicConfig?.llm?.model_options?.[selectedProvider] || [];

  function updateSetting(key, value) {
    setSettings((currentSettings) => ({
      ...currentSettings,
      [key]: value,
    }));
  }

  const optionValues = modelOptions.map((option) => option.value);
  const defaultModel = modelOptions[0]?.value || '';
  const currentModel = settings.model || '';
  // A typed model that is not one of the provider's listed options is "custom".
  const isCustomModel = currentModel !== '' && !optionValues.includes(currentModel);
  const [customMode, setCustomMode] = useState(false);
  const showCustomInput = customMode || isCustomModel;
  // Blank model means "use the first listed model" (the provider default), so
  // the dropdown shows that entry selected rather than an extra base option.
  const selectValue = showCustomInput ? CUSTOM_MODEL_OPTION : (currentModel || defaultModel);

  function handleProviderChange(provider) {
    // Switching provider invalidates the previously selected model; reset to the
    // new provider's base model (blank) so the dropdown shows valid choices.
    setCustomMode(false);
    setSettings((currentSettings) => ({
      ...currentSettings,
      provider,
      model: '',
    }));
  }

  function handleModelSelect(value) {
    if (value === CUSTOM_MODEL_OPTION) {
      setCustomMode(true);
      updateSetting('model', '');
      return;
    }
    setCustomMode(false);
    updateSetting('model', value);
  }

  return (
    <div className="byom-panel">
      <FormField label="Provider">
        <Segmented
          label="LLM provider"
          onChange={handleProviderChange}
          options={PROVIDERS.map((provider) => ({
            value: provider.value,
            label: provider.label,
          }))}
          value={selectedProvider}
        />
      </FormField>

      <div className="byom-grid">
        <FormField label="Model" hint="pick a model or choose Custom">
          <select
            className="input"
            onChange={(event) => handleModelSelect(event.target.value)}
            value={selectValue}
          >
            {modelOptions.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
            <option value={CUSTOM_MODEL_OPTION}>Custom...</option>
          </select>
          {showCustomInput ? (
            <input
              className="input"
              onChange={(event) => updateSetting('model', event.target.value)}
              placeholder={baseModel || 'provider/model-name'}
              style={{ marginTop: 8 }}
              type="text"
              value={currentModel}
            />
          ) : null}
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

        <FormField label="API Gateway URL" hint="blank uses provider default">
          <input
            className="input mono"
            onChange={(event) => updateSetting('gatewayUrl', event.target.value)}
            placeholder={baseUrl || 'https://litellm.local:4000'}
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
