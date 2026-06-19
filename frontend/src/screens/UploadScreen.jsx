import { useEffect, useMemo, useState } from 'react';

import FormField from '../components/FormField.jsx';
import MetadataProgress from '../components/MetadataProgress.jsx';
import Segmented from '../components/Segmented.jsx';
import StatusPill from '../components/StatusPill.jsx';
import {
  fetchPublicConfig,
  fetchRecentUploads,
  startMetadata,
  startValidation,
  uploadDataset,
} from '../api/client.js';
import { streamMetadataEvents, streamValidationEvents } from '../api/events.js';
import { startTraining } from '../api/training.js';
import { llmConfigKey } from '../data.js';
import { Icons } from '../icons.jsx';

const VALIDATION_KEYS = [
  'format',
  'rows',
  'nulls',
  'variance',
  'pii',
  'target',
  'metadata_match',
];

const PROBLEM_OPTIONS = [
  { value: 'auto', label: 'Auto' },
  { value: 'classification', label: 'Classify' },
  { value: 'regression', label: 'Regress' },
  { value: 'unsupervised', label: 'Cluster' },
];

function UploadScreen({ go, startRun, llmSettings, llmSmokeStatus }) {
  const [publicConfig, setPublicConfig] = useState(null);
  const [recentUploads, setRecentUploads] = useState([]);
  const [datasetFile, setDatasetFile] = useState(null);
  const [metadataFile, setMetadataFile] = useState(null);
  const [selectedRecent, setSelectedRecent] = useState(null);
  const [sessionSummary, setSessionSummary] = useState(null);
  const [activeSessionId, setActiveSessionId] = useState('');
  const [validationEvents, setValidationEvents] = useState([]);
  const [metadataEvents, setMetadataEvents] = useState([]);
  const [validationPhase, setValidationPhase] = useState('idle');
  const [metadataPhase, setMetadataPhase] = useState('idle');
  const [trainingPhase, setTrainingPhase] = useState('idle');
  const [error, setError] = useState(null);
  const [form, setForm] = useState({
    problemType: 'auto',
    targetCol: '',
    validationSplit: 0.8,
    description: '',
    dataType: 'tabular',
  });

  useEffect(() => {
    let ignore = false;

    async function loadOptions() {
      try {
        const [configPayload, recentPayload] = await Promise.all([
          fetchPublicConfig(),
          fetchRecentUploads(5),
        ]);
        if (!ignore) {
          setPublicConfig(configPayload);
          setRecentUploads(recentPayload.uploads || []);
          setForm((currentForm) => ({
            ...currentForm,
            validationSplit: configPayload.pipeline.train_test_split,
          }));
        }
      } catch (optionsError) {
        if (!ignore) {
          setError(optionsError.message);
        }
      }
    }

    loadOptions();
    return () => {
      ignore = true;
    };
  }, []);

  const acceptedExtensions = publicConfig?.upload?.allowed_extensions || ['.csv', '.xls', '.xlsx'];
  const validationByKey = useMemo(
    () => Object.fromEntries(
      validationEvents
        .filter((event) => event.type === 'check')
        .map((event) => [event.key, event])
    ),
    [validationEvents],
  );
  // Ray training is only allowed once validation has passed and metadata
  // generation has completed for the current inputs.
  const canRunPipeline = validationPhase === 'done' && metadataPhase === 'done';
  const baseModel = publicConfig?.llm?.base_models?.[llmSettings.provider] || '';
  const effectiveModel = llmSettings.model || baseModel || 'Provider base model';
  const reviewStarted = validationPhase !== 'idle';
  // Mandatory gates for Validate & Review: a dataset is selected and the
  // current LLM configuration passed a connection test in Settings.
  const hasDataset = Boolean(datasetFile || selectedRecent);
  const llmVerified = (
    llmSmokeStatus.status === 'passed'
    && llmSmokeStatus.configKey === llmConfigKey(llmSettings)
  );
  const busy = validationPhase === 'running' || metadataPhase === 'running' || trainingPhase === 'running';
  const canReview = hasDataset && llmVerified && !busy;

  function updateForm(key, value) {
    setForm((currentForm) => ({
      ...currentForm,
      [key]: value,
    }));
  }

  function resetRunState() {
    setValidationEvents([]);
    setMetadataEvents([]);
    setValidationPhase('idle');
    setMetadataPhase('idle');
    setTrainingPhase('idle');
    setSessionSummary(null);
    setActiveSessionId('');
    setError(null);
  }

  function handleDatasetChange(file) {
    setDatasetFile(file);
    setSelectedRecent(null);
    resetRunState();
  }

  function handleMetadataChange(file) {
    // Changing the optional metadata file invalidates any prior validation
    // and metadata run, so reset the phases to re-gate Ray training.
    setMetadataFile(file);
    resetRunState();
  }

  function handleRecentSelect(uploadRecord) {
    resetRunState();
    setSelectedRecent(uploadRecord);
    setActiveSessionId(uploadRecord.session_id);
    setDatasetFile(null);
    setMetadataFile(null);
  }

  async function handleValidateAndReview() {
    setError(null);
    setValidationEvents([]);
    setMetadataEvents([]);
    setValidationPhase('running');
    setMetadataPhase('idle');

    try {
      let sessionId = selectedRecent?.session_id;
      if (datasetFile) {
        const uploadPayload = await uploadDataset({
          datasetFile,
          metadataFile,
        });
        sessionId = uploadPayload.session_id;
        setActiveSessionId(sessionId);
        setSessionSummary(uploadPayload.summary);
      } else if (selectedRecent) {
        setSessionSummary({
          row_count: selectedRecent.row_count,
          column_count: selectedRecent.column_count,
          file_size_bytes: selectedRecent.file_size_bytes,
        });
      }

      setActiveSessionId(sessionId || '');

      if (!sessionId) {
        throw new Error('Select a dataset file or recent upload.');
      }

      await startValidation({
        sessionId,
        targetCol: form.problemType === 'unsupervised' ? null : form.targetCol,
        validationSplit: Number(form.validationSplit),
      });
      const validationDoneEvent = await collectValidationEvents(sessionId);
      setValidationPhase(validationDoneEvent.passed ? 'done' : 'error');

      if (validationDoneEvent.passed) {
        await runMetadata(sessionId);
      }
    } catch (flowError) {
      setError(flowError.message);
      if (validationPhase === 'running') {
        setValidationPhase('error');
      }
    }
  }

  function collectValidationEvents(sessionId) {
    return new Promise((resolve, reject) => {
      streamValidationEvents(sessionId, {
        onEvent: (event) => {
          setValidationEvents((currentEvents) => [...currentEvents, event]);
        },
        onDone: resolve,
        onError: reject,
      });
    });
  }

  async function runMetadata(sessionId) {
    setMetadataPhase('running');

    try {
      await startMetadata({
        sessionId,
        description: form.description,
        targetCol: form.problemType === 'unsupervised' ? null : form.targetCol,
        problemType: form.problemType === 'auto' ? null : form.problemType,
        provider: llmSettings.provider,
        model: llmSettings.model,
        apiKey: llmSettings.apiKey || '',
        gatewayUrl: llmSettings.gatewayUrl,
      });

      const metadataDoneEvent = await collectMetadataEvents(sessionId);
      setMetadataPhase(metadataDoneEvent.type === 'done' ? 'done' : 'error');
    } catch (metadataError) {
      setMetadataPhase('error');
      throw metadataError;
    }
  }

  function collectMetadataEvents(sessionId) {
    return new Promise((resolve, reject) => {
      streamMetadataEvents(sessionId, {
        onEvent: (event) => {
          setMetadataEvents((currentEvents) => [...currentEvents, event]);
        },
        onDone: resolve,
        onError: reject,
      });
    });
  }

  async function handleStartTraining() {
    const sessionId = String(activeSessionId || '').trim();
    if (!sessionId) {
      setError('No active session is available for training.');
      return;
    }

    setError(null);
    setTrainingPhase('running');
    try {
      await startTraining({
        sessionId,
        targetColumn: form.problemType === 'unsupervised' ? null : form.targetCol,
        executionMode: 'ray',
      });
      setTrainingPhase('accepted');
      startRun(sessionId);
    } catch (trainingError) {
      setTrainingPhase('error');
      setError(trainingError.message);
    }
  }

  return (
    <div className="screen-stack">
      {error ? (
        <div className="card callout error">
          <strong>Run setup failed</strong>
          <span>{error}</span>
        </div>
      ) : null}

      <div className="upload-grid">
        <section className="card panel-section">
          <div className="section-head">
            <div>
              <p className="section-kicker">Dataset</p>
              <h2>Upload</h2>
            </div>
            <StatusPill status={datasetFile || selectedRecent ? 'passed' : 'queued'} label={datasetFile || selectedRecent ? 'Selected' : 'Required'} />
          </div>

          <label
            className="dropzone"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              handleDatasetChange(event.dataTransfer.files[0]);
            }}
          >
            <input
              accept={acceptedExtensions.join(',')}
              onChange={(event) => handleDatasetChange(event.target.files?.[0] || null)}
              type="file"
            />
            <Icons.upload size={26} />
            <strong>{datasetFile ? datasetFile.name : 'Drop a dataset or browse'}</strong>
            <span>{acceptedExtensions.join(', ')} up to {publicConfig?.upload?.max_file_size_mb || 200} MB</span>
          </label>

          <FormField label="Optional metadata file" hint=".json or .csv">
            <input
              accept=".json,.csv"
              className="input"
              disabled={!datasetFile}
              onChange={(event) => handleMetadataChange(event.target.files?.[0] || null)}
              type="file"
            />
          </FormField>

          <div className="recent-list">
            <p className="section-kicker">Latest 5 uploaded datasets</p>
            {recentUploads.length ? recentUploads.map((uploadRecord) => (
              <button
                className={`recent-row ${selectedRecent?.session_id === uploadRecord.session_id ? 'active' : ''}`}
                key={uploadRecord.session_id}
                onClick={() => handleRecentSelect(uploadRecord)}
                type="button"
              >
                <Icons.doc size={17} />
                <span>
                  <strong>{uploadRecord.original_filename}</strong>
                  <small className="mono">{uploadRecord.session_id}</small>
                </span>
                <em>{uploadRecord.row_count ?? '-'} rows</em>
              </button>
            )) : (
              <div className="empty-state compact">
                <span>No uploaded datasets found.</span>
              </div>
            )}
          </div>
        </section>

        <section className="card panel-section">
          <div className="section-head">
            <div>
              <p className="section-kicker">Run Metadata</p>
              <h2>Review Inputs</h2>
            </div>
          </div>

          <FormField label="Problem type">
            <Segmented
              label="Problem type"
              onChange={(value) => updateForm('problemType', value)}
              options={PROBLEM_OPTIONS}
              value={form.problemType}
            />
          </FormField>

          <FormField label="Target column" hint="blank for unsupervised">
            <input
              className="input"
              onChange={(event) => updateForm('targetCol', event.target.value)}
              type="text"
              value={form.targetCol}
            />
          </FormField>

          <FormField label="Validation split">
            <input
              className="input"
              max="0.95"
              min="0.05"
              onChange={(event) => updateForm('validationSplit', event.target.value)}
              step="0.05"
              type="number"
              value={form.validationSplit}
            />
          </FormField>

          <FormField label="Description">
            <textarea
              className="input textarea"
              onChange={(event) => updateForm('description', event.target.value)}
              rows={4}
              value={form.description}
            />
          </FormField>

          <FormField label="Data type">
            <select
              className="input"
              onChange={(event) => updateForm('dataType', event.target.value)}
              value={form.dataType}
            >
              <option value="tabular">Tabular</option>
              <option value="time_series">Time series</option>
              <option value="text">Text</option>
            </select>
          </FormField>

          <div className="llm-ref">
            <div className="llm-ref-head">
              <span className="section-kicker">LLM for this run</span>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => go('settings')}
                type="button"
              >
                <Icons.gear size={14} />
                Configure in Settings
              </button>
            </div>
            <dl className="detail-list">
              <div>
                <dt>Provider</dt>
                <dd className="capitalize">{llmSettings.provider || 'Not set'}</dd>
              </div>
              <div>
                <dt>Model</dt>
                <dd className="mono">{effectiveModel}</dd>
              </div>
              <div>
                <dt>API Key</dt>
                <dd>{llmSettings.apiKey ? 'Set' : 'Using server credentials'}</dd>
              </div>
              <div>
                <dt>Connection</dt>
                <dd>
                  <StatusPill
                    status={llmVerified ? 'passed' : 'queued'}
                    label={llmVerified ? 'Verified' : 'Test in Settings'}
                  />
                </dd>
              </div>
            </dl>
          </div>

          {!canReview && !busy ? (
            <p className="muted gate-hint">
              {!hasDataset ? 'Select a dataset to continue. ' : ''}
              {!llmVerified ? 'Run a successful LLM connection test in Settings to continue.' : ''}
            </p>
          ) : null}

          <button
            className="btn btn-primary full-width"
            disabled={!canReview}
            onClick={handleValidateAndReview}
            type="button"
          >
            {validationPhase === 'running' ? <span className="spinner" /> : <Icons.checkCircle size={16} />}
            Validate & Review
          </button>
        </section>
      </div>

      {reviewStarted ? (
      <>
      <section className="card panel-section">
        <div className="section-head">
          <div>
            <p className="section-kicker">Validation</p>
            <h2>Checks</h2>
          </div>
          <StatusPill status={validationPhase === 'running' ? 'running' : validationPhase === 'done' ? 'passed' : validationPhase === 'error' ? 'failed' : 'queued'} spin={validationPhase === 'running'} />
        </div>
        {sessionSummary ? (
          <div className="summary-strip">
            <span>{sessionSummary.row_count ?? '-'} rows</span>
            <span>{sessionSummary.column_count ?? '-'} columns</span>
            <span>{sessionSummary.file_size_bytes ?? '-'} bytes</span>
          </div>
        ) : null}
        <div className="check-grid">
          {VALIDATION_KEYS.filter((checkKey) => (
            // metadata_match only applies when a metadata file is supplied.
            checkKey !== 'metadata_match' || metadataFile || validationByKey.metadata_match
          )).map((checkKey) => {
            const event = validationByKey[checkKey];
            const status = event?.status || 'queued';
            return (
              <div className={`check-card ${status}`} key={checkKey}>
                <StatusPill status={status === 'pass' ? 'passed' : status === 'fail' ? 'failed' : status === 'warn' ? 'warn' : 'queued'} />
                <strong>{event?.label || checkKey}</strong>
                <span>{event?.warn_message || event?.detail || 'Waiting for validation.'}</span>
              </div>
            );
          })}
        </div>
      </section>

      <section className="card panel-section">
        <div className="section-head">
          <div>
            <p className="section-kicker">Metadata</p>
            <h2>Generation</h2>
          </div>
          <StatusPill status={metadataPhase === 'running' ? 'running' : metadataPhase === 'done' ? 'passed' : metadataPhase === 'error' ? 'failed' : 'queued'} spin={metadataPhase === 'running'} />
        </div>
        <MetadataProgress
          phase={metadataPhase}
          events={metadataEvents}
          llm={llmSettings}
          errorMessage={error}
        />
        <button
          className="btn btn-primary"
          disabled={!canRunPipeline || trainingPhase === 'running'}
          onClick={handleStartTraining}
          type="button"
        >
          {trainingPhase === 'running' ? <span className="spinner" /> : <Icons.play size={16} />}
          {trainingPhase === 'running' ? 'Starting training...' : 'Start Ray training'}
        </button>
      </section>
      </>
      ) : null}
    </div>
  );
}

export default UploadScreen;
