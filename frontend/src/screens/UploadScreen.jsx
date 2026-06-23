import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import FormField from '../components/FormField.jsx';
import MetadataProgress from '../components/MetadataProgress.jsx';
import RunConfigurationPanel from '../components/RunConfigurationPanel.jsx';
import Segmented from '../components/Segmented.jsx';
import StatusPill from '../components/StatusPill.jsx';
import Toast from '../components/Toast.jsx';
import {
  fetchPublicConfig,
  fetchRecentUploads,
  fetchRunMetadata,
  fetchRunProgress,
  startFeatureEngineering,
  startMetadata,
  startValidation,
  uploadDataset,
} from '../api/client.js';
import { streamMetadataEvents, streamValidationEvents } from '../api/events.js';
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

// What the user should actually do when a given check fails or warns, shown
// next to the check so "I don't know how to make the checks pass" has a
// concrete answer instead of just a status pill.
const CHECK_REMEDIATION = {
  format: 'Re-export the file as a clean CSV/XLS with a single header row.',
  rows: 'Upload a dataset with more rows, or relax the minimum row requirement.',
  nulls: 'Fill or drop columns with excessive missing values before re-uploading.',
  variance: 'Remove or merge constant/near-constant columns and re-upload.',
  pii: 'Strip personally identifiable columns (names, emails, IDs) and re-upload.',
  target: 'Set Target column above to a column that exists in the dataset.',
  metadata_match: 'Make sure the optional metadata file lists the same columns as the dataset.',
};

const PROBLEM_OPTIONS = [
  { value: 'auto', label: 'Auto' },
  { value: 'classification', label: 'Classify' },
  { value: 'regression', label: 'Regress' },
  { value: 'unsupervised', label: 'Cluster' },
];

// Ordered pipeline phases shown in the resume panel. Keys match the backend
// /api/runs/{id}/progress response (config.ini [pipeline_phases]).
const PHASE_LABELS = [
  ['validation', 'Validation'],
  ['metadata', 'Metadata generation'],
  ['feature_engineering', 'Feature engineering'],
  ['training', 'Training'],
  ['evaluation', 'Evaluation'],
];
const PHASE_LABEL_MAP = Object.fromEntries(PHASE_LABELS);

function UploadScreen({ go, startRun, enterFeatureEngineering, resumeSession, route, incomingDataset, onIncomingDatasetConsumed, llmSettings, llmSmokeStatus, setLlmSettings, setLlmSmokeStatus }) {
  const [publicConfig, setPublicConfig] = useState(null);
  const reviewSectionRef = useRef(null);
  const [recentUploads, setRecentUploads] = useState([]);
  const [datasetFile, setDatasetFile] = useState(null);
  const [metadataFile, setMetadataFile] = useState(null);
  const [selectedRecent, setSelectedRecent] = useState(null);
  // Per-phase completion for a selected existing session, so completed phases
  // are skipped on resume and the user can continue from the last checkpoint.
  const [sessionProgress, setSessionProgress] = useState(null);
  const [sessionSummary, setSessionSummary] = useState(null);
  const [activeSessionId, setActiveSessionId] = useState('');
  const [validationEvents, setValidationEvents] = useState([]);
  const [metadataEvents, setMetadataEvents] = useState([]);
  const [validationPhase, setValidationPhase] = useState('idle');
  const [metadataPhase, setMetadataPhase] = useState('idle');
  const [featurePhase, setFeaturePhase] = useState('idle');
  const [error, setError] = useState(null);
  const [form, setForm] = useState({
    problemType: 'auto',
    targetCol: '',
    validationSplit: 0.8,
    description: '',
  });

  // Reusable so the recent-uploads list can refresh on navigation and after a
  // new upload, not only once on mount (the screen stays mounted across routes).
  const loadRecentUploads = useCallback(async () => {
    try {
      const recentPayload = await fetchRecentUploads(5);
      setRecentUploads(recentPayload.uploads || []);
    } catch {
      // Transient backend hiccup (e.g. restart): keep the last known list.
    }
  }, []);

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

  // Refresh the recent-uploads list whenever the Upload screen becomes active so
  // newly uploaded datasets show up without a full page reload.
  useEffect(() => {
    if (route === 'upload') {
      loadRecentUploads();
    }
  }, [route, loadRecentUploads]);

  const acceptedExtensions = publicConfig?.upload?.allowed_extensions || ['.csv', '.xls', '.xlsx'];
  const validationByKey = useMemo(
    () => Object.fromEntries(
      validationEvents
        .filter((event) => event.type === 'check')
        .map((event) => [event.key, event])
    ),
    [validationEvents],
  );
  // Feature engineering requires real metadata.json, so the run can only
  // continue once validation passed AND metadata generation succeeded. If
  // metadata failed the pipeline hard-fails here (no fallback artifacts).
  const metadataFailed = validationPhase === 'done' && metadataPhase === 'error';
  const canContinueToFeatureEngineering = validationPhase === 'done' && metadataPhase === 'done';
  const reviewStarted = validationPhase !== 'idle';
  // Mandatory gates for Validate & Review: a dataset is selected and the
  // current LLM configuration passed a connection test in Settings.
  const hasDataset = Boolean(datasetFile || selectedRecent);
  const llmVerified = (
    llmSmokeStatus.status === 'passed'
    && llmSmokeStatus.configKey === llmConfigKey(llmSettings)
  );
  const busy = validationPhase === 'running' || metadataPhase === 'running' || featurePhase === 'running';
  const canReview = hasDataset && llmVerified && !busy;

  // Auto-scroll to the checks/metadata section as soon as a review starts, so
  // the user is never left wondering whether they need to scroll down.
  useEffect(() => {
    if (reviewStarted && reviewSectionRef.current) {
      reviewSectionRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [reviewStarted]);

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
    setFeaturePhase('idle');
    setSessionSummary(null);
    setActiveSessionId('');
    setError(null);
  }

  function handleDatasetChange(file) {
    setDatasetFile(file);
    setSelectedRecent(null);
    setSessionProgress(null);
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
    // Fetch per-phase progress so completed phases can be skipped on resume.
    setSessionProgress(null);
    fetchRunProgress(uploadRecord.session_id)
      .then((progress) => setSessionProgress(progress))
      .catch(() => setSessionProgress(null));
  }

  // Reopen an existing dataset (from the dashboard): select it, load its phase
  // progress, and pre-fill the run form (target column, problem type) from the
  // session's stored metadata so every stage reflects that session.
  function selectExistingSession(record) {
    handleRecentSelect(record);
    fetchRunMetadata(record.session_id)
      .then((metadata) => {
        const target = metadata?.target_col || metadata?.target_column || '';
        const problemType = metadata?.problem_type;
        const subtype = metadata?.problem_subtype;
        setForm((currentForm) => ({
          ...currentForm,
          targetCol: target || currentForm.targetCol,
          problemType: ['classification', 'regression', 'unsupervised'].includes(problemType)
            ? problemType
            : (problemType === 'supervised' && ['classification', 'regression'].includes(subtype)
              ? subtype
              : currentForm.problemType),
        }));
      })
      .catch(() => {});
  }

  // Consume a dataset opened from the dashboard exactly once.
  useEffect(() => {
    if (!incomingDataset) {
      return;
    }
    selectExistingSession(incomingDataset);
    onIncomingDatasetConsumed?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [incomingDataset]);

  // Resume: route to the screen that owns the first incomplete phase, starting
  // it where this screen owns the handoff. Completed phases are never re-run.
  function handleResumeSession() {
    const sessionId = String(activeSessionId || '').trim();
    if (!sessionId) {
      setError('Select a recent upload to resume.');
      return;
    }
    const nextPhase = sessionProgress?.next_phase ?? null;
    if (nextPhase === 'validation' || nextPhase === 'metadata') {
      // Earliest phases are still missing; run the standard (skip-aware) flow.
      handleValidateAndReview();
      return;
    }
    if (nextPhase === 'feature_engineering') {
      handleContinueToFeatureEngineering();
      return;
    }
    if (nextPhase === 'training') {
      // FE is complete; land on the FE page where "Continue to Training" lives.
      enterFeatureEngineering(sessionId);
      return;
    }
    // evaluation pending or everything complete -> jump to the leaderboard.
    resumeSession(sessionId, nextPhase);
  }

  async function handleRerunMetadata() {
    const sessionId = String(activeSessionId || '').trim();
    if (!sessionId) {
      return;
    }
    await runMetadata(sessionId, { force: true });
    fetchRunProgress(sessionId)
      .then((progress) => setSessionProgress(progress))
      .catch(() => {});
  }

  function handleRerunFeatureEngineering() {
    handleContinueToFeatureEngineering({ force: true });
  }

  async function handleValidateAndReview() {
    // Surface a concrete, visible error instead of silently no-op'ing when a
    // mandatory precondition (dataset selected, LLM connection verified) is
    // missing - a disabled button alone left users clueless about why.
    const missingReasons = [];
    if (!hasDataset) {
      missingReasons.push('Select a dataset file or a recent upload above.');
    }
    if (!llmVerified) {
      missingReasons.push('Click "Test connection" in Run Configuration above and wait for it to pass.');
    }
    if (missingReasons.length) {
      setError(`Cannot validate yet: ${missingReasons.join(' ')}`);
      return;
    }

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
        // Surface the just-uploaded dataset in the recent list immediately.
        loadRecentUploads();
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
      // Use functional update to read current state rather than the stale closure
      // value — validationPhase in this closure still reflects the pre-setValidationPhase
      // snapshot, so a direct comparison would always miss the 'running' transition.
      setValidationPhase((currentPhase) => currentPhase === 'running' ? 'error' : currentPhase);
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

  async function runMetadata(sessionId, { force = false } = {}) {
    setMetadataPhase('running');

    try {
      const response = await startMetadata({
        sessionId,
        description: form.description,
        targetCol: form.problemType === 'unsupervised' ? null : form.targetCol,
        problemType: form.problemType === 'auto' ? null : form.problemType,
        provider: llmSettings.provider,
        model: llmSettings.model,
        apiKey: llmSettings.apiKey || '',
        gatewayUrl: llmSettings.gatewayUrl,
        force,
      });

      // Cached metadata.json reused: the backend skipped the agent and no SSE
      // events will arrive, so mark done without waiting on the stream.
      if (response?.status === 'skipped') {
        setMetadataPhase('done');
        return;
      }

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

  // Manual gate: kick off feature engineering (PipelinePrep) and move to the
  // Feature Engineering tab. Training is started later from that tab, only
  // after FE + model selection succeed. This replaces the old flow that jumped
  // straight to training with deterministic fallback artifacts.
  async function handleContinueToFeatureEngineering({ force = false } = {}) {
    const sessionId = String(activeSessionId || '').trim();
    if (!sessionId) {
      setError('No active session is available for feature engineering.');
      return;
    }

    setError(null);
    setFeaturePhase('running');
    try {
      await startFeatureEngineering({
        sessionId,
        targetCol: form.problemType === 'unsupervised' ? null : form.targetCol,
        problemType: form.problemType === 'auto' ? null : form.problemType,
        provider: llmSettings.provider,
        model: llmSettings.model,
        apiKey: llmSettings.apiKey || '',
        gatewayUrl: llmSettings.gatewayUrl,
        force,
      });
      setFeaturePhase('accepted');
      enterFeatureEngineering(sessionId);
    } catch (featureError) {
      setFeaturePhase('error');
      setError(featureError.message);
    }
  }

  return (
    <div className="screen-stack">
      <Toast message={error} onDismiss={() => setError(null)} tone="error" />

      <RunConfigurationPanel
        llmSettings={llmSettings}
        llmSmokeStatus={llmSmokeStatus}
        publicConfig={publicConfig}
        setLlmSettings={setLlmSettings}
        setLlmSmokeStatus={setLlmSmokeStatus}
      />

      <div className="upload-grid">
        <section className="card panel-section">
          <div className="section-head">
            <div>
              <p className="section-kicker">Step 2 · Dataset</p>
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

          {selectedRecent && sessionProgress ? (
            <div className="recent-list" style={{ marginTop: 16 }}>
              <p className="section-kicker">Session progress (resume from checkpoint)</p>
              {PHASE_LABELS.map(([phaseKey, phaseLabel]) => {
                const phaseStatus = sessionProgress.phases?.[phaseKey] || 'pending';
                const isDone = phaseStatus === 'complete' || phaseStatus === 'passed';
                return (
                  <div
                    className="recent-row"
                    key={phaseKey}
                    style={{ cursor: 'default' }}
                  >
                    <Icons.doc size={16} />
                    <span><strong>{phaseLabel}</strong></span>
                    <StatusPill
                      status={isDone ? 'passed' : (phaseStatus === 'failed' ? 'failed' : 'queued')}
                      label={phaseStatus}
                    />
                  </div>
                );
              })}
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 12 }}>
                <button
                  className="btn btn-primary"
                  disabled={busy}
                  onClick={handleResumeSession}
                  type="button"
                >
                  {sessionProgress.next_phase
                    ? `Continue (${PHASE_LABEL_MAP[sessionProgress.next_phase] || sessionProgress.next_phase})`
                    : 'View leaderboard'}
                </button>
                {(sessionProgress.phases?.metadata === 'complete') ? (
                  <button
                    className="btn btn-secondary"
                    disabled={busy}
                    onClick={handleRerunMetadata}
                    type="button"
                  >
                    Re-run metadata
                  </button>
                ) : null}
                {(sessionProgress.phases?.feature_engineering === 'complete') ? (
                  <button
                    className="btn btn-secondary"
                    disabled={busy}
                    onClick={handleRerunFeatureEngineering}
                    type="button"
                  >
                    Re-run feature engineering
                  </button>
                ) : null}
              </div>
            </div>
          ) : null}
        </section>

        <section className="card panel-section">
          <div className="section-head">
            <div>
              <p className="section-kicker">Step 3 · Run Metadata</p>
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

          <button
            className="btn btn-primary full-width"
            disabled={busy}
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
      <section className="card panel-section" ref={reviewSectionRef}>
        <div className="section-head">
          <div>
            <p className="section-kicker">Validation</p>
            <h2>Checks</h2>
          </div>
          <StatusPill status={validationPhase === 'running' ? 'running' : validationPhase === 'done' ? 'passed' : validationPhase === 'error' ? 'failed' : 'queued'} spin={validationPhase === 'running'} />
        </div>
        {validationPhase === 'running' ? (
          <div className="progress-bar indeterminate">
            <span />
          </div>
        ) : null}
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
            const needsFix = status === 'fail' || status === 'warn';
            return (
              <div className={`check-card ${status}`} key={checkKey}>
                <StatusPill status={status === 'pass' ? 'passed' : status === 'fail' ? 'failed' : status === 'warn' ? 'warn' : 'queued'} />
                <strong>{event?.label || checkKey}</strong>
                <span>{event?.warn_message || event?.detail || 'Waiting for validation.'}</span>
                {needsFix ? (
                  <span className="check-hint">Fix: {CHECK_REMEDIATION[checkKey] || 'Adjust the dataset and re-upload.'}</span>
                ) : null}
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
        {metadataFailed ? (
          <div className="callout error compact">
            Metadata generation failed. Feature engineering needs valid metadata, so the run cannot continue. Fix the inputs (or LLM settings) and re-run metadata.
          </div>
        ) : null}
        <button
          className="btn btn-primary"
          disabled={!canContinueToFeatureEngineering || featurePhase === 'running'}
          onClick={handleContinueToFeatureEngineering}
          type="button"
        >
          {featurePhase === 'running' ? <span className="spinner" /> : <Icons.play size={16} />}
          {featurePhase === 'running' ? 'Starting feature engineering...' : 'Continue to Feature Engineering'}
        </button>
      </section>
      </>
      ) : null}
    </div>
  );
}

export default UploadScreen;
