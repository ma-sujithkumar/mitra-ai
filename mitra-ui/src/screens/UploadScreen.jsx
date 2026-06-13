import React, { useState, useEffect, useRef } from 'react';
import { Icons } from '../icons.jsx';
import { SAMPLE_DATASETS, AGENT_MAP } from '../data.js';
import { AgentAvatar } from '../components/AgentAvatar.jsx';
import { StatusPill } from '../components/StatusPill.jsx';
import { Segmented } from '../components/Segmented.jsx';
import { FormField, FIELD_INPUT_STYLE } from '../components/FormField.jsx';
import { ByomFields } from '../components/ByomFields.jsx';
import { uploadFile, streamValidation } from '../api.js';

const MAX_FILE_SIZE_MB = 200;
const ALLOWED_EXTENSIONS = ['.csv', '.xls', '.xlsx', '.zip'];

const VALIDATION_CHECK_LABELS = {
  format:   'File format & encoding',
  rows:     'Row count',
  nulls:    'Null density',
  variance: 'Zero-variance scan',
  pii:      'PII heuristic',
  target:   'Target separability',
};

const CHECK_ORDER = ['format', 'rows', 'nulls', 'variance', 'pii', 'target'];

function ValidationCheck({ checkData, shown }) {
  const isWarn = checkData.status === 'warn';
  const isFail = checkData.status === 'fail';
  return (
    <div className={shown ? 'fade-up' : ''} style={{
      display: 'flex', alignItems: 'flex-start', gap: 11, padding: '14px 22px',
      borderBottom: '1px solid var(--line-2)',
      opacity: shown ? 1 : 0.35,
    }}>
      <div style={{
        flex: 'none', marginTop: 1,
        color: !shown ? 'var(--ink-4)' : isFail ? 'var(--err)' : isWarn ? 'var(--warn)' : 'var(--ok)',
      }}>
        {!shown
          ? <Icons.dot size={16} />
          : isFail
          ? <Icons.x size={17} />
          : isWarn
          ? <Icons.alert size={17} />
          : <Icons.checkCircle size={18} />}
      </div>
      <div className="col" style={{ lineHeight: 1.4 }}>
        <span style={{ fontSize: 13, fontWeight: 600 }}>{checkData.label || VALIDATION_CHECK_LABELS[checkData.key] || checkData.key}</span>
        <span className="faint" style={{ fontSize: 12 }}>
          {shown
            ? (isWarn ? checkData.warn_message || checkData.detail : checkData.detail)
            : 'queued...'}
        </span>
      </div>
    </div>
  );
}

export function UploadScreen({ go }) {
  const [stagedFile, setStagedFile] = useState(null);
  const [selectedFixture, setSelectedFixture] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploadError, setUploadError] = useState('');

  const [problemType, setProblemType] = useState('auto');
  const [targetCol, setTargetCol] = useState('');
  const [description, setDescription] = useState('');
  const [dataType, setDataType] = useState('csv');
  const [llm, setLlm] = useState({ provider: 'anthropic', apiKey: '', gateway: '' });

  const [phase, setPhase] = useState('idle'); // idle | uploading | validating | done | error
  const [sessionId, setSessionId] = useState('');
  const [checks, setChecks] = useState({});  // key -> check event data
  const [revealedKeys, setRevealedKeys] = useState([]);

  const fileInputRef = useRef(null);

  function validateFileFrontend(file) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      return `File type '${ext}' not allowed. Accepted: ${ALLOWED_EXTENSIONS.join(', ')}`;
    }
    const sizeMb = file.size / (1024 * 1024);
    if (sizeMb > MAX_FILE_SIZE_MB) {
      return `File too large (${sizeMb.toFixed(1)} MB). Max: ${MAX_FILE_SIZE_MB} MB`;
    }
    return '';
  }

  function handleFileDrop(evt) {
    evt.preventDefault();
    setDragOver(false);
    const file = evt.dataTransfer.files[0];
    if (!file) return;
    const errorMsg = validateFileFrontend(file);
    if (errorMsg) { setUploadError(errorMsg); return; }
    setStagedFile(file);
    setSelectedFixture(null);
    setUploadError('');
    resetValidation();
  }

  function handleFileInput(evt) {
    const file = evt.target.files[0];
    if (!file) return;
    const errorMsg = validateFileFrontend(file);
    if (errorMsg) { setUploadError(errorMsg); return; }
    setStagedFile(file);
    setSelectedFixture(null);
    setUploadError('');
    resetValidation();
  }

  function handleFixtureSelect(fixture) {
    setSelectedFixture(fixture);
    setStagedFile(null);
    setUploadError('');
    resetValidation();
    if (fixture.task === 'Classification') {
      setProblemType('classification');
      setTargetCol('species');
      setDescription('Fixture dataset: ' + fixture.name);
    } else if (fixture.task === 'Regression') {
      setProblemType('regression');
      setTargetCol('median_house_value');
      setDescription('Fixture dataset: ' + fixture.name);
    }
  }

  function resetValidation() {
    setPhase('idle');
    setSessionId('');
    setChecks({});
    setRevealedKeys([]);
  }

  async function handleValidate() {
    if (!stagedFile && !selectedFixture) return;
    if (phase === 'uploading' || phase === 'validating') return;

    setUploadError('');
    setChecks({});
    setRevealedKeys([]);
    setPhase('uploading');

    try {
      // For fixture datasets, we need to fetch the file. For real uploads, use stagedFile.
      let fileToUpload = stagedFile;
      if (selectedFixture && !stagedFile) {
        // Fixture mode: show a placeholder validation with local mock since fixture files
        // aren't bundled. In production, fixtures would be served from backend.
        setPhase('validating');
        runMockValidation();
        return;
      }

      const uploadResult = await uploadFile(fileToUpload, {
        target_col: targetCol,
        problem_type: problemType,
        description,
      });

      setSessionId(uploadResult.session_id);
      setPhase('validating');

      streamValidation(
        uploadResult.session_id,
        targetCol,
        (checkEvent) => {
          setChecks(prev => ({ ...prev, [checkEvent.key]: checkEvent }));
          setRevealedKeys(prev => [...prev, checkEvent.key]);
        },
        () => setPhase('done'),
        (err) => { setPhase('error'); setUploadError(err.message); },
      );
    } catch (err) {
      setPhase('error');
      setUploadError(err.message || 'Upload failed');
    }
  }

  // Mock validation for fixture selection (when fixture file isn't actually uploaded)
  function runMockValidation() {
    const mockChecks = [
      { key: 'format',   status: 'pass', detail: 'utf-8, comma-delimited, 5 columns' },
      { key: 'rows',     status: 'pass', detail: '150 rows, above minimum (10)' },
      { key: 'nulls',    status: 'pass', detail: '0 columns exceed 80% threshold' },
      { key: 'variance', status: 'pass', detail: 'No constant columns detected' },
      { key: 'pii',      status: 'pass', detail: 'No PII-suspect column names' },
      { key: 'target',   status: 'warn', detail: 'species - 3 balanced classes', warn_message: 'Mild class overlap on sepal width' },
    ];
    mockChecks.forEach((check, index) => {
      setTimeout(() => {
        setChecks(prev => ({ ...prev, [check.key]: check }));
        setRevealedKeys(prev => [...prev, check.key]);
        if (index === mockChecks.length - 1) {
          setTimeout(() => setPhase('done'), 340);
        }
      }, index * 340);
    });
  }

  const validationPhase = phase === 'validating' || phase === 'done';
  const allChecksDone = phase === 'done';
  const hasBlocker = Object.values(checks).some(check => check.status === 'fail');
  const canRunPipeline = allChecksDone && !hasBlocker;

  const dataTypeSummary = validationPhase && revealedKeys.length > 0
    ? `${revealedKeys.filter(k => checks[k]?.status === 'pass').length} passed · ${revealedKeys.filter(k => checks[k]?.status === 'warn').length} warning · ${revealedKeys.filter(k => checks[k]?.status === 'fail').length} fail`
    : 'Running checks against the data profile...';

  return (
    <div className="page page-in">
      <div style={{ display: 'grid', gridTemplateColumns: '1.35fr 1fr', gap: 18 }}>

        {/* left: dropzone + fixture picker */}
        <div className="col gap-18">
          <div className="card" style={{ padding: 24 }}>
            {/* dropzone */}
            <div
              onDragOver={evt => { evt.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleFileDrop}
              style={{
                border: `2px dashed ${dragOver ? 'var(--accent)' : 'var(--accent-line)'}`,
                borderRadius: 14, padding: '30px 24px',
                background: dragOver ? 'var(--accent-soft)' : 'linear-gradient(180deg,#fbfaff,#fff)',
                textAlign: 'center', display: 'flex', flexDirection: 'column',
                alignItems: 'center', gap: 10, transition: 'all .15s',
              }}
            >
              <div style={{
                width: 50, height: 50, borderRadius: 14, background: 'var(--accent-soft)',
                display: 'grid', placeItems: 'center', color: 'var(--accent)',
              }}>
                <Icons.upload size={24} />
              </div>
              <div className="col gap-2">
                {stagedFile ? (
                  <>
                    <span style={{ fontWeight: 700, fontSize: 15, color: 'var(--accent)' }}>{stagedFile.name}</span>
                    <span className="faint" style={{ fontSize: 12 }}>
                      {(stagedFile.size / (1024 * 1024)).toFixed(2)} MB
                    </span>
                  </>
                ) : (
                  <>
                    <span style={{ fontWeight: 700, fontSize: 15 }}>Drop a dataset to begin</span>
                    <span className="faint" style={{ fontSize: 12.5 }}>
                      CSV or image .zip · up to 200 MB · processed locally
                    </span>
                  </>
                )}
              </div>
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.xls,.xlsx,.zip"
                style={{ display: 'none' }}
                onChange={handleFileInput}
              />
              <button
                className="btn btn-secondary btn-sm"
                style={{ marginTop: 4 }}
                onClick={() => fileInputRef.current?.click()}
              >
                Browse files
              </button>
              {uploadError && (
                <span style={{ fontSize: 12, color: 'var(--err)', fontWeight: 600 }}>{uploadError}</span>
              )}
            </div>

            {/* fixture picker */}
            <div className="mono faint" style={{ fontSize: 10.5, letterSpacing: '.06em', margin: '20px 0 10px' }}>
              OR PICK A FIXTURE
            </div>
            <div className="col gap-8">
              {SAMPLE_DATASETS.map(dataset => {
                const isSelected = selectedFixture?.name === dataset.name;
                return (
                  <button
                    key={dataset.name}
                    onClick={() => handleFixtureSelect(dataset)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 12, padding: '11px 14px', textAlign: 'left',
                      border: `1px solid ${isSelected ? 'var(--accent)' : 'var(--line-3)'}`,
                      borderRadius: 11, cursor: 'pointer',
                      background: isSelected ? 'var(--accent-soft)' : '#fff', transition: 'all .14s',
                    }}
                  >
                    <div style={{
                      width: 32, height: 32, borderRadius: 8,
                      background: isSelected ? '#fff' : 'var(--panel-3)',
                      display: 'grid', placeItems: 'center',
                      color: isSelected ? 'var(--accent)' : 'var(--ink-3)', flex: 'none',
                    }}>
                      <Icons.doc size={17} />
                    </div>
                    <div className="col" style={{ lineHeight: 1.3 }}>
                      <span className="mono" style={{ fontSize: 13, fontWeight: 600 }}>{dataset.name}</span>
                      <span className="faint" style={{ fontSize: 11.5 }}>{dataset.rows} rows · {dataset.cols} cols · {dataset.size}</span>
                    </div>
                    <span className="tag" style={{ marginLeft: 'auto' }}>{dataset.task}</span>
                    {isSelected && <Icons.checkCircle size={18} style={{ color: 'var(--accent)', flex: 'none' }} />}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* right: metadata form */}
        <div className="card" style={{ padding: 24, alignSelf: 'flex-start' }}>
          <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 3 }}>Metadata</h3>
          <p className="faint" style={{ fontSize: 12, margin: '0 0 18px' }}>
            Minimal hints - agents infer the rest into <span className="mono">metadata.json</span>
          </p>

          <FormField label="Problem type">
            <Segmented
              value={problemType}
              onChange={setProblemType}
              options={[
                { value: 'auto', label: 'Auto-detect' },
                { value: 'classification', label: 'Classify' },
                { value: 'regression', label: 'Regress' },
                { value: 'unsupervised', label: 'Cluster' },
              ]}
            />
          </FormField>

          <FormField label="Target column" hint="leave blank for unsupervised">
            <input
              className="focusable"
              value={targetCol}
              onChange={evt => setTargetCol(evt.target.value)}
              style={FIELD_INPUT_STYLE}
              placeholder="e.g. species, price, label"
            />
          </FormField>

          <FormField label="Description" hint=">= 20 chars - guides feature and model agents">
            <textarea
              className="focusable"
              value={description}
              onChange={evt => setDescription(evt.target.value)}
              rows={4}
              style={{ ...FIELD_INPUT_STYLE, resize: 'vertical', lineHeight: 1.5 }}
              placeholder="Describe your dataset and task..."
            />
          </FormField>

          <FormField label="Data type">
            <Segmented
              value={dataType}
              onChange={setDataType}
              options={[
                { value: 'csv', label: 'CSV' },
                { value: 'excel', label: 'Excel' },
                { value: 'image', label: 'Image' },
              ]}
            />
          </FormField>

          <ByomFields llm={llm} setLlm={setLlm} />

          <button
            className={`btn btn-primary ${(!stagedFile && !selectedFixture) ? '' : 'cta-pulse'}`}
            style={{ width: '100%', marginTop: 18, justifyContent: 'center', padding: '12px 16px', fontSize: 14, fontWeight: 600 }}
            onClick={handleValidate}
            disabled={(!stagedFile && !selectedFixture) || phase === 'uploading' || phase === 'validating'}
          >
            {phase === 'uploading'
              ? <><span className="spinner" />Uploading...</>
              : phase === 'validating'
              ? <><span className="spinner" />Validating your data...</>
              : <><Icons.checkCircle size={18} />Validate &amp; Review</>}
          </button>
        </div>
      </div>

      {/* validation report */}
      {phase !== 'idle' && (
        <div className="card fade-up" style={{ marginTop: 18, padding: 0, overflow: 'hidden' }}>
          <div className="row" style={{ justifyContent: 'space-between', padding: '15px 22px', borderBottom: '1px solid var(--line)' }}>
            <div className="row gap-10">
              <AgentAvatar agent={AGENT_MAP.validator} size={30} state={allChecksDone ? 'done' : 'running'} />
              <div className="col" style={{ lineHeight: 1.3 }}>
                <span style={{ fontSize: 14, fontWeight: 700 }}>Data Validator</span>
                <span className="mono faint" style={{ fontSize: 11 }}>validation_report.json</span>
              </div>
            </div>
            {allChecksDone
              ? <StatusPill status="passed" label="Passed - ready to run" />
              : <StatusPill status="running" label="Validating..." spin />}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0 }}>
            {CHECK_ORDER.map((checkKey, index) => {
              const isShown = revealedKeys.includes(checkKey);
              const checkData = checks[checkKey] || { key: checkKey, status: 'pending', detail: '' };
              return (
                <div
                  key={checkKey}
                  style={{
                    borderRight: index % 2 === 0 ? '1px solid var(--line-2)' : 'none',
                  }}
                >
                  <ValidationCheck checkData={checkData} shown={isShown} />
                </div>
              );
            })}
          </div>

          <div className="row" style={{ justifyContent: 'space-between', padding: '16px 22px', background: 'var(--panel-2)' }}>
            <span className="faint" style={{ fontSize: 12.5 }}>
              {allChecksDone
                ? `${Object.values(checks).filter(c => c.status === 'pass').length} passed · ${Object.values(checks).filter(c => c.status === 'warn').length} warning · no blockers — the pipeline is clear to launch.`
                : 'Running checks against the data profile...'}
            </span>
            <button
              className="btn btn-primary"
              disabled={!canRunPipeline}
              onClick={() => go('pipeline')}
            >
              <Icons.play size={15} />Run pipeline
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default UploadScreen;
