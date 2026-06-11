/* ============================================================
   MITRA AI — app shell, routing, run-simulation engine, tweaks
   ============================================================ */

// flatten stage logs into a single timeline of events
const FLAT = [];
STAGES.forEach((s, si) => {
  const lines = STAGE_LOGS[s.key] || [];
  lines.forEach((ln, li) => {
    FLAT.push({ stageIndex: si, stageKey: s.key, level: ln[0], msg: ln[1],
      lineInStage: li, stageLineCount: lines.length });
  });
});
function nowTs() {
  const d = new Date();
  return `${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}:${String(d.getSeconds()).padStart(2,'0')}`;
}

const ACCENTS = {
  '#6c47ff': { accent:'#6c47ff', strong:'#5a37e0', soft:'#efeaff', line:'#ddd4ff', ink:'#4a2fc4' },
  '#3b6fff': { accent:'#3b6fff', strong:'#2f59e6', soft:'#e9efff', line:'#cfdcff', ink:'#1f47cc' },
  '#12a06a': { accent:'#12a06a', strong:'#0d8559', soft:'#e4f6ee', line:'#c2ebd6', ink:'#0a6e49' },
  '#e08a2b': { accent:'#e08a2b', strong:'#c5760f', soft:'#fbf0df', line:'#f3ddb3', ink:'#9c5d0c' },
};

const ROUTE_META = {
  dashboard:   { title:'Dashboard',    sub:'Runs, agents, and system health at a glance', icon:'grid' },
  upload:      { title:'New Run',      sub:'Upload a dataset and let the agents take over', icon:'upload' },
  pipeline:    { title:'Live Pipeline',sub:'Eight specialist agents · real-time deliberation', icon:'flow' },
  leaderboard: { title:'Leaderboard',  sub:"Ranked models with the Judge's justification", icon:'trophy' },
};

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accent": "#6c47ff",
  "speed": 1.8,
  "corners": "soft",
  "glow": true
}/*EDITMODE-END*/;

function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [route, setRoute] = useState('dashboard');

  // ---- run state ----
  const [runState, setRunState] = useState('idle'); // idle | running | done
  const [cursor, setCursor] = useState(0);           // events revealed
  const [elapsed, setElapsed] = useState(0);
  const speedRef = useRef(t.speed);
  speedRef.current = t.speed;

  function startRun() {
    setCursor(0); setElapsed(0); setRunState('running'); setRoute('pipeline');
  }
  function go(r) { setRoute(r); }

  // reveal engine
  useEffect(() => {
    if (runState !== 'running') return;
    if (cursor >= FLAT.length) { setRunState('done'); return; }
    const interval = Math.max(140, Math.round(540 / (speedRef.current || 1)));
    const id = setTimeout(() => setCursor(c => c + 1), cursor === 0 ? 250 : interval);
    return () => clearTimeout(id);
  }, [runState, cursor]);

  // elapsed timer
  useEffect(() => {
    if (runState !== 'running') return;
    const id = setInterval(() => setElapsed(e => e + 1), 1000);
    return () => clearInterval(id);
  }, [runState]);

  // apply tweaks → CSS vars
  useEffect(() => {
    const root = document.documentElement.style;
    const a = ACCENTS[t.accent] || ACCENTS['#6c47ff'];
    root.setProperty('--accent', a.accent);
    root.setProperty('--accent-strong', a.strong);
    root.setProperty('--accent-soft', a.soft);
    root.setProperty('--accent-line', a.line);
    root.setProperty('--accent-ink', a.ink);
    const r = t.corners === 'sharp' ? ['8px','6px','4px','3px'] : ['20px','16px','11px','8px'];
    root.setProperty('--r-xl', r[0]); root.setProperty('--r-card', r[1]);
    root.setProperty('--r-md', r[2]); root.setProperty('--r-sm', r[3]);
    root.setProperty('--bg-grad', t.glow
      ? 'radial-gradient(1200px 600px at 80% -10%, color-mix(in oklab, var(--accent) 12%, #fff) 0%, rgba(255,255,255,0) 55%), #f4f5f8'
      : '#f4f5f8');
  }, [t.accent, t.corners, t.glow]);

  // derive run view-model
  const cur = cursor > 0 ? FLAT[cursor - 1] : null;
  const stageIndex = runState === 'done' ? STAGES.length - 1 : (cur ? cur.stageIndex : 0);
  const stageProgress = cur ? (cur.lineInStage + 1) / cur.stageLineCount : 0;
  const logs = FLAT.slice(0, cursor).map(e => ({ level: e.level, msg: e.msg, ts: e._ts || (e._ts = nowTs()) }));
  const verdict = FLAT.slice(0, cursor).some(e => e.stageKey === 'judge' && e.level === 'ok') || runState === 'done';
  const run = { state: runState, stageIndex, stageProgress, logs, elapsed, verdict };

  const meta = ROUTE_META[route];
  const RightActions = (
    <>
      {runState === 'running' && route !== 'pipeline' && (
        <button className="btn btn-ghost btn-sm" onClick={() => go('pipeline')}>
          <span className="spinner" style={{ width: 11, height: 11 }} />Run in progress
        </button>
      )}
      <button className="btn btn-ghost btn-sm" style={{ padding: '0 9px' }} title="Notifications"><Icons.bell size={18} /></button>
      {route !== 'upload' && <button className="btn btn-primary btn-sm" onClick={() => go('upload')}><Icons.plus size={15} />New run</button>}
    </>
  );

  return (
    <div className="app">
      <Sidebar route={route} go={go} runState={runState} />
      <main style={{ display: 'flex', flexDirection: 'column', height: '100%', overflowY: 'auto' }}>
        <TopBar title={meta.title} sub={meta.sub} icon={meta.icon} right={RightActions} />
        <div style={{ flex: 1 }}>
          {route === 'dashboard'   && <Dashboard go={go} startRun={startRun} />}
          {route === 'upload'      && <UploadScreen go={go} startRun={startRun} />}
          {route === 'pipeline'    && <PipelineScreen go={go} run={run} startRun={startRun} />}
          {route === 'leaderboard' && <LeaderboardScreen go={go} startRun={startRun} />}
        </div>
      </main>

      <TweaksPanel>
        <TweakSection label="Brand" />
        <TweakColor label="Accent" value={t.accent}
          options={['#6c47ff', '#3b6fff', '#12a06a', '#e08a2b']}
          onChange={(v) => setTweak('accent', v)} />
        <TweakRadio label="Corners" value={t.corners} options={['soft', 'sharp']}
          onChange={(v) => setTweak('corners', v)} />
        <TweakToggle label="Accent glow" value={t.glow} onChange={(v) => setTweak('glow', v)} />
        <TweakSection label="Run simulation" />
        <TweakSlider label="Agent speed" value={t.speed} min={0.5} max={2.5} step={0.1} unit="×"
          onChange={(v) => setTweak('speed', v)} />
        <TweakButton label="Restart run" onClick={startRun} />
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
