/* ============================================================
   MITRA AI — P1 · Upload & Validate
   ============================================================ */

const VALIDATION_CHECKS = [
  { key:'format',   label:'File format & encoding', detail:'utf-8 · comma-delimited · 5 columns', state:'pass' },
  { key:'rows',     label:'Row count',              detail:'150 rows · above minimum (10)',        state:'pass' },
  { key:'nulls',    label:'Null density',           detail:'0 columns exceed 80% threshold',       state:'pass' },
  { key:'variance', label:'Zero-variance scan',     detail:'No constant columns detected',         state:'pass' },
  { key:'pii',      label:'PII heuristic',          detail:'No PII-suspect column names',           state:'pass' },
  { key:'target',   label:'Target separability',    detail:'species · 3 balanced classes',         state:'warn', warn:'Mild class overlap on sepal width' },
];

function UploadScreen({ go, startRun }) {
  const [dataset, setDataset] = useState(SAMPLE_DATASETS[0]);
  const [task, setTask] = useState('auto');
  const [target, setTarget] = useState('species');
  const [desc, setDesc] = useState('Iris flower measurements — classify the species from sepal and petal dimensions. 150 labelled samples across 3 classes.');
  const [provider, setProvider] = useState('anthropic');
  const [phase, setPhase] = useState('idle'); // idle | validating | done
  const [revealed, setRevealed] = useState(0);

  function validate() {
    setPhase('validating'); setRevealed(0);
  }
  useEffect(()=>{
    if (phase!=='validating') return;
    if (revealed >= VALIDATION_CHECKS.length) { setPhase('done'); return; }
    const t = setTimeout(()=>setRevealed(r=>r+1), 340);
    return ()=>clearTimeout(t);
  }, [phase, revealed]);

  const passed = phase==='done';

  return (
    <div className="page page-in">
      {/* stepper */}
      <div className="row gap-8" style={{marginBottom:20}}>
        {[['Upload dataset', true],['Describe & configure', true],['Validate', passed]].map(([t,active],i)=>(
          <React.Fragment key={i}>
            <div className="row gap-8" style={{opacity: active?1:0.5}}>
              <div style={{width:24,height:24,borderRadius:99, display:'grid', placeItems:'center', flex:'none',
                background: active?'var(--accent)':'var(--panel-3)', color: active?'#fff':'var(--ink-3)',
                border: active?'none':'1px solid var(--line-3)', fontSize:12, fontWeight:700}} className="mono">{i+1}</div>
              <span style={{fontSize:13, fontWeight:600, color: active?'var(--ink)':'var(--ink-3)'}}>{t}</span>
            </div>
            {i<2 && <div style={{width:34,height:2,background:'var(--line-3)',borderRadius:2}}/>}
          </React.Fragment>
        ))}
      </div>

      <div style={{display:'grid', gridTemplateColumns:'1.35fr 1fr', gap:18}}>
        {/* left: dropzone + samples */}
        <div className="col gap-18">
          <div className="card" style={{padding:24}}>
            <div style={{
              border:'2px dashed var(--accent-line)', borderRadius:14, padding:'30px 24px',
              background:'linear-gradient(180deg,#fbfaff,#fff)', textAlign:'center',
              display:'flex', flexDirection:'column', alignItems:'center', gap:10,
            }}>
              <div style={{width:50,height:50,borderRadius:14,background:'var(--accent-soft)',display:'grid',placeItems:'center',color:'var(--accent)'}}>
                <Icons.upload size={24}/>
              </div>
              <div className="col gap-2">
                <span style={{fontWeight:700, fontSize:15}}>Drop a dataset to begin</span>
                <span className="faint" style={{fontSize:12.5}}>CSV or image .zip · up to 200 MB · processed locally</span>
              </div>
              <button className="btn btn-secondary btn-sm" style={{marginTop:4}}>Browse files</button>
            </div>

            <div className="mono faint" style={{fontSize:10.5, letterSpacing:'.06em', margin:'20px 0 10px'}}>OR PICK A FIXTURE</div>
            <div className="col gap-8">
              {SAMPLE_DATASETS.map(d=>{
                const sel = d.name===dataset.name;
                return (
                  <button key={d.name} onClick={()=>{setDataset(d); setPhase('idle');}} style={{
                    display:'flex', alignItems:'center', gap:12, padding:'11px 14px', textAlign:'left',
                    border:`1px solid ${sel?'var(--accent)':'var(--line-3)'}`, borderRadius:11, cursor:'pointer',
                    background: sel?'var(--accent-soft)':'#fff', transition:'all .14s',
                  }}>
                    <div style={{width:32,height:32,borderRadius:8,background: sel?'#fff':'var(--panel-3)',display:'grid',placeItems:'center',color: sel?'var(--accent)':'var(--ink-3)',flex:'none'}}>
                      <Icons.doc size={17}/>
                    </div>
                    <div className="col" style={{lineHeight:1.3}}>
                      <span className="mono" style={{fontSize:13, fontWeight:600}}>{d.name}</span>
                      <span className="faint" style={{fontSize:11.5}}>{d.rows} rows · {d.cols} cols · {d.size}</span>
                    </div>
                    <span className="tag" style={{marginLeft:'auto'}}>{d.task}</span>
                    {sel && <Icons.checkCircle size={18} style={{color:'var(--accent)', flex:'none'}}/>}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* right: metadata form */}
        <div className="card" style={{padding:24, alignSelf:'flex-start'}}>
          <h3 style={{fontSize:15, fontWeight:700, marginBottom:3}}>Metadata</h3>
          <p className="faint" style={{fontSize:12, margin:'0 0 18px'}}>Minimal hints — agents infer the rest into <span className="mono">metadata.json</span></p>

          <Field label="Problem type">
            <Segmented value={task} onChange={setTask} options={[
              {value:'auto',label:'Auto-detect'},{value:'classification',label:'Classify'},
              {value:'regression',label:'Regress'},{value:'unsupervised',label:'Cluster'},
            ]}/>
          </Field>

          <Field label="Target column" hint="leave blank for unsupervised">
            <input className="focusable" value={target} onChange={e=>setTarget(e.target.value)} style={inputStyle}/>
          </Field>

          <Field label="Description" hint="≥ 20 chars · guides feature & model agents">
            <textarea className="focusable" value={desc} onChange={e=>setDesc(e.target.value)} rows={4} style={{...inputStyle, resize:'vertical', lineHeight:1.5}}/>
          </Field>

          <Field label="Model provider (BYOM)">
            <Segmented value={provider} onChange={setProvider} options={[
              {value:'anthropic',label:'Anthropic'},{value:'openai',label:'OpenAI'},{value:'gemini',label:'Gemini'},
            ]}/>
          </Field>

          <button className="btn btn-secondary" style={{width:'100%', marginTop:6, justifyContent:'center'}}
            onClick={validate} disabled={phase==='validating'}>
            {phase==='validating' ? <><span className="spinner"/>Validating…</> : <><Icons.checkCircle size={16}/>Run Data Validator</>}
          </button>
        </div>
      </div>

      {/* validation report */}
      {phase!=='idle' && (
        <div className="card fade-up" style={{marginTop:18, padding:0, overflow:'hidden'}}>
          <div className="row" style={{justifyContent:'space-between', padding:'15px 22px', borderBottom:'1px solid var(--line)'}}>
            <div className="row gap-10">
              <AgentAvatar agent={AGENT.validator} size={30} state={passed?'done':'running'}/>
              <div className="col" style={{lineHeight:1.3}}>
                <span style={{fontSize:14, fontWeight:700}}>Data Validator</span>
                <span className="mono faint" style={{fontSize:11}}>validation_report.json</span>
              </div>
            </div>
            {passed
              ? <StatusPill status="passed" label="Passed · ready to run"/>
              : <StatusPill status="running" label="Validating…" spin/>}
          </div>
          <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:0}}>
            {VALIDATION_CHECKS.map((c,i)=>{
              const shown = i<revealed;
              const warn = c.state==='warn';
              return (
                <div key={c.key} className={shown?'fade-up':''} style={{
                  display: 'flex', alignItems:'flex-start', gap:11, padding:'14px 22px',
                  borderBottom:'1px solid var(--line-2)', borderRight: i%2===0?'1px solid var(--line-2)':'none',
                  opacity: shown?1:0.35,
                }}>
                  <div style={{flex:'none', marginTop:1, color: !shown?'var(--ink-4)':warn?'var(--warn)':'var(--ok)'}}>
                    {!shown ? <Icons.dot size={16}/> : warn ? <Icons.alert size={17}/> : <Icons.checkCircle size={18}/>}
                  </div>
                  <div className="col" style={{lineHeight:1.4}}>
                    <span style={{fontSize:13, fontWeight:600}}>{c.label}</span>
                    <span className="faint" style={{fontSize:12}}>{shown ? (warn ? c.warn : c.detail) : 'queued…'}</span>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="row" style={{justifyContent:'space-between', padding:'16px 22px', background:'var(--panel-2)'}}>
            <span className="faint" style={{fontSize:12.5}}>
              {passed ? <>5 passed · 1 warning · no blockers — the pipeline is clear to launch.</> : 'Running checks against the data profile…'}
            </span>
            <button className="btn btn-primary" disabled={!passed} onClick={startRun}>
              <Icons.play size={15}/>Run pipeline
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

const inputStyle = {
  width:'100%', padding:'10px 12px', borderRadius:9, border:'1px solid var(--line-3)',
  fontSize:13, fontFamily:'var(--sans)', color:'var(--ink)', background:'#fff', outline:'none',
};
function Field({ label, hint, children }) {
  return (
    <div className="col gap-8" style={{marginBottom:16}}>
      <div className="row" style={{justifyContent:'space-between'}}>
        <label style={{fontSize:12.5, fontWeight:600}}>{label}</label>
        {hint && <span className="faint" style={{fontSize:11}}>{hint}</span>}
      </div>
      {children}
    </div>
  );
}

window.UploadScreen = UploadScreen;
