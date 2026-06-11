/* ============================================================
   MITRA AI — P2 · Live Multi-Agent Pipeline (the hero)
   ============================================================ */

const LEVEL_COLOR = {
  info:'var(--ink-2)', ok:'var(--ok)', llm:'var(--accent)',
  ray:'#2f7fe0', hpt:'#d6398f', warn:'var(--warn)', error:'var(--err)',
};
const STAGE_ICON = { encode:'layers', scale:'scale', eval:'spark' };

function fmtElapsed(s){ const m=Math.floor(s/60), ss=s%60; return `${m}:${String(ss).padStart(2,'0')}`; }

function PipelineScreen({ go, run, startRun }) {
  const { state, stageIndex, stageProgress, logs, elapsed, verdict } = run;
  const logRef = useRef(null);
  useEffect(()=>{ if(logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [logs]);

  if (state==='idle') {
    return (
      <div className="page page-in" style={{display:'grid', placeItems:'center', minHeight:'70vh'}}>
        <div className="card" style={{padding:'48px 44px', textAlign:'center', maxWidth:440, display:'flex', flexDirection:'column', alignItems:'center', gap:16}}>
          <div style={{width:60,height:60,borderRadius:16,background:'var(--accent-soft)',display:'grid',placeItems:'center',color:'var(--accent)'}}><Icons.flow size={30}/></div>
          <div className="col gap-6">
            <h2 style={{fontSize:19, fontWeight:760}}>No active run</h2>
            <p className="muted" style={{margin:0, fontSize:13.5, lineHeight:1.55}}>Launch a run and watch all eight agents deliberate in real time, stage by stage.</p>
          </div>
          <button className="btn btn-primary" onClick={startRun}><Icons.play size={15}/>Start a run</button>
        </div>
      </div>
    );
  }

  const overall = Math.min(100, Math.round(((stageIndex + (state==='done'?0:stageProgress)) / STAGES.length) * 100)) || (state==='done'?100:0);
  const done = state==='done';

  function stageState(i){
    if (done) return 'done';
    if (i < stageIndex) return 'done';
    if (i === stageIndex) return 'running';
    return 'queued';
  }

  return (
    <div className="page page-wide page-in">
      {/* run header */}
      <div className="card" style={{padding:'16px 22px', marginBottom:18}}>
        <div className="row gap-16" style={{justifyContent:'space-between', flexWrap:'wrap'}}>
          <div className="row gap-12">
            <div style={{width:38,height:38,borderRadius:10,background:'var(--accent-soft)',display:'grid',placeItems:'center',color:'var(--accent)',flex:'none'}}><Icons.database size={20}/></div>
            <div className="col" style={{lineHeight:1.3}}>
              <div className="row gap-8"><span style={{fontWeight:700, fontSize:15}}>iris.csv</span><span className="tag">Classification</span></div>
              <span className="mono faint" style={{fontSize:11.5}}>run_4f2a · 6 candidate models · Ray ×4 workers</span>
            </div>
          </div>
          <div className="row gap-18">
            <Metric label="Elapsed" value={fmtElapsed(elapsed)}/>
            <Metric label="Stage" value={`${Math.min(stageIndex+ (done?0:1), STAGES.length)}/${STAGES.length}`}/>
            <Metric label="Models" value={stageIndex>=6?'6':'—'}/>
            {done
              ? <button className="btn btn-primary" onClick={()=>go('leaderboard')}><Icons.trophy size={16}/>View leaderboard</button>
              : <StatusPill status="running" label="Pipeline running" spin/>}
          </div>
        </div>
        <div className="row gap-12" style={{marginTop:14, alignItems:'center'}}>
          <div className="bar" style={{flex:1, height:7}}><i style={{width:`${overall}%`}}/></div>
          <span className="mono" style={{fontSize:12.5, fontWeight:600, color: done?'var(--ok)':'var(--accent)'}}>{overall}%</span>
        </div>
      </div>

      <div style={{display:'grid', gridTemplateColumns:'1.05fr 0.95fr', gap:18, alignItems:'start'}}>
        {/* pipeline spine */}
        <div className="card" style={{padding:'10px 22px 16px'}}>
          <div className="row" style={{justifyContent:'space-between', padding:'12px 0 8px'}}>
            <h3 style={{fontSize:14.5, fontWeight:700}}>Pipeline</h3>
            <span className="mono faint" style={{fontSize:11}}>typed artifact on every edge</span>
          </div>
          <div>
            {STAGES.map((s,i)=>{
              const st = stageState(i);
              const ag = s.agent ? AGENT[s.agent] : null;
              const SIcon = !ag ? Icons[STAGE_ICON[s.key]||'cpu'] : null;
              const last = i===STAGES.length-1;
              const lineColor = (st==='done') ? 'var(--accent)' : 'var(--line-3)';
              return (
                <div key={s.key} style={{display:'grid', gridTemplateColumns:'36px 1fr', gap:14}}>
                  {/* spine col */}
                  <div style={{display:'flex', flexDirection:'column', alignItems:'center'}}>
                    {ag
                      ? <AgentAvatar agent={ag} size={36} state={st}/>
                      : <div style={{width:36,height:36,borderRadius:11,flex:'none',display:'grid',placeItems:'center',
                          background: st==='done'?'var(--ink)':'var(--panel-3)', color: st==='done'?'#fff':st==='running'?'var(--accent)':'var(--ink-3)',
                          border:`1px solid ${st==='running'?'var(--accent-line)':'var(--line-3)'}`,
                          animation: st==='running'?'pulse-ring 1.6s infinite':'none'}}>
                          {st==='done'?<Icons.check size={18} sw={2.4}/>:<SIcon size={18}/>}
                        </div>}
                    {!last && <div style={{width:2, flex:1, minHeight:26, background:lineColor, margin:'4px 0', borderRadius:2, transition:'background .4s'}}/>}
                  </div>
                  {/* content */}
                  <div style={{paddingBottom: last?4:18, opacity: st==='queued'?0.55:1, transition:'opacity .3s'}}>
                    <div className="row gap-8" style={{justifyContent:'space-between'}}>
                      <div className="col" style={{lineHeight:1.3}}>
                        <div className="row gap-8">
                          <span style={{fontSize:13.5, fontWeight:650}}>{s.label}</span>
                          {ag ? <span className="tag">{ag.type}</span> : <span className="tag">Python</span>}
                        </div>
                        <span className="faint" style={{fontSize:11.5}}>{s.sub}</span>
                      </div>
                      <div style={{flex:'none'}}>
                        {st==='running' && <StatusPill status="running" label={`${Math.round(stageProgress*100)}%`} spin/>}
                        {st==='done' && <Icons.checkCircle size={18} style={{color:'var(--ok)'}}/>}
                        {st==='queued' && <span className="pill pill-queued"><span className="dot"/>Queued</span>}
                      </div>
                    </div>
                    <div className="row gap-6" style={{marginTop:7}}>
                      <Icons.doc size={12} style={{color:'var(--ink-4)'}}/>
                      <span className="mono" style={{fontSize:11, color: st==='done'?'var(--ink-2)':'var(--ink-4)'}}>{s.artifact}</span>
                    </div>
                    {st==='running' && <div className="bar" style={{height:4, marginTop:8, maxWidth:200}}><i style={{width:`${stageProgress*100}%`}}/></div>}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* right: terminal + verdict */}
        <div className="col gap-18" style={{position:'sticky', top:84}}>
          <div className="card" style={{padding:0, overflow:'hidden', background:'#16171d', border:'1px solid #23252e'}}>
            <div className="row" style={{justifyContent:'space-between', padding:'12px 16px', borderBottom:'1px solid #23252e'}}>
              <div className="row gap-8">
                <div className="row gap-6">
                  <span style={{width:10,height:10,borderRadius:99,background:'#ff5f57'}}/>
                  <span style={{width:10,height:10,borderRadius:99,background:'#febc2e'}}/>
                  <span style={{width:10,height:10,borderRadius:99,background:'#28c840'}}/>
                </div>
                <span className="mono" style={{fontSize:12, color:'#9aa0ac', marginLeft:6}}>SSE event stream</span>
              </div>
              <span className="row gap-6" style={{fontSize:11, color:'#6f7682'}}>
                <span style={{width:7,height:7,borderRadius:99,background: done?'#28c840':'var(--accent)', animation: done?'none':'blink 1.2s infinite'}}/>
                <span className="mono">{done?'closed':'live'}</span>
              </span>
            </div>
            <div ref={logRef} style={{height:360, overflowY:'auto', padding:'12px 16px', fontFamily:'var(--mono)', fontSize:12, lineHeight:1.85}}>
              {logs.length===0 && <span style={{color:'#6f7682'}}>// awaiting first event…</span>}
              {logs.map((l,i)=>(
                <div key={i} className="row gap-8" style={{alignItems:'flex-start'}}>
                  <span style={{color:'#565c68', flex:'none'}}>{l.ts}</span>
                  <span style={{color: LEVEL_COLOR[l.level]||'#9aa0ac', flex:'none', fontWeight:600, width:48}}>{l.level==='ok'?'done':l.level}</span>
                  <span style={{color:'#c8cdd6'}}>{l.msg}</span>
                </div>
              ))}
              {!done && <div className="row gap-8"><span style={{color:'#565c68'}}>{'>'}</span><span style={{width:7,height:14,background:'var(--accent)',display:'inline-block',animation:'blink 1s infinite'}}/></div>}
            </div>
          </div>

          {/* judge verdict card */}
          <div className="card" style={{padding:'18px 20px', borderColor: verdict?'var(--accent-line)':'var(--line)', background: verdict?'linear-gradient(120deg,#fff,#faf9ff)':'#fff', transition:'all .4s'}}>
            <div className="row gap-10" style={{marginBottom: verdict?14:0}}>
              <AgentAvatar agent={AGENT.judge} size={32} state={verdict?'done':stageIndex>=8?'running':'idle'}/>
              <div className="col" style={{lineHeight:1.3}}>
                <span style={{fontSize:14, fontWeight:700}}>Judge verdict</span>
                <span className="mono faint" style={{fontSize:11}}>verdict.json</span>
              </div>
              {verdict && <span className="pill pill-done" style={{marginLeft:'auto'}}><Icons.check size={12} sw={3}/>Converged</span>}
            </div>
            {verdict ? (
              <div className="fade-up">
                <div className="row gap-12" style={{padding:'12px 14px', background:'var(--accent-soft)', borderRadius:11, marginBottom:12}}>
                  <Icons.trophy size={22} style={{color:'var(--accent)'}}/>
                  <div className="col" style={{lineHeight:1.3}}>
                    <span style={{fontSize:14.5, fontWeight:750}}>XGBoost wins</span>
                    <span className="faint" style={{fontSize:11.5}}>best F1 · lowest overfit gap (0.018)</span>
                  </div>
                  <div className="col" style={{marginLeft:'auto', alignItems:'flex-end', lineHeight:1.2}}>
                    <span className="mono" style={{fontSize:20, fontWeight:750, color:'var(--accent-ink)'}}>94.6</span>
                    <span className="faint" style={{fontSize:10}}>judge score</span>
                  </div>
                </div>
                <div className="col gap-6">
                  {[['Generalization gap ≤ threshold','pass'],['Accuracy floor (0.90) met','pass'],['SHAP stable across folds','pass']].map(([t],k)=>(
                    <div key={k} className="row gap-8" style={{fontSize:12.5}}><Icons.checkCircle size={15} style={{color:'var(--ok)'}}/>{t}</div>
                  ))}
                </div>
              </div>
            ) : (
              <p className="faint" style={{margin:'10px 0 0', fontSize:12.5, lineHeight:1.5}}>The Judge weighs every agent's proposal — confidence, task metrics, and resource cost — then converges once generalization is proven.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="col" style={{lineHeight:1.2, alignItems:'flex-start'}}>
      <span className="mono" style={{fontSize:17, fontWeight:700}}>{value}</span>
      <span className="faint" style={{fontSize:10.5, textTransform:'uppercase', letterSpacing:'.05em'}}>{label}</span>
    </div>
  );
}

window.PipelineScreen = PipelineScreen;
