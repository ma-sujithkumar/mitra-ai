/* ============================================================
   MITRA AI — Dashboard (home overview)
   ============================================================ */

function DriftDot({ drift }) {
  const map = { stable:['var(--ok)','Stable'], watch:['var(--warn)','Watch'], '—':['var(--ink-4)','—'] };
  const [c,t] = map[drift] || map['—'];
  return <span className="row gap-6" style={{fontSize:12, color:'var(--ink-2)'}}><span style={{width:7,height:7,borderRadius:99,background:c}}/>{t}</span>;
}

function Dashboard({ go, startRun }) {
  const accSeries = [0.88, 0.91, 0.884, 0.94, 0.96, 0.973];
  return (
    <div className="page page-in">
      {/* hero */}
      <div className="card" style={{
        padding:'26px 28px', marginBottom:22, position:'relative', overflow:'hidden',
        background:'linear-gradient(120deg, #fff 0%, #faf9ff 55%, #f3efff 100%)',
        border:'1px solid var(--accent-line)',
      }}>
        <div style={{position:'absolute', right:-40, top:-50, width:260, height:260, borderRadius:'50%',
          background:'radial-gradient(circle, rgba(108,71,255,.10), transparent 70%)'}}/>
        <div className="row" style={{justifyContent:'space-between', alignItems:'flex-start', gap:20, position:'relative'}}>
          <div className="col gap-10" style={{maxWidth:560}}>
            <h1 style={{fontSize:25, fontWeight:780}}>A team of agents, one optimized model.</h1>
            <p className="muted" style={{margin:0, fontSize:14, lineHeight:1.55}}>
              Upload a dataset and minimal metadata. MITRA's specialist agents profile, engineer,
              train, and tune in parallel — then the Judge converges on the best model and explains why.
            </p>
            <div className="row gap-10" style={{marginTop:6}}>
              <button className="btn btn-primary" onClick={startRun}><Icons.play size={15}/>Start a new run</button>
              <button className="btn btn-secondary" onClick={()=>go('leaderboard')}><Icons.trophy size={16}/>View last leaderboard</button>
            </div>
          </div>

        </div>
      </div>

      {/* stats */}
      <div style={{display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:14, marginBottom:22}}>
        <Stat icon="layers" label="Total runs" value="24" delta="+5 this week"/>
        <Stat icon="cpu" label="Models trained" value="148" accent/>
        <Stat icon="target" label="Best accuracy" value="97.3" unit="%" accent/>
        <Stat icon="gauge" label="Avg run time" value="3.8" unit="min"/>
      </div>

      <div style={{display:'grid', gridTemplateColumns:'1.55fr 1fr', gap:18}}>
        {/* recent runs */}
        <div className="card" style={{padding:0, overflow:'hidden'}}>
          <div className="row" style={{justifyContent:'space-between', padding:'16px 20px', borderBottom:'1px solid var(--line)'}}>
            <div className="col" style={{lineHeight:1.3}}>
              <h3 style={{fontSize:15, fontWeight:700}}>Recent runs</h3>
              <span className="faint" style={{fontSize:12}}>Latest pipeline executions</span>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={()=>go('upload')}><Icons.plus size={15}/>New</button>
          </div>
          <table style={{width:'100%', borderCollapse:'collapse', fontSize:13}}>
            <thead>
              <tr style={{color:'var(--ink-3)', fontSize:11, textTransform:'uppercase', letterSpacing:'.04em'}}>
                {['Run','Dataset','Task','Best model','Acc','Drift',''].map((h,i)=>(
                  <th key={i} style={{textAlign: i>=4&&i<6?'right':'left', fontWeight:600, padding:'10px 20px', borderBottom:'1px solid var(--line)'}}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {RUNS.map((r,i)=>(
                <tr key={r.id} style={{borderBottom: i<RUNS.length-1?'1px solid var(--line-2)':'none', cursor:'pointer'}}
                    onClick={()=>go('leaderboard')}
                    onMouseEnter={e=>e.currentTarget.style.background='var(--panel-2)'}
                    onMouseLeave={e=>e.currentTarget.style.background='transparent'}>
                  <td style={{padding:'11px 20px'}}><span className="mono" style={{fontSize:12, color:'var(--ink-2)'}}>{r.id}</span></td>
                  <td style={{padding:'11px 20px', fontWeight:600}}>{r.dataset}</td>
                  <td style={{padding:'11px 20px', color:'var(--ink-2)'}}><span className="tag">{r.task}</span></td>
                  <td style={{padding:'11px 20px'}}>{r.best}</td>
                  <td style={{padding:'11px 20px', textAlign:'right'}} className="mono">{r.acc!=null?(r.acc*100).toFixed(1)+'%':'—'}</td>
                  <td style={{padding:'11px 20px', textAlign:'right'}}><div style={{display:'inline-flex'}}><DriftDot drift={r.drift}/></div></td>
                  <td style={{padding:'11px 12px', textAlign:'right', color:'var(--ink-4)'}}><Icons.arrowR size={16}/></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* right column */}
        <div className="col gap-18">
          <div className="card" style={{padding:'18px 20px'}}>
            <div className="row" style={{justifyContent:'space-between', marginBottom:12}}>
              <h3 style={{fontSize:14.5, fontWeight:700}}>Accuracy trend</h3>
              <span className="pill pill-done"><span className="dot"/>+9.3 pts</span>
            </div>
            <div className="row" style={{justifyContent:'space-between', alignItems:'flex-end'}}>
              <div className="col gap-2">
                <span className="mono" style={{fontSize:26, fontWeight:750}}>97.3<span style={{fontSize:14, color:'var(--ink-3)'}}>%</span></span>
                <span className="faint" style={{fontSize:11.5}}>last 6 runs</span>
              </div>
              <Sparkline points={accSeries} w={150} h={48}/>
            </div>
          </div>

          <div className="card" style={{padding:'18px 20px', flex:1}}>
            <h3 style={{fontSize:14.5, fontWeight:700, marginBottom:4}}>Agent roster</h3>
            <p className="faint" style={{fontSize:12, margin:'0 0 14px'}}>One agent per teammate · 8 owners</p>
            <div className="col gap-10">
              {AGENTS.map(a=>(
                <div key={a.id} className="row gap-10">
                  <AgentAvatar agent={a} size={30}/>
                  <div className="col" style={{lineHeight:1.25, minWidth:0}}>
                    <span style={{fontSize:13, fontWeight:600}}>{a.name}</span>
                    <span className="faint" style={{fontSize:11, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis'}}>{a.role}</span>
                  </div>
                  <div className="row gap-6" style={{marginLeft:'auto', flex:'none'}}>
                    <span className="tag">{a.type}</span>
                    <span style={{width:7,height:7,borderRadius:99,background:'var(--ok)'}}/>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

window.Dashboard = Dashboard;
