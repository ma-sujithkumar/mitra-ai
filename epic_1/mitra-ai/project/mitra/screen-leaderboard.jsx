/* ============================================================
   MITRA AI — P3 · Leaderboard + SHAP report
   ============================================================ */

function LeaderboardScreen({ go, startRun }) {
  const winner = LEADERBOARD[0];
  const maxAcc = Math.max(...LEADERBOARD.map(m=>m.acc));
  return (
    <div className="page page-wide page-in">
      {/* winner banner */}
      <div className="card" style={{
        padding:'22px 26px', marginBottom:20, overflow:'hidden', position:'relative',
        background:'linear-gradient(120deg,#fff 0%, #faf9ff 55%, #f1ecff 100%)', border:'1px solid var(--accent-line)',
      }}>
        <div style={{position:'absolute', right:-30, top:-60, width:240, height:240, borderRadius:'50%', background:'radial-gradient(circle, rgba(108,71,255,.12), transparent 70%)'}}/>
        <div className="row gap-20" style={{justifyContent:'space-between', flexWrap:'wrap', position:'relative'}}>
          <div className="row gap-16">
            <div style={{width:56,height:56,borderRadius:16,background:'linear-gradient(150deg,var(--accent),var(--accent-strong))',display:'grid',placeItems:'center',color:'#fff',flex:'none',boxShadow:'0 8px 22px rgba(108,71,255,.35)'}}>
              <Icons.trophy size={28}/>
            </div>
            <div className="col gap-4">
              <span className="pill pill-done" style={{alignSelf:'flex-start'}}><Icons.check size={12} sw={3}/>Judge converged · validated for generalization</span>
              <h1 style={{fontSize:23, fontWeight:780}}>{winner.model} <span className="muted" style={{fontWeight:500, fontSize:16}}>is the recommended model</span></h1>
              <span className="mono faint" style={{fontSize:12}}>iris.csv · classification · 6 candidates evaluated</span>
            </div>
          </div>
          <div className="row gap-22">
            <BigMetric value={(winner.acc*100).toFixed(1)} unit="%" label="Accuracy"/>
            <BigMetric value={winner.f1.toFixed(3)} label="F1 score"/>
            <BigMetric value={winner.judge.toFixed(1)} label="Judge score" accent/>
            <div className="row gap-10" style={{alignItems:'center'}}>
              <button className="btn btn-secondary"><Icons.download size={16}/>Export</button>
              <button className="btn btn-primary" onClick={startRun}><Icons.play size={15}/>New run</button>
            </div>
          </div>
        </div>
      </div>

      <div style={{display:'grid', gridTemplateColumns:'1.55fr 1fr', gap:18, alignItems:'start'}}>
        {/* leaderboard table */}
        <div className="card" style={{padding:0, overflow:'hidden'}}>
          <div className="row" style={{justifyContent:'space-between', padding:'16px 22px', borderBottom:'1px solid var(--line)'}}>
            <div className="col" style={{lineHeight:1.3}}>
              <h3 style={{fontSize:15, fontWeight:700}}>Model leaderboard</h3>
              <span className="faint" style={{fontSize:12}}>Ranked by Judge score · quantitative justification per row</span>
            </div>
          </div>
          <table style={{width:'100%', borderCollapse:'collapse', fontSize:13}}>
            <thead>
              <tr style={{color:'var(--ink-3)', fontSize:10.5, textTransform:'uppercase', letterSpacing:'.04em'}}>
                {[['#','left'],['Model','left'],['Accuracy','left'],['F1','right'],['AUC','right'],['Gap','right'],['Judge','right']].map(([h,a],i)=>(
                  <th key={i} style={{textAlign:a, fontWeight:600, padding:'11px 18px', borderBottom:'1px solid var(--line)'}}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {LEADERBOARD.map((m,i)=>(
                <tr key={m.model} style={{borderBottom: i<LEADERBOARD.length-1?'1px solid var(--line-2)':'none', background: m.winner?'var(--accent-soft)':'transparent'}}>
                  <td style={{padding:'13px 18px'}}>
                    <div style={{width:24,height:24,borderRadius:7,display:'grid',placeItems:'center',fontWeight:700,fontSize:12,
                      background: m.winner?'var(--accent)':'var(--panel-3)', color: m.winner?'#fff':'var(--ink-3)'}} className="mono">{m.rank}</div>
                  </td>
                  <td style={{padding:'13px 18px'}}>
                    <div className="col" style={{lineHeight:1.3}}>
                      <span className="row gap-7" style={{fontWeight:650, fontSize:13.5}}>{m.model}{m.winner && <Icons.trophy size={14} style={{color:'var(--accent)'}}/>}</span>
                      <span className="mono faint" style={{fontSize:10.5}}>{m.hp}</span>
                    </div>
                  </td>
                  <td style={{padding:'13px 18px', minWidth:150}}>
                    <div className="row gap-8">
                      <div className="bar" style={{flex:1, height:7}}><i style={{width:`${(m.acc/maxAcc)*100}%`, background: m.winner?'var(--accent)':'var(--ink-4)'}}/></div>
                      <span className="mono" style={{fontSize:12, fontWeight:600, width:42, textAlign:'right'}}>{(m.acc*100).toFixed(1)}</span>
                    </div>
                  </td>
                  <td style={{padding:'13px 18px', textAlign:'right'}} className="mono">{m.f1.toFixed(3)}</td>
                  <td style={{padding:'13px 18px', textAlign:'right'}} className="mono">{m.auc.toFixed(3)}</td>
                  <td style={{padding:'13px 18px', textAlign:'right'}} className="mono">
                    <span style={{color: m.gap<=0.025?'var(--ok)':m.gap<=0.04?'var(--warn)':'var(--err)'}}>{m.gap.toFixed(3)}</span>
                  </td>
                  <td style={{padding:'13px 18px', textAlign:'right'}}>
                    <span className="mono" style={{fontWeight:700, fontSize:13, color: m.winner?'var(--accent-ink)':'var(--ink)'}}>{m.judge.toFixed(1)}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* right rail */}
        <div className="col gap-18">
          {/* SHAP */}
          <div className="card" style={{padding:'18px 20px'}}>
            <div className="row gap-8" style={{marginBottom:4}}>
              <Icons.spark size={17} style={{color:'var(--accent)'}}/>
              <h3 style={{fontSize:14.5, fontWeight:700}}>Why this model</h3>
            </div>
            <p className="faint" style={{fontSize:12, margin:'0 0 16px'}}>SHAP feature importance · {winner.model}</p>
            <HBars data={SHAP}/>
            <p className="faint" style={{fontSize:11.5, margin:'14px 0 0', lineHeight:1.5}}>Petal geometry drives ~78% of predictions — consistent with botanical separability.</p>
          </div>

          {/* judge reasoning */}
          <div className="card" style={{padding:'18px 20px'}}>
            <div className="row gap-10" style={{marginBottom:12}}>
              <AgentAvatar agent={AGENT.judge} size={30} state="done"/>
              <div className="col" style={{lineHeight:1.3}}>
                <span style={{fontSize:14, fontWeight:700}}>Judge reasoning</span>
                <span className="mono faint" style={{fontSize:11}}>auditable · verdict.json</span>
              </div>
            </div>
            <div className="col gap-9">
              {[
                ['Lowest overfit gap','0.018 vs 0.044 worst','ok'],
                ['Accuracy floor (0.90)','5 / 6 candidates clear it','ok'],
                ['Best F1 under budget','0.972 within 2.4s train','ok'],
                ['SHAP stability','consistent across 5 folds','ok'],
              ].map(([t,d],k)=>(
                <div key={k} className="row gap-10" style={{padding:'9px 0', borderBottom: k<3?'1px solid var(--line-2)':'none'}}>
                  <Icons.checkCircle size={17} style={{color:'var(--ok)', flex:'none', marginTop:1}}/>
                  <div className="col" style={{lineHeight:1.35}}>
                    <span style={{fontSize:13, fontWeight:600}}>{t}</span>
                    <span className="mono faint" style={{fontSize:11}}>{d}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* drift monitor — v2 teaser */}
          <div className="card" style={{padding:'18px 20px', position:'relative', overflow:'hidden'}}>
            <div className="row" style={{justifyContent:'space-between', marginBottom:12}}>
              <div className="row gap-8"><Icons.gauge size={17} style={{color:'var(--ink-3)'}}/><h3 style={{fontSize:14.5, fontWeight:700}}>Drift monitor</h3></div>
              <span className="pill pill-queued"><span className="dot"/>v2 · roadmap</span>
            </div>
            <div className="col gap-10" style={{opacity:.62}}>
              {[['Covariate drift','no shift detected',8],['Concept drift','within tolerance',14]].map(([t,d,p],k)=>(
                <div key={k} className="col gap-6">
                  <div className="row" style={{justifyContent:'space-between', fontSize:12.5}}><span>{t}</span><span className="mono faint">{d}</span></div>
                  <div className="bar" style={{height:6}}><i style={{width:`${p}%`, background:'var(--ink-3)'}}/></div>
                </div>
              ))}
            </div>
            <p className="faint" style={{fontSize:11.5, margin:'12px 0 0', lineHeight:1.5}}>A production agent will watch deployed models and trigger re-evaluation on drift.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function BigMetric({ value, unit, label, accent }) {
  return (
    <div className="col" style={{lineHeight:1.1, alignItems:'flex-start'}}>
      <span className="row" style={{alignItems:'baseline', gap:2}}>
        <span className="mono" style={{fontSize:26, fontWeight:760, color: accent?'var(--accent-ink)':'var(--ink)'}}>{value}</span>
        {unit && <span className="mono" style={{fontSize:14, color:'var(--ink-3)', fontWeight:600}}>{unit}</span>}
      </span>
      <span className="faint" style={{fontSize:10.5, textTransform:'uppercase', letterSpacing:'.05em', marginTop:3}}>{label}</span>
    </div>
  );
}

window.LeaderboardScreen = LeaderboardScreen;
