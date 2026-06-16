/* ============================================================
   MITRA AI — shared components
   ============================================================ */
const { useState, useEffect, useRef } = React;

// hue → harmonized agent color (consistent chroma/lightness, varying hue)
function agentColor(hue) {
  return {
    fg:   `oklch(0.55 0.16 ${hue})`,
    bg:   `oklch(0.96 0.045 ${hue})`,
    line: `oklch(0.90 0.07 ${hue})`,
    solid:`oklch(0.62 0.17 ${hue})`,
  };
}

// rounded-square monogram avatar for an agent
function AgentAvatar({ agent, size=34, state='idle' }) {
  const c = agentColor(agent.hue);
  const running = state === 'running';
  const done = state === 'done';
  return (
    <div style={{
      width:size, height:size, borderRadius: size*0.3,
      display:'grid', placeItems:'center', flex:'none',
      background: done ? c.solid : c.bg,
      border:`1px solid ${done ? c.solid : c.line}`,
      color: done ? '#fff' : c.fg,
      fontFamily:'var(--mono)', fontWeight:700, fontSize:size*0.36,
      letterSpacing:'-0.02em', position:'relative',
      animation: running ? 'pulse-ring 1.6s infinite' : 'none',
      transition:'background .3s, color .3s, border-color .3s',
    }}>
      {done ? <Icons.check size={size*0.5} sw={2.4}/> : agent.short}
    </div>
  );
}

// status pill
const STATUS_MAP = {
  idle:   ['pill-idle','Idle'],
  queued: ['pill-queued','Queued'],
  running:['pill-run','Running'],
  done:   ['pill-done','Done'],
  review: ['pill-warn','Review'],
  warn:   ['pill-warn','Watch'],
  error:  ['pill-err','Error'],
  passed: ['pill-done','Passed'],
};
function StatusPill({ status, label, spin=false }) {
  const [cls, txt] = STATUS_MAP[status] || ['pill-idle', status];
  return (
    <span className={`pill ${cls}`}>
      {status==='running' && spin ? <span className="spinner" style={{width:9,height:9,borderWidth:1.6}}/> : <span className="dot"/>}
      {label || txt}
    </span>
  );
}

// ---- sidebar ----
const NAV = [
  { key:'dashboard', label:'Dashboard',  icon:'grid' },
  { key:'upload',    label:'New Run',    icon:'upload' },
  { key:'pipeline',  label:'Pipeline',   icon:'flow' },
  { key:'leaderboard', label:'Leaderboard', icon:'trophy' },
];

function Sidebar({ route, go, runState }) {
  return (
    <aside style={{
      background:'#fff', borderRight:'1px solid var(--line)',
      display:'flex', flexDirection:'column', padding:'20px 14px', gap:4, height:'100%',
    }}>
      {/* brand */}
      <div className="row gap-10" style={{padding:'4px 8px 18px'}}>
        <div style={{
          width:34, height:34, borderRadius:10, flex:'none',
          background:'linear-gradient(150deg, var(--accent), var(--accent-strong))',
          display:'grid', placeItems:'center', color:'#fff',
          boxShadow:'0 4px 12px rgba(108,71,255,.35)',
        }}>
          <Icons.layers size={19} sw={1.9}/>
        </div>
        <div className="col" style={{lineHeight:1.1}}>
          <div style={{fontWeight:800, fontSize:16, letterSpacing:'-0.03em'}}>MITRA<span style={{color:'var(--accent)'}}> AI</span></div>
          <div className="mono" style={{fontSize:9.5, color:'var(--ink-3)', letterSpacing:'.02em', marginTop:2}}>AGENTIC AUTOML</div>
        </div>
      </div>

      <div className="mono" style={{fontSize:10, color:'var(--ink-4)', letterSpacing:'.08em', padding:'0 10px 6px'}}>WORKSPACE</div>
      {NAV.map(n => {
        const I = Icons[n.icon]; const active = route===n.key;
        const badge = n.key==='pipeline' && runState==='running';
        return (
          <button key={n.key} onClick={()=>go(n.key)} className={`nav-item focusable ${active?'active':''}`}>
            <I size={18} sw={active?1.9:1.7} className="nav-ic"/>
            {n.label}
            {badge && <span className="spinner" style={{marginLeft:'auto', width:11, height:11}}/>}
            {n.key==='pipeline' && runState==='done' && <Icons.checkCircle size={15} style={{marginLeft:'auto', color:'var(--ok)'}}/>}
          </button>
        );
      })}

      <div style={{flex:1}}/>

      <button onClick={()=>go('settings')} className={`nav-item focusable ${route==='settings'?'active':''}`} style={{marginBottom:6}}>
        <Icons.gear size={18} sw={route==='settings'?1.9:1.7} className="nav-ic"/>
        Settings
      </button>

      <button onClick={()=>go('settings')} className="row gap-10 focusable" style={{padding:'12px 8px 0', background:'none', border:'none', borderTop:'1px solid var(--line)', cursor:'pointer', textAlign:'left', width:'100%'}}>
        <div style={{width:30, height:30, borderRadius:99, background:'var(--panel-3)', border:'1px solid var(--line-3)', display:'grid', placeItems:'center', fontWeight:700, fontSize:12, color:'var(--ink-2)'}}>K</div>
        <div className="col" style={{lineHeight:1.2}}>
          <div style={{fontSize:12.5, fontWeight:600}}>Course Team</div>
          <div style={{fontSize:11, color:'var(--ink-3)'}}>Self-hosted · local</div>
        </div>
        <Icons.gear size={16} style={{marginLeft:'auto', color:'var(--ink-3)'}}/>
      </button>
    </aside>
  );
}

// ---- top bar ----
function TopBar({ title, sub, right, icon }) {
  const I = icon ? Icons[icon] : null;
  return (
    <header style={{
      height:64, flex:'none', borderBottom:'1px solid var(--line)',
      background:'rgba(255,255,255,0.8)', backdropFilter:'blur(10px)',
      display:'flex', alignItems:'center', padding:'0 28px', gap:14, position:'sticky', top:0, zIndex:5,
    }}>
      {I && <div style={{width:32,height:32,borderRadius:9,background:'var(--accent-soft)',display:'grid',placeItems:'center',color:'var(--accent)',flex:'none'}}><I size={18}/></div>}
      <div className="col" style={{lineHeight:1.25}}>
        <div style={{fontWeight:700, fontSize:16, letterSpacing:'-0.02em'}}>{title}</div>
        {sub && <div style={{fontSize:12, color:'var(--ink-3)'}}>{sub}</div>}
      </div>
      <div className="row gap-10" style={{marginLeft:'auto'}}>{right}</div>
    </header>
  );
}

// ---- stat tile ----
function Stat({ icon, label, value, unit, delta, accent }) {
  const I = Icons[icon];
  return (
    <div className="card" style={{padding:'16px 18px', display:'flex', flexDirection:'column', gap:10}}>
      <div className="row gap-8" style={{justifyContent:'space-between'}}>
        <span style={{fontSize:12.5, color:'var(--ink-2)', fontWeight:550}}>{label}</span>
        {I && <div style={{width:28,height:28,borderRadius:8,background: accent?'var(--accent-soft)':'var(--panel-3)',display:'grid',placeItems:'center',color: accent?'var(--accent)':'var(--ink-3)'}}><I size={16}/></div>}
      </div>
      <div className="row" style={{alignItems:'baseline', gap:5}}>
        <span style={{fontSize:27, fontWeight:750, letterSpacing:'-0.03em'}} className="mono">{value}</span>
        {unit && <span style={{fontSize:13, color:'var(--ink-3)', fontWeight:600}}>{unit}</span>}
        {delta && <span style={{marginLeft:'auto', fontSize:11.5, fontWeight:650, color:'var(--ok)'}}>{delta}</span>}
      </div>
    </div>
  );
}

// ---- horizontal bar chart (SHAP) ----
function HBars({ data, max, fmt=(v)=>v.toFixed(3), color='var(--accent)' }) {
  const m = max || Math.max(...data.map(d=>d.value));
  return (
    <div className="col gap-12">
      {data.map((d,i)=>(
        <div key={d.feature} className="col gap-6">
          <div className="row" style={{justifyContent:'space-between', fontSize:12.5}}>
            <span className="mono" style={{color:'var(--ink-2)'}}>{d.feature}</span>
            <span className="mono" style={{fontWeight:600}}>{fmt(d.value)}</span>
          </div>
          <div className="bar" style={{height:9}}>
            <i style={{width:`${(d.value/m)*100}%`, background:color, animation:`fadeUp .5s ${i*0.08}s both`}}/>
          </div>
        </div>
      ))}
    </div>
  );
}

// ---- mini sparkline ----
function Sparkline({ points, w=120, h=34, color='var(--accent)' }) {
  const max = Math.max(...points), min = Math.min(...points);
  const rng = max-min || 1;
  const step = w/(points.length-1);
  const pts = points.map((p,i)=>`${i*step},${h-2-((p-min)/rng)*(h-4)}`).join(' ');
  return (
    <svg width={w} height={h} style={{display:'block', overflow:'visible'}}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <circle cx={(points.length-1)*step} cy={h-2-((points[points.length-1]-min)/rng)*(h-4)} r="3" fill={color}/>
    </svg>
  );
}

// ---- segmented control ----
function Segmented({ options, value, onChange }) {
  return (
    <div className="row" style={{background:'var(--panel-3)', border:'1px solid var(--line-3)', borderRadius:9, padding:3, gap:2}}>
      {options.map(o=>{
        const active = o.value===value;
        return (
          <button key={o.value} onClick={()=>onChange(o.value)} style={{
            border:'none', cursor:'pointer', padding:'5px 12px', borderRadius:6,
            fontSize:12.5, fontWeight:600, fontFamily:'var(--sans)',
            background: active?'#fff':'transparent', color: active?'var(--ink)':'var(--ink-2)',
            boxShadow: active?'var(--sh-sm)':'none', transition:'all .14s',
          }}>{o.label}</button>
        );
      })}
    </div>
  );
}

// ---- shared form field + input ----
const FIELD_INPUT = {
  width:'100%', padding:'10px 12px', borderRadius:9, border:'1px solid var(--line-3)',
  fontSize:13, fontFamily:'var(--sans)', color:'var(--ink)', background:'#fff', outline:'none',
};
function FormField({ label, hint, children }) {
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

// ---- shared BYOM (bring-your-own-model) configuration fields ----
const PROVIDERS = [
  { value:'anthropic', label:'Anthropic', key:'sk-ant-…' },
  { value:'openai',    label:'OpenAI',    key:'sk-…' },
  { value:'gemini',    label:'Gemini',    key:'AIza…' },
];
function ByomFields({ llm, setLlm }) {
  const [show, setShow] = useState(false);
  const set = (k, v) => setLlm({ ...llm, [k]: v });
  const ph = (PROVIDERS.find(p => p.value === llm.provider) || PROVIDERS[0]).key;
  return (
    <div>
      <FormField label="Provider">
        <Segmented value={llm.provider} onChange={v => set('provider', v)}
          options={PROVIDERS.map(p => ({ value: p.value, label: p.label }))} />
      </FormField>
      <FormField label="API key" hint="required">
        <div style={{ position:'relative' }}>
          <input type={show ? 'text' : 'password'} value={llm.apiKey}
            onChange={e => set('apiKey', e.target.value)} placeholder={ph} className="focusable"
            style={{ ...FIELD_INPUT, paddingRight:60, fontFamily:'var(--mono)', letterSpacing: show?'0':'.08em' }} />
          <button onClick={() => setShow(s => !s)} type="button" style={{
            position:'absolute', right:6, top:'50%', transform:'translateY(-50%)',
            border:'none', background:'var(--panel-3)', color:'var(--ink-2)', cursor:'pointer',
            fontSize:11, fontWeight:600, padding:'4px 9px', borderRadius:6 }}>{show ? 'Hide' : 'Show'}</button>
        </div>
      </FormField>
      <FormField label="Gateway server URL" hint="optional">
        <input value={llm.gateway} onChange={e => set('gateway', e.target.value)} className="focusable"
          placeholder="https://litellm.local:4000" style={{ ...FIELD_INPUT, fontFamily:'var(--mono)' }} />
      </FormField>
      <div className="row gap-8" style={{ fontSize:11.5, color:'var(--ink-3)', lineHeight:1.45 }}>
        <Icons.cpu size={14} style={{ flex:'none', marginTop:1 }} />
        <span>Every agent routes through the LiteLLM factory — no agent ever makes a direct LLM call.</span>
      </div>
    </div>
  );
}

Object.assign(window, { agentColor, AgentAvatar, StatusPill, Sidebar, TopBar, Stat, HBars, Sparkline, Segmented, NAV, FormField, FIELD_INPUT, ByomFields, PROVIDERS });
