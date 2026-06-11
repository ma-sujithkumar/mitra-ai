/* ============================================================
   MITRA AI — line icons (simple stroke glyphs)
   ============================================================ */
const Icon = ({ d, size=18, sw=1.7, fill=false, style, className, children }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill?'currentColor':'none'}
       stroke={fill?'none':'currentColor'} strokeWidth={sw} strokeLinecap="round"
       strokeLinejoin="round" style={style} className={className} aria-hidden="true">
    {children || <path d={d} />}
  </svg>
);

const Icons = {
  grid:    (p)=> <Icon {...p}><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></Icon>,
  upload:  (p)=> <Icon {...p}><path d="M12 16V4m0 0L7 9m5-5 5 5"/><path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2"/></Icon>,
  flow:    (p)=> <Icon {...p}><circle cx="5" cy="6" r="2.4"/><circle cx="19" cy="6" r="2.4"/><circle cx="12" cy="18" r="2.4"/><path d="M7.2 7.3 10 15.8M16.8 7.3 14 15.8M7 6h10"/></Icon>,
  trophy:  (p)=> <Icon {...p}><path d="M7 4h10v4a5 5 0 0 1-10 0V4Z"/><path d="M7 5H4v2a3 3 0 0 0 3 3M17 5h3v2a3 3 0 0 1-3 3M9 16h6M10 16v3M14 16v3M8 21h8"/></Icon>,
  play:    (p)=> <Icon {...p} fill={true}><path d="M7 5.5v13l11-6.5-11-6.5Z"/></Icon>,
  check:   (p)=> <Icon {...p}><path d="M4 12.5 9 17.5 20 6.5"/></Icon>,
  checkCircle:(p)=> <Icon {...p}><circle cx="12" cy="12" r="9"/><path d="M8 12.2l2.6 2.6L16 9.4"/></Icon>,
  alert:   (p)=> <Icon {...p}><path d="M12 3 1.8 20.5h20.4L12 3Z"/><path d="M12 9.5v5M12 17.5v.01"/></Icon>,
  x:       (p)=> <Icon {...p}><path d="M6 6l12 12M18 6 6 18"/></Icon>,
  arrowR:  (p)=> <Icon {...p}><path d="M5 12h14M13 6l6 6-6 6"/></Icon>,
  spark:   (p)=> <Icon {...p}><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6.3 6.3l2.6 2.6M15.1 15.1l2.6 2.6M17.7 6.3l-2.6 2.6M8.9 15.1l-2.6 2.6"/></Icon>,
  cpu:     (p)=> <Icon {...p}><rect x="6" y="6" width="12" height="12" rx="2.5"/><path d="M9.5 9.5h5v5h-5z"/><path d="M9 3v2.5M15 3v2.5M9 18.5V21M15 18.5V21M3 9h2.5M3 15h2.5M18.5 9H21M18.5 15H21"/></Icon>,
  doc:     (p)=> <Icon {...p}><path d="M6 3h8l4 4v14a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z"/><path d="M13 3v5h5"/></Icon>,
  gauge:   (p)=> <Icon {...p}><path d="M4 18a8 8 0 1 1 16 0"/><path d="M12 18l4-5"/><circle cx="12" cy="18" r="1.3" fill="currentColor" stroke="none"/></Icon>,
  layers:  (p)=> <Icon {...p}><path d="M12 3 3 8l9 5 9-5-9-5Z"/><path d="M3 13l9 5 9-5M3 18l9 5 9-5" opacity="0.5"/></Icon>,
  scale:   (p)=> <Icon {...p}><path d="M12 4v16M6 8h12M6 8l-3 6a3 3 0 0 0 6 0L6 8ZM18 8l-3 6a3 3 0 0 0 6 0l-3-6Z"/></Icon>,
  filter:  (p)=> <Icon {...p}><path d="M3 5h18l-7 8v6l-4 2v-8L3 5Z"/></Icon>,
  gear:    (p)=> <Icon {...p}><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M19.1 4.9 17 7M7 17l-2.1 2.1"/></Icon>,
  bell:    (p)=> <Icon {...p}><path d="M6 9a6 6 0 0 1 12 0c0 5 2 6 2 6H4s2-1 2-6Z"/><path d="M10 20a2 2 0 0 0 4 0"/></Icon>,
  search:  (p)=> <Icon {...p}><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></Icon>,
  scales:  (p)=> <Icon {...p}><path d="M12 3v18M5 21h14M8 7l-4 7a3 3 0 0 0 6 0L8 7l8-2"/></Icon>,
  brain:   (p)=> <Icon {...p}><path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 3 3 3 0 0 0 1 4 3 3 0 0 0 3 4 2.5 2.5 0 0 0 3-1V5a2 2 0 0 0-2-1ZM15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 3 3 3 0 0 1-1 4 3 3 0 0 1-3 4 2.5 2.5 0 0 1-3-1"/></Icon>,
  target:  (p)=> <Icon {...p}><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1" fill="currentColor" stroke="none"/></Icon>,
  clock:   (p)=> <Icon {...p}><circle cx="12" cy="12" r="8.5"/><path d="M12 7v5l3.5 2"/></Icon>,
  database:(p)=> <Icon {...p}><ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6"/></Icon>,
  download:(p)=> <Icon {...p}><path d="M12 4v11m0 0 4-4m-4 4-4-4"/><path d="M5 19h14"/></Icon>,
  plus:    (p)=> <Icon {...p}><path d="M12 5v14M5 12h14"/></Icon>,
  dot:     (p)=> <Icon {...p} fill={true}><circle cx="12" cy="12" r="4"/></Icon>,
};

window.Icon = Icon;
window.Icons = Icons;
