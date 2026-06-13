import React, { useState } from 'react';
import './theme.css';
import { ROUTE_META } from './data.js';
import { Sidebar } from './components/Sidebar.jsx';
import { TopBar } from './components/TopBar.jsx';
import { Dashboard } from './screens/Dashboard.jsx';
import { UploadScreen } from './screens/UploadScreen.jsx';
import { Icons } from './icons.jsx';

// Placeholder screens for routes not yet implemented in Epic 1
function PlaceholderScreen({ title }) {
  return (
    <div className="page page-in" style={{ textAlign: 'center', paddingTop: 80 }}>
      <div style={{ fontSize: 40, marginBottom: 16, color: 'var(--ink-4)' }}>
        <Icons.clock size={48} />
      </div>
      <h2 style={{ color: 'var(--ink-2)', fontWeight: 600 }}>{title}</h2>
      <p className="faint" style={{ fontSize: 14 }}>This page is implemented in a later epic.</p>
    </div>
  );
}

export default function App() {
  const [route, setRoute] = useState('dashboard');
  const [runState, setRunState] = useState('idle'); // idle | running | done

  function go(routeName) {
    setRoute(routeName);
  }

  function startRun() {
    setRunState('running');
    setRoute('pipeline');
  }

  const meta = ROUTE_META[route] || ROUTE_META.dashboard;

  const rightActions = (
    <>
      {runState === 'running' && route !== 'pipeline' && (
        <button className="btn btn-ghost btn-sm" onClick={() => go('pipeline')}>
          <span className="spinner" style={{ width: 11, height: 11 }} />
          Run in progress
        </button>
      )}
    </>
  );

  return (
    <div className="app">
      <Sidebar route={route} go={go} runState={runState} />
      <main style={{ display: 'flex', flexDirection: 'column', height: '100%', overflowY: 'auto' }}>
        <TopBar title={meta.title} sub={meta.sub} icon={meta.icon} right={rightActions} />
        <div style={{ flex: 1 }}>
          {route === 'dashboard'   && <Dashboard go={go} startRun={startRun} />}
          {route === 'upload'      && <UploadScreen go={go} />}
          {route === 'pipeline'    && <PlaceholderScreen title="Live Pipeline" />}
          {route === 'leaderboard' && <PlaceholderScreen title="Leaderboard" />}
        </div>
      </main>
    </div>
  );
}
