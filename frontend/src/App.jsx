import { useState } from 'react';

import Dashboard from './screens/Dashboard.jsx';
import LeaderboardScreen from './screens/LeaderboardScreen.jsx';
import PipelineScreen from './screens/PipelineScreen.jsx';
import Settings from './screens/Settings.jsx';
import UploadScreen from './screens/UploadScreen.jsx';
import Sidebar from './components/Sidebar.jsx';
import TopBar from './components/TopBar.jsx';
import { Icons } from './icons.jsx';

const ROUTE_META = {
  dashboard: {
    title: 'Dashboard',
    sub: 'Runs, agents, and system health at a glance',
    icon: 'grid',
  },
  upload: {
    title: 'New Run',
    sub: 'Upload a dataset and let the agents take over',
    icon: 'upload',
  },
  pipeline: {
    title: 'Live Pipeline',
    sub: 'Eight specialist agents with staged outputs',
    icon: 'flow',
  },
  leaderboard: {
    title: 'Leaderboard',
    sub: 'Ranked model candidates and judge context',
    icon: 'trophy',
  },
  settings: {
    title: 'Settings',
    sub: 'Runtime status and public defaults',
    icon: 'gear',
  },
};

function App() {
  const [route, setRoute] = useState('dashboard');
  const [runState, setRunState] = useState('idle');
  const meta = ROUTE_META[route] || ROUTE_META.dashboard;

  function go(nextRoute) {
    setRoute(nextRoute);
  }

  function startRun() {
    setRunState('running');
    setRoute('pipeline');
  }

  const rightActions = (
    <button className="btn btn-primary" onClick={() => setRoute('upload')} type="button">
      <Icons.plus size={16} />
      New Run
    </button>
  );

  const screens = {
    dashboard: <Dashboard go={go} startRun={startRun} />,
    upload: <UploadScreen go={go} startRun={startRun} />,
    pipeline: (
      <PipelineScreen
        go={go}
        runState={runState}
        setRunState={setRunState}
        startRun={startRun}
      />
    ),
    leaderboard: <LeaderboardScreen startRun={startRun} />,
    settings: <Settings />,
  };

  return (
    <div className="app">
      <Sidebar go={go} route={route} runState={runState} />
      <main className="workspace">
        <TopBar icon={meta.icon} right={rightActions} sub={meta.sub} title={meta.title} />
        <div className="screen-frame">{screens[route]}</div>
      </main>
    </div>
  );
}

export default App;
