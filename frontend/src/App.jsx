import { useState } from 'react';

import Dashboard from './screens/Dashboard.jsx';
import LeaderboardScreen from './screens/LeaderboardScreen.jsx';
import TrainingPage from './screens/TrainingPage.jsx';
import Settings from './screens/Settings.jsx';
import UploadScreen from './screens/UploadScreen.jsx';
import Sidebar from './components/Sidebar.jsx';
import TopBar from './components/TopBar.jsx';

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
    title: 'Live Training',
    sub: 'Live Ray model training, metrics, and event logs',
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
  const [activeSessionId, setActiveSessionId] = useState(
    () => window.localStorage.getItem('mitra.activeTrainingSession') || '',
  );
  const [llmSettings, setLlmSettings] = useState({
    provider: 'anthropic',
    model: '',
    gatewayUrl: '',
    apiKey: '',
  });
  // configKey records which exact LLM settings were last smoke-tested, so a
  // prior "passed" result is ignored once any provider/model/key/gateway edit
  // changes the configuration.
  const [llmSmokeStatus, setLlmSmokeStatus] = useState({
    status: 'idle',
    message: '',
    configKey: '',
  });
  const meta = ROUTE_META[route] || ROUTE_META.dashboard;

  function go(nextRoute) {
    setRoute(nextRoute);
  }

  function startRun(sessionId) {
    const normalized = typeof sessionId === 'string' ? sessionId.trim() : '';
    if (normalized) {
      setActiveSessionId(normalized);
      window.localStorage.setItem('mitra.activeTrainingSession', normalized);
    }
    setRunState('running');
    setRoute('pipeline');
  }

  const screens = {
    dashboard: <Dashboard go={go} startRun={startRun} />,
    pipeline: (
      <TrainingPage
        activeSessionId={activeSessionId}
        go={go}
        runState={runState}
        setActiveSessionId={setActiveSessionId}
        setRunState={setRunState}
      />
    ),
    leaderboard: <LeaderboardScreen startRun={startRun} />,
    settings: (
      <Settings
        llmSettings={llmSettings}
        llmSmokeStatus={llmSmokeStatus}
        setLlmSettings={setLlmSettings}
        setLlmSmokeStatus={setLlmSmokeStatus}
      />
    ),
  };

  return (
    <div className="app">
      <Sidebar go={go} route={route} runState={runState} />
      <main className="workspace">
        <TopBar icon={meta.icon} sub={meta.sub} title={meta.title} />
        <div className="screen-frame">
          {/* UploadScreen stays mounted across navigation so the selected
              file, form inputs, and validation/metadata results persist until
              the user picks a different file. Hidden (not unmounted) when the
              active route is not upload. */}
          <div className={route === 'upload' ? undefined : 'screen-hidden'}>
            <UploadScreen
              go={go}
              llmSettings={llmSettings}
              llmSmokeStatus={llmSmokeStatus}
              startRun={startRun}
            />
          </div>
          {route === 'upload' ? null : screens[route]}
        </div>
      </main>
    </div>
  );
}

export default App;
