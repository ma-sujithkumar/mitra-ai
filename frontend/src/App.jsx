import { useState, useEffect } from 'react';

import AuthPage from './auth/AuthPage.jsx';
import Dashboard from './screens/Dashboard.jsx';
import FeatureEngineeringPage from './screens/FeatureEngineeringPage.jsx';
import LeaderboardScreen from './screens/LeaderboardScreen.jsx';
import TrainingPage from './screens/TrainingPage.jsx';
import Settings from './screens/Settings.jsx';
import UploadScreen from './screens/UploadScreen.jsx';
import VisualizationPage from './screens/VisualizationPage.jsx';
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
  features: {
    title: 'Feature Engineering',
    sub: 'Step-by-step feature pipeline status and agent reasoning',
    icon: 'layers',
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
  visualize: {
    title: 'Visualizations',
    sub: 'All generated plots grouped by pipeline stage',
    icon: 'chart',
  },
  settings: {
    title: 'Settings',
    sub: 'Runtime status and public defaults',
    icon: 'gear',
  },
};

function App() {
  const [authUser, setAuthUser] = useState(() => {
    const stored = window.localStorage.getItem('mitra.authUser');
    return stored ? JSON.parse(stored) : null;
  });
  const [darkMode, setDarkMode] = useState(() => {
    const stored = window.localStorage.getItem('mitra.darkMode');
    return stored ? JSON.parse(stored) : false;
  });
  const [route, setRoute] = useState('dashboard');
  const [previousRoute, setPreviousRoute] = useState('dashboard');
  const [runState, setRunState] = useState('idle');
  const [incomingDataset, setIncomingDataset] = useState(null);
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

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  function toggleDarkMode() {
    setDarkMode((previous) => {
      const next = !previous;
      window.localStorage.setItem('mitra.darkMode', JSON.stringify(next));
      return next;
    });
  }

  function handleLogout() {
    setAuthUser(null);
    window.localStorage.removeItem('mitra.authUser');
  }

  // Reopen an already-uploaded dataset from the dashboard: make it the active
  // session and hand the record to the Upload screen, which selects it, loads
  // its per-phase progress, and pre-fills the run form from its metadata.
  function handleOpenDataset(run) {
    const normalized = typeof run?.session_id === 'string' ? run.session_id.trim() : '';
    if (!normalized) {
      return;
    }
    setActiveSessionId(normalized);
    window.localStorage.setItem('mitra.activeTrainingSession', normalized);
    setIncomingDataset(run);
    setRoute('upload');
  }

  function handleAuthenticated(user) {
    setAuthUser(user);
    window.localStorage.setItem('mitra.authUser', JSON.stringify(user));
  }

  function go(nextRoute) {
    if (nextRoute !== route) {
      setPreviousRoute(route);
    }
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

  // Manual gate after metadata: set the shared session and land on the Feature
  // Engineering tab (which then runs PipelinePrep and shows live status). This
  // is the stage the web flow previously skipped before jumping to training.
  function enterFeatureEngineering(sessionId) {
    const normalized = typeof sessionId === 'string' ? sessionId.trim() : '';
    if (normalized) {
      setActiveSessionId(normalized);
      window.localStorage.setItem('mitra.activeTrainingSession', normalized);
    }
    setRoute('features');
  }

  // Resume an existing session by routing to the screen that owns the first
  // phase not yet complete. nextPhase === null means every phase is done, so
  // land on the leaderboard. Completed phases are not re-run (the backend skips
  // them and their agents are not re-triggered here).
  function resumeSession(sessionId, nextPhase) {
    const normalized = typeof sessionId === 'string' ? sessionId.trim() : '';
    if (!normalized) {
      return;
    }
    setActiveSessionId(normalized);
    window.localStorage.setItem('mitra.activeTrainingSession', normalized);
    const phaseRoute = {
      validation: 'upload',
      metadata: 'upload',
      feature_engineering: 'features',
      training: 'pipeline',
      evaluation: 'leaderboard',
    };
    const targetRoute = nextPhase ? (phaseRoute[nextPhase] || 'features') : 'leaderboard';
    if (targetRoute === 'pipeline') {
      setRunState('running');
    }
    setRoute(targetRoute);
  }

  const screens = {
    dashboard: <Dashboard go={go} onOpenDataset={handleOpenDataset} startRun={startRun} />,
    features: <FeatureEngineeringPage activeSessionId={activeSessionId} go={go} startRun={startRun} />,
    pipeline: (
      <TrainingPage
        activeSessionId={activeSessionId}
        go={go}
        runState={runState}
        setActiveSessionId={setActiveSessionId}
        setRunState={setRunState}
      />
    ),
    leaderboard: <LeaderboardScreen activeSessionId={activeSessionId} go={go} startRun={startRun} />,
    visualize: <VisualizationPage activeSessionId={activeSessionId} />,
    settings: (
      <Settings
        activeSessionId={activeSessionId}
        backRoute={previousRoute}
        go={go}
        llmSettings={llmSettings}
        llmSmokeStatus={llmSmokeStatus}
        setLlmSettings={setLlmSettings}
        setLlmSmokeStatus={setLlmSmokeStatus}
      />
    ),
  };

  // Auth gate: render the login/signup page until a user is authenticated.
  if (!authUser) {
    return <AuthPage onAuthenticated={handleAuthenticated} />;
  }

  return (
    <div className="app">
      <Sidebar authUser={authUser} go={go} onLogout={handleLogout} route={route} runState={runState} />
      <main className="workspace">
        <TopBar darkMode={darkMode} icon={meta.icon} onToggleDark={toggleDarkMode} sub={meta.sub} title={meta.title} />
        <div className="screen-frame">
          {/* UploadScreen stays mounted across navigation so the selected
              file, form inputs, and validation/metadata results persist until
              the user picks a different file. Hidden (not unmounted) when the
              active route is not upload. */}
          <div className={route === 'upload' ? undefined : 'screen-hidden'}>
            <UploadScreen
              enterFeatureEngineering={enterFeatureEngineering}
              go={go}
              incomingDataset={incomingDataset}
              llmSettings={llmSettings}
              llmSmokeStatus={llmSmokeStatus}
              onIncomingDatasetConsumed={() => setIncomingDataset(null)}
              resumeSession={resumeSession}
              route={route}
              setLlmSettings={setLlmSettings}
              setLlmSmokeStatus={setLlmSmokeStatus}
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
