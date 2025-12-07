import React, { useState, useEffect, useRef } from 'react';
import { MOCK_USERS } from '../constants';
import { UserProfile, TestResult, SystemEvent } from '../types';
import { Shield, GraduationCap, School, LogIn, Fingerprint, Mic, Hand, User, Activity, Database, CheckCircle2, Lock, LayoutDashboard, BarChart3, BookOpen, Settings, Bell, LogOut, Smartphone, Cpu, Play, Terminal, Wifi, Check, XCircle, AlertTriangle, Command } from 'lucide-react';

// Speech Recognition Type Definition
declare global {
  interface Window {
    webkitSpeechRecognition: any;
    SpeechRecognition: any;
  }
}

interface DashboardContainerProps {
  children: React.ReactNode;
  isMobileMode: boolean;
}

const DashboardContainer: React.FC<DashboardContainerProps> = ({ children, isMobileMode }) => (
  <div className={`transition-all duration-500 ease-in-out bg-slate-50 h-full w-full flex ${isMobileMode ? 'items-center justify-center p-8 bg-slate-800' : ''}`}>
      <div className={`flex flex-col bg-slate-50 transition-all duration-500 ${isMobileMode ? 'w-[375px] h-[812px] rounded-[3rem] border-[8px] border-slate-900 shadow-2xl overflow-hidden relative' : 'w-full h-full'}`}>
          {isMobileMode && <div className="absolute top-0 left-1/2 -translate-x-1/2 w-40 h-6 bg-slate-900 rounded-b-xl z-20"></div>}
          {children}
      </div>
  </div>
);

export const PlatformAccess: React.FC = () => {
  const [activeRole, setActiveRole] = useState<'student' | 'teacher' | 'admin'>('student');
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(null);
  const [authMethod, setAuthMethod] = useState<'password' | 'biometric' | 'voice' | 'gesture'>('password');
  const [loginStep, setLoginStep] = useState<'role' | 'credentials' | 'dashboard'>('role');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // Advanced Features State
  const [isMobileMode, setIsMobileMode] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [adminTab, setAdminTab] = useState<'health' | 'infra' | 'testing'>('health');
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [isRunningTests, setIsRunningTests] = useState(false);
  const [systemEvents, setSystemEvents] = useState<SystemEvent[]>([]);
  const [recognition, setRecognition] = useState<any>(null);

  const availableUsers = MOCK_USERS.filter(u => u.role === activeRole);

  // --- VOICE NAVIGATION INIT ---
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognitionInstance = new SpeechRecognition();
        recognitionInstance.continuous = false;
        recognitionInstance.lang = 'en-US';
        recognitionInstance.interimResults = false;

        recognitionInstance.onresult = (event: any) => {
            const transcript = event.results[0][0].transcript.toLowerCase();
            handleVoiceCommand(transcript);
            setIsListening(false);
        };

        recognitionInstance.onerror = (event: any) => {
            console.error("Speech recognition error", event.error);
            setIsListening(false);
        };
        
        recognitionInstance.onend = () => setIsListening(false);

        setRecognition(recognitionInstance);
    }
  }, [currentUser]);

  // --- REAL-TIME EVENT TICKER ---
  useEffect(() => {
      if (!currentUser) return;
      const eventTypes: SystemEvent['type'][] = ['info', 'success', 'warning'];
      const messages = [
          "Syncing local database shard...",
          "AI Model v2.4 inference complete (12ms)",
          "Received Federated Learning update from Node #42",
          "Websocket connection established",
          "Accessibility check passed (WCAG 2.1)",
          "Background data backup started",
          "New student interaction logged",
          "Latency spike detected in region: Tigray_East"
      ];

      const interval = setInterval(() => {
          const newEvent: SystemEvent = {
              id: Date.now().toString(),
              timestamp: new Date().toLocaleTimeString(),
              type: eventTypes[Math.floor(Math.random() * eventTypes.length)],
              message: messages[Math.floor(Math.random() * messages.length)]
          };
          setSystemEvents(prev => [newEvent, ...prev].slice(0, 5));
      }, 3500);

      return () => clearInterval(interval);
  }, [currentUser]);

  const handleVoiceCommand = (command: string) => {
      // Simple command mapping
      const addLog = (msg: string) => setSystemEvents((prev) => {
          const newEvent: SystemEvent = {
              id: Date.now().toString(),
              timestamp: new Date().toLocaleTimeString(),
              type: 'success',
              message: `Voice Command: "${msg}"`
          };
          return [newEvent, ...prev].slice(0, 5);
      });

      if (command.includes('dashboard') || command.includes('home')) {
          setAdminTab('health');
          addLog("Navigating to Dashboard");
      } else if (command.includes('test') || command.includes('testing')) {
          setAdminTab('testing');
          addLog("Opening Test Suite");
      } else if (command.includes('data') || command.includes('infrastructure') || command.includes('sql')) {
          setAdminTab('infra');
          addLog("Opening Infrastructure Console");
      } else if (command.includes('run') && command.includes('test')) {
          runSystemTests();
          addLog("Running Tests");
      } else if (command.includes('logout') || command.includes('sign out')) {
          handleLogout();
      } else if (command.includes('mobile') || command.includes('phone')) {
          setIsMobileMode(!isMobileMode);
          addLog("Toggling Mobile Mode");
      } else {
          addLog(`Unrecognized command: ${command}`);
      }
  };

  const toggleListening = () => {
      if (recognition) {
          if (isListening) recognition.stop();
          else recognition.start();
          setIsListening(!isListening);
      } else {
          alert("Speech recognition not supported in this browser.");
      }
  };

  const runSystemTests = async () => {
      setIsRunningTests(true);
      setAdminTab('testing');
      const suite: TestResult[] = [
          { id: 't1', name: 'AuthService: Token Validation', level: 'unit', status: 'pending' },
          { id: 't2', name: 'SQL Model: Schema Integrity', level: 'integration', status: 'pending' },
          { id: 't3', name: 'AI Model: Inference Latency < 100ms', level: 'system', status: 'pending' },
          { id: 't4', name: 'UI: Accessibility ARIA Labels', level: 'unit', status: 'pending' },
          { id: 't5', name: 'Federated Learning: Weight Aggregation', level: 'system', status: 'pending' },
      ];
      setTestResults(suite);

      // Simulate execution
      for (let i = 0; i < suite.length; i++) {
          await new Promise(resolve => setTimeout(resolve, 800)); // Simulate test time
          setTestResults(prev => prev.map((t, idx) => {
              if (idx === i) return { ...t, status: 'passed', duration: `${Math.floor(Math.random() * 50) + 10}ms` };
              return t;
          }));
      }
      setIsRunningTests(false);
  };

  const handleLogin = () => {
      setIsLoading(true);
      setTimeout(() => {
          const user = availableUsers.find(u => u.username === username) || availableUsers[0];
          if (user) {
              setCurrentUser(user);
              setLoginStep('dashboard');
          }
          setIsLoading(false);
      }, 1500);
  };

  const handleBiometricLogin = () => {
      setIsLoading(true);
      setTimeout(() => {
          const user = availableUsers.find(u => u.biometricEnabled) || availableUsers[0];
          setCurrentUser(user);
          setLoginStep('dashboard');
          setIsLoading(false);
      }, 2000);
  };

  const handleLogout = () => {
      setCurrentUser(null);
      setLoginStep('role');
      setUsername('');
      setPassword('');
      setIsMobileMode(false);
  };

  // --- RENDER HELPERS ---
  const renderLoginScreen = () => (
      <div className="flex flex-col items-center justify-center min-h-[600px] w-full max-w-md mx-auto p-6 animate-in fade-in zoom-in-95 duration-500">
          <div className="mb-8 text-center">
              <div className="w-16 h-16 bg-indigo-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg shadow-indigo-200">
                  <Lock className="text-white w-8 h-8" />
              </div>
              <h2 className="text-2xl font-bold text-slate-900">AISLE Platform Access</h2>
              <p className="text-slate-500 mt-2">Secure Unified Login Portal</p>
          </div>

          {/* Role Tabs */}
          <div className="grid grid-cols-3 gap-2 bg-slate-100 p-1 rounded-xl w-full mb-8">
              <button onClick={() => setActiveRole('student')} className={`py-2 px-4 rounded-lg text-sm font-semibold transition-all ${activeRole === 'student' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}>Student</button>
              <button onClick={() => setActiveRole('teacher')} className={`py-2 px-4 rounded-lg text-sm font-semibold transition-all ${activeRole === 'teacher' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}>Teacher</button>
              <button onClick={() => setActiveRole('admin')} className={`py-2 px-4 rounded-lg text-sm font-semibold transition-all ${activeRole === 'admin' ? 'bg-white text-indigo-600 shadow-sm' : 'text-slate-500 hover:text-slate-700'}`}>Admin</button>
          </div>

          {/* Icon Differentiator */}
          <div className="mb-6">
              {activeRole === 'student' && <GraduationCap size={48} className="text-emerald-500 mx-auto" />}
              {activeRole === 'teacher' && <School size={48} className="text-amber-500 mx-auto" />}
              {activeRole === 'admin' && <Shield size={48} className="text-rose-500 mx-auto" />}
          </div>

          {/* Input Fields */}
          <div className="w-full space-y-4">
              <div className="space-y-2">
                  <label className="text-xs font-bold text-slate-500 uppercase">Username</label>
                  <div className="relative">
                      <User className="absolute left-3 top-2.5 text-slate-400 w-4 h-4" />
                      <input 
                        type="text" 
                        value={username} 
                        onChange={e => setUsername(e.target.value)} 
                        className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none text-sm" 
                        placeholder={availableUsers[0]?.username || "username"}
                      />
                  </div>
              </div>
              <div className="space-y-2">
                  <label className="text-xs font-bold text-slate-500 uppercase">Password</label>
                  <div className="relative">
                      <Lock className="absolute left-3 top-2.5 text-slate-400 w-4 h-4" />
                      <input 
                        type="password" 
                        value={password} 
                        onChange={e => setPassword(e.target.value)} 
                        className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-indigo-500 outline-none text-sm" 
                        placeholder="••••••••" 
                      />
                  </div>
              </div>

              <button 
                onClick={handleLogin} 
                disabled={isLoading}
                className="w-full bg-indigo-600 text-white py-2.5 rounded-lg font-bold shadow-lg shadow-indigo-200 hover:bg-indigo-700 transition-all flex items-center justify-center gap-2"
              >
                  {isLoading ? <span className="animate-pulse">Verifying...</span> : <><LogIn size={18} /> Sign In</>}
              </button>
          </div>

          {/* Multi-Modal Options */}
          <div className="mt-8 w-full">
              <div className="relative flex py-2 items-center">
                  <div className="flex-grow border-t border-slate-200"></div>
                  <span className="flex-shrink-0 mx-4 text-gray-400 text-xs uppercase tracking-widest">Alternative Login</span>
                  <div className="flex-grow border-t border-slate-200"></div>
              </div>
              
              <div className="grid grid-cols-3 gap-4 mt-4">
                  <button onClick={handleBiometricLogin} className="flex flex-col items-center gap-2 p-3 border border-slate-200 rounded-xl hover:bg-slate-50 transition-colors group">
                      <Fingerprint size={24} className="text-slate-400 group-hover:text-indigo-600" />
                      <span className="text-[10px] font-medium text-slate-500">Biometric</span>
                  </button>
                  <button className="flex flex-col items-center gap-2 p-3 border border-slate-200 rounded-xl hover:bg-slate-50 transition-colors group">
                      <Mic size={24} className="text-slate-400 group-hover:text-indigo-600" />
                      <span className="text-[10px] font-medium text-slate-500">Voice</span>
                  </button>
                  <button className="flex flex-col items-center gap-2 p-3 border border-slate-200 rounded-xl hover:bg-slate-50 transition-colors group">
                      <Hand size={24} className="text-slate-400 group-hover:text-indigo-600" />
                      <span className="text-[10px] font-medium text-slate-500">Gesture</span>
                  </button>
              </div>
          </div>
      </div>
  );

  const renderDashboard = () => {
      if (!currentUser) return null;

      return (
          <DashboardContainer isMobileMode={isMobileMode}>
              <div className="flex h-full w-full bg-slate-50 relative">
                  {/* Sidebar (Hidden on Mobile unless toggled - simplified here to always show on desktop, shrunk on mobile) */}
                  <div className={`${isMobileMode ? 'w-16 items-center' : 'w-64'} bg-slate-900 text-slate-300 flex flex-col p-4 transition-all duration-300`}>
                      <div className={`flex items-center gap-3 mb-8 ${isMobileMode ? 'justify-center' : 'px-2'}`}>
                          <div className="w-8 h-8 bg-indigo-500 rounded-lg flex items-center justify-center text-white font-bold flex-shrink-0">A</div>
                          {!isMobileMode && <span className="font-bold text-white tracking-wide">AISLE</span>}
                      </div>
                      
                      <div className="space-y-1 flex-1">
                          {!isMobileMode && <div className="px-2 py-2 text-xs font-bold uppercase text-slate-500">Menu</div>}
                          <button onClick={() => setAdminTab('health')} className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg ${adminTab === 'health' ? 'bg-indigo-600/20 text-indigo-400' : 'hover:bg-slate-800'} font-medium`}>
                              <LayoutDashboard size={18}/> {!isMobileMode && "Dashboard"}
                          </button>
                          
                          {currentUser.role === 'admin' && (
                              <>
                                <button onClick={() => setAdminTab('infra')} className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg ${adminTab === 'infra' ? 'bg-indigo-600/20 text-indigo-400' : 'hover:bg-slate-800'} transition-colors`}><Database size={18}/> {!isMobileMode && "Infrastructure"}</button>
                                <button onClick={() => setAdminTab('testing')} className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg ${adminTab === 'testing' ? 'bg-indigo-600/20 text-indigo-400' : 'hover:bg-slate-800'} transition-colors`}><CheckCircle2 size={18}/> {!isMobileMode && "System Tests"}</button>
                              </>
                          )}
                      </div>

                      {/* Profile Snippet */}
                      <div className="mt-auto pt-4 border-t border-slate-700">
                          <div className="flex items-center gap-3 justify-center">
                              <div className="w-8 h-8 rounded-full bg-indigo-500 flex items-center justify-center text-white font-bold text-xs flex-shrink-0">
                                  {currentUser.name.charAt(0)}
                              </div>
                              {!isMobileMode && (
                                  <div className="flex-1 min-w-0">
                                      <p className="text-xs font-medium text-white truncate">{currentUser.name}</p>
                                      <p className="text-[10px] text-slate-500 truncate capitalize">{currentUser.role}</p>
                                  </div>
                              )}
                              <button onClick={handleLogout} className="text-slate-400 hover:text-white"><LogOut size={16}/></button>
                          </div>
                      </div>
                  </div>

                  {/* Main Content Area */}
                  <div className="flex-1 overflow-y-auto overflow-x-hidden flex flex-col">
                      {/* Header */}
                      <header className="bg-white border-b border-slate-200 px-6 py-4 flex justify-between items-center sticky top-0 z-10">
                          <div>
                              <h1 className={`${isMobileMode ? 'text-lg' : 'text-2xl'} font-bold text-slate-900`}>{!isMobileMode && 'Welcome back, '}{currentUser.name.split(' ')[0]}</h1>
                              {!isMobileMode && <p className="text-sm text-slate-500">{currentUser.role} portal</p>}
                          </div>
                          <div className="flex gap-2">
                              <button onClick={toggleListening} className={`p-2 rounded-full border shadow-sm transition-all ${isListening ? 'bg-red-50 border-red-200 text-red-600 animate-pulse' : 'bg-white border-slate-200 text-slate-500 hover:text-indigo-600'}`} title="Voice Control">
                                  <Mic size={20} />
                              </button>
                              {!isMobileMode && (
                                  <button onClick={() => setIsMobileMode(!isMobileMode)} className={`p-2 rounded-full border shadow-sm transition-all ${isMobileMode ? 'bg-indigo-50 border-indigo-200 text-indigo-600' : 'bg-white border-slate-200 text-slate-500 hover:text-indigo-600'}`} title="Mobile Simulation">
                                      <Smartphone size={20} />
                                  </button>
                              )}
                              <button className="p-2 bg-white rounded-full border border-slate-200 text-slate-500 hover:text-indigo-600 shadow-sm relative">
                                  <Bell size={20} />
                                  <span className="absolute top-0 right-0 w-2.5 h-2.5 bg-red-500 rounded-full border-2 border-white"></span>
                              </button>
                          </div>
                      </header>

                      <div className="p-6 flex-1">
                          {/* VIEW: DASHBOARD (Overview) */}
                          {adminTab === 'health' && (
                              <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
                                  {/* Voice Command Hint */}
                                  <div className="mb-6 bg-indigo-50 border border-indigo-100 rounded-lg p-3 flex items-center gap-3">
                                      <div className="bg-indigo-600 text-white p-1.5 rounded-full"><Command size={14}/></div>
                                      <p className="text-xs text-indigo-800 font-medium">Voice Command Enabled. Try saying: <em>"Run System Tests"</em>, <em>"Open Database"</em>, or <em>"Toggle Mobile Mode"</em>.</p>
                                  </div>

                                  <div className={`grid ${isMobileMode ? 'grid-cols-1' : 'grid-cols-1 md:grid-cols-3'} gap-6 mb-8`}>
                                      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                                          <div className="flex justify-between items-start mb-4">
                                              <div className="p-2 bg-indigo-50 rounded-lg text-indigo-600"><Activity size={24}/></div>
                                              <span className="text-xs font-bold text-emerald-600 bg-emerald-50 px-2 py-1 rounded">+12%</span>
                                          </div>
                                          <h3 className="text-2xl font-bold text-slate-900 mb-1">
                                              {currentUser.role === 'admin' ? '99.9%' : currentUser.role === 'teacher' ? '85%' : '92/100'}
                                          </h3>
                                          <p className="text-xs text-slate-500 uppercase font-bold tracking-wide">
                                              {currentUser.role === 'admin' ? 'System Uptime' : currentUser.role === 'teacher' ? 'Class Avg.' : 'Current Grade'}
                                          </p>
                                      </div>
                                      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                                          <div className="flex justify-between items-start mb-4">
                                              <div className="p-2 bg-amber-50 rounded-lg text-amber-600"><Database size={24}/></div>
                                              <span className="text-xs font-bold text-slate-500">Live</span>
                                          </div>
                                          <h3 className="text-2xl font-bold text-slate-900 mb-1">
                                              {currentUser.role === 'admin' ? '1.2TB' : currentUser.role === 'teacher' ? '15' : '4'}
                                          </h3>
                                          <p className="text-xs text-slate-500 uppercase font-bold tracking-wide">
                                              {currentUser.role === 'admin' ? 'Data Processed' : currentUser.role === 'teacher' ? 'Active Students' : 'Pending Tasks'}
                                          </p>
                                      </div>
                                      <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                                          <div className="flex justify-between items-start mb-4">
                                              <div className="p-2 bg-rose-50 rounded-lg text-rose-600"><Shield size={24}/></div>
                                              <span className="text-xs font-bold text-emerald-600 bg-emerald-50 px-2 py-1 rounded">Secure</span>
                                          </div>
                                          <h3 className="text-2xl font-bold text-slate-900 mb-1">Active</h3>
                                          <p className="text-xs text-slate-500 uppercase font-bold tracking-wide">
                                              {currentUser.role === 'admin' ? 'Security Protocol' : 'DPIA Status'}
                                          </p>
                                      </div>
                                  </div>
                              </div>
                          )}

                          {/* VIEW: INFRASTRUCTURE (Admin) */}
                          {adminTab === 'infra' && (
                              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                                  <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2"><Database size={20}/> Infrastructure Console</h2>
                                  
                                  {/* Dynamic SQL Model Mock */}
                                  <div className="bg-slate-900 rounded-lg border border-slate-700 overflow-hidden text-slate-300 font-mono text-xs">
                                      <div className="bg-slate-800 px-4 py-2 border-b border-slate-700 flex justify-between">
                                          <span>SQL Monitor: Primary Shard</span>
                                          <span className="text-emerald-400">Connected</span>
                                      </div>
                                      <div className="p-4 space-y-2">
                                          <div className="flex gap-2"><span className="text-blue-400">SELECT</span> * <span className="text-blue-400">FROM</span> users <span className="text-blue-400">WHERE</span> role = 'student'; <span className="text-slate-500 ml-auto">12ms</span></div>
                                          <div className="flex gap-2"><span className="text-blue-400">UPDATE</span> model_weights <span className="text-blue-400">SET</span> version = 'v2.4'; <span className="text-slate-500 ml-auto">45ms</span></div>
                                          <div className="flex gap-2"><span className="text-blue-400">INSERT INTO</span> logs (event, time) <span className="text-blue-400">VALUES</span> (...); <span className="text-slate-500 ml-auto">8ms</span></div>
                                      </div>
                                  </div>

                                  {/* AI Model Status */}
                                  <div className="bg-white rounded-lg border border-slate-200 p-6">
                                      <div className="flex items-center justify-between mb-4">
                                          <h3 className="font-bold text-slate-800 flex items-center gap-2"><Cpu size={18}/> AI Model Status</h3>
                                          <span className="bg-emerald-100 text-emerald-800 text-xs px-2 py-1 rounded font-bold">Online</span>
                                      </div>
                                      <div className="grid grid-cols-2 gap-4">
                                          <div className="p-4 bg-slate-50 rounded border border-slate-100">
                                              <div className="text-xs text-slate-500 uppercase font-bold mb-1">Inference Latency</div>
                                              <div className="text-xl font-mono font-bold text-indigo-600">12ms</div>
                                          </div>
                                          <div className="p-4 bg-slate-50 rounded border border-slate-100">
                                              <div className="text-xs text-slate-500 uppercase font-bold mb-1">Model Version</div>
                                              <div className="text-xl font-mono font-bold text-slate-700">v2.4.1 (Quantized)</div>
                                          </div>
                                      </div>
                                  </div>
                              </div>
                          )}

                          {/* VIEW: SYSTEM TESTING (Admin) */}
                          {adminTab === 'testing' && (
                              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                                  <div className="flex justify-between items-center">
                                      <h2 className="text-lg font-bold text-slate-800 flex items-center gap-2"><CheckCircle2 size={20}/> System Test Suite</h2>
                                      <button onClick={runSystemTests} disabled={isRunningTests} className="bg-indigo-600 text-white px-4 py-2 rounded-lg text-sm font-bold shadow-sm hover:bg-indigo-700 flex items-center gap-2">
                                          {isRunningTests ? <Activity className="animate-spin" size={16}/> : <Play size={16}/>}
                                          Run Full Suite
                                      </button>
                                  </div>

                                  <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
                                      <div className="grid grid-cols-1 divide-y divide-slate-100">
                                          {testResults.length === 0 ? (
                                              <div className="p-8 text-center text-slate-400">
                                                  <Terminal size={32} className="mx-auto mb-2 opacity-50"/>
                                                  <p className="text-sm">No tests run yet. Start the sequence to validate the platform.</p>
                                              </div>
                                          ) : (
                                              testResults.map(test => (
                                                  <div key={test.id} className="p-4 flex items-center justify-between hover:bg-slate-50 transition-colors">
                                                      <div className="flex items-center gap-4">
                                                          {test.status === 'pending' && <div className="w-5 h-5 rounded-full border-2 border-slate-200"></div>}
                                                          {test.status === 'running' && <Activity className="w-5 h-5 text-indigo-500 animate-spin"/>}
                                                          {test.status === 'passed' && <Check className="w-5 h-5 text-emerald-500"/>}
                                                          {test.status === 'failed' && <XCircle className="w-5 h-5 text-red-500"/>}
                                                          <div>
                                                              <div className="text-sm font-bold text-slate-800">{test.name}</div>
                                                              <div className="text-xs text-slate-500 uppercase tracking-wide">{test.level}</div>
                                                          </div>
                                                      </div>
                                                      <div className="text-xs font-mono text-slate-400">{test.duration || '--'}</div>
                                                  </div>
                                              ))
                                          )}
                                      </div>
                                  </div>
                              </div>
                          )}
                      </div>

                      {/* Real-time Event Ticker */}
                      <div className="bg-slate-900 text-slate-300 text-xs py-2 px-6 border-t border-slate-700 flex items-center gap-4 overflow-hidden whitespace-nowrap">
                          <span className="font-bold text-emerald-400 flex items-center gap-1 uppercase tracking-wider"><Wifi size={12} className="animate-pulse"/> Live System</span>
                          <div className="flex-1 overflow-hidden relative h-4">
                              <div className="absolute animate-slide-up top-0 left-0 w-full">
                                  {systemEvents.map(ev => (
                                      <span key={ev.id} className="mr-8 inline-flex items-center gap-2">
                                          <span className="text-slate-500">[{ev.timestamp}]</span>
                                          <span className={ev.type === 'error' ? 'text-red-400' : ev.type === 'warning' ? 'text-amber-400' : 'text-slate-300'}>{ev.message}</span>
                                      </span>
                                  ))}
                              </div>
                          </div>
                      </div>
                  </div>
              </div>
          </DashboardContainer>
      );
  };

  return (
    <div className="w-full h-full bg-white flex items-center justify-center">
        {loginStep === 'dashboard' ? renderDashboard() : renderLoginScreen()}
    </div>
  );
};