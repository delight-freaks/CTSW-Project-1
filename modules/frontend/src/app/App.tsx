import { ChatInterface } from './components/ChatInterface';
import { StatusBar } from './components/StatusBar';
import { LiveAnalysis } from './components/LiveAnalysis';
import { SessionInfo } from './components/SessionInfo';
import { MentalWellness } from './components/MentalWellness';
import { SessionProvider } from './context/SessionContext';
import { WebcamProvider } from './context/WebcamContext';
import { Brain, AlertTriangle } from 'lucide-react';
import { useDevMode } from './hooks/useDevMode';

export default function App() {
  const isDevMode = useDevMode();

  return (
    <SessionProvider>
      <WebcamProvider>
      <div className="size-full bg-[#1E1E1E] flex flex-col">
        {/* Developer Mode Banner */}
        {isDevMode && (
          <div className="bg-yellow-500/10 border-b border-yellow-500/30 px-4 py-2">
            <div className="flex items-center justify-center gap-2">
              <AlertTriangle size={16} className="text-yellow-400" />
              <span className="text-yellow-400 text-sm font-medium">개발자 모드</span>
              <span className="text-yellow-400/70 text-xs">
                (디버그 패널 활성화 - URL에 ?dev=true 제거 시 숨김)
              </span>
            </div>
          </div>
        )}

        {/* Header */}
        <header className="bg-[#252525] border-b border-gray-700 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 rounded-lg p-2">
              <Brain size={24} className="text-white" />
            </div>
            <div>
              <h1 className="text-white text-xl font-bold">KAI</h1>
              <p className="text-gray-400 text-sm">Multimodal AI Mental Counseling Platform</p>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="flex-1 flex min-h-0">
          {/* Chat Area */}
          <div className={`${isDevMode ? 'flex-[7]' : 'flex-1'} flex flex-col min-w-0`}>
            <ChatInterface />
            {isDevMode && <StatusBar />}
          </div>

          {/* Debug Sidebar - Only visible in dev mode */}
          {isDevMode && (
            <div className="flex-[3] bg-[#252525] border-l border-gray-700 p-4 overflow-y-auto">
              <div className="space-y-4">
                <LiveAnalysis />
                <SessionInfo />
                <MentalWellness />
              </div>
            </div>
          )}
        </div>
      </div>
      </WebcamProvider>
    </SessionProvider>
  );
}
