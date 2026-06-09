import { Brain } from 'lucide-react';
import { SessionProvider } from './context/SessionContext';
import { BaselineChatInterface } from './components/BaselineChatInterface';

export default function BaselineApp() {
  return (
    <SessionProvider>
      <div className="size-full bg-[#1E1E1E] flex flex-col">
        <header className="bg-[#252525] border-b border-gray-700 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 rounded-lg p-2">
              <Brain size={24} className="text-white" />
            </div>
            <div>
              <h1 className="text-white text-xl font-bold">KAI</h1>
              <p className="text-gray-400 text-sm">AI Mental Counseling Platform</p>
            </div>
          </div>
        </header>

        <div className="flex-1 flex min-h-0">
          <div className="flex-1 flex flex-col min-w-0">
            <BaselineChatInterface />
          </div>
        </div>
      </div>
    </SessionProvider>
  );
}
