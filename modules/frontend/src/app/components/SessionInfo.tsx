import { useEffect, useState } from 'react';
import { User, Clock, Hash, MessageSquare } from 'lucide-react';
import { useSession } from '../context/SessionContext';

export function SessionInfo() {
  const { sessionId, turnId } = useSession();
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setDuration((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${String(secs).padStart(2, '0')}`;
  };

  return (
    <div className="bg-[#2A2A2A] rounded-lg p-4">
      <h3 className="text-white font-semibold mb-3">현재 세션 정보</h3>

      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <User size={16} className="text-blue-400" />
          <span className="text-gray-400 text-sm">참가자:</span>
          <span className="text-white text-sm">익명 (가명처리됨)</span>
        </div>

        <div className="flex items-center gap-2">
          <Clock size={16} className="text-blue-400" />
          <span className="text-gray-400 text-sm">진행 시간:</span>
          <span className="text-white text-sm font-mono">{formatDuration(duration)}</span>
        </div>

        <div className="flex items-center gap-2">
          <MessageSquare size={16} className="text-blue-400" />
          <span className="text-gray-400 text-sm">현재 턴:</span>
          <span className="text-white text-sm font-mono">{turnId}</span>
        </div>

        <div className="pt-2 border-t border-gray-700">
          <div className="flex items-start gap-2">
            <Hash size={16} className="text-gray-500 mt-0.5" />
            <div className="flex-1">
              <div className="text-gray-400 text-xs mb-1">세션 ID (UUID v4)</div>
              <div className="text-gray-500 text-xs font-mono break-all">{sessionId}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
