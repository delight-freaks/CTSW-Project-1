import { Clock, Activity, Ghost } from 'lucide-react';
import { useSession } from '../context/SessionContext';

export function StatusBar() {
  const { silenceTime, deletedSegments, typingRhythm } = useSession();

  const formatSilenceTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}s`;
  };

  const latestDeleted = deletedSegments.length > 0
    ? deletedSegments[deletedSegments.length - 1].text
    : '';

  const isNearThreshold = silenceTime >= 6 && silenceTime < 8;
  const isOverThreshold = silenceTime >= 8;

  return (
    <div className="border-t border-gray-700 bg-[#252525] px-4 py-3">
      <div className="flex items-center gap-6 text-sm">
        <div className="flex items-center gap-2">
          <Clock
            size={16}
            className={`${
              isOverThreshold
                ? 'text-red-400'
                : isNearThreshold
                ? 'text-yellow-400'
                : 'text-blue-400'
            }`}
          />
          <span className="text-gray-400">침묵 타이머:</span>
          <span
            className={`font-mono ${
              isOverThreshold
                ? 'text-red-400 font-bold'
                : isNearThreshold
                ? 'text-yellow-400'
                : 'text-white'
            }`}
          >
            {formatSilenceTime(silenceTime)}
          </span>
          {isOverThreshold && (
            <span className="text-xs text-red-400">(임계값 초과)</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <Activity size={16} className="text-blue-400" />
          <span className="text-gray-400">타이핑 리듬:</span>
          <div className="flex items-center gap-0.5 h-4">
            {typingRhythm.map((height, i) => (
              <div
                key={i}
                className="w-1.5 bg-blue-400 rounded-full transition-all duration-200"
                style={{ height: `${Math.max(height * 2, 4)}px` }}
              />
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2 flex-1 min-w-0">
          <Ghost size={16} className="text-gray-500 flex-shrink-0" />
          <span className="text-gray-400 flex-shrink-0">삭제된 텍스트:</span>
          <span className="text-gray-500 italic text-xs opacity-60 truncate">
            {latestDeleted || '(없음)'}
          </span>
        </div>
      </div>
    </div>
  );
}
