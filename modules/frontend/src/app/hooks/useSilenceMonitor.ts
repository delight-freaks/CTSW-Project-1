import { useEffect, useRef, useState } from 'react';
import { useSession } from '../context/SessionContext';

const POLL_INTERVAL_MS = 200; // 0.2초
const SILENCE_THRESHOLD_SEC = 8; // 8초

interface SilenceEvent {
  session_id: string;
  turn_id: number;
  type: 'silence_event';
  silence_duration_sec: number;
  last_keystroke_at: number;
  context: 'after_llm_response' | 'mid_typing';
  timestamp: number;
}

interface UseSilenceMonitorProps {
  inputRef: React.RefObject<HTMLTextAreaElement>;
  onSilenceDetected?: (event: SilenceEvent) => void;
}

export function useSilenceMonitor({ inputRef, onSilenceDetected }: UseSilenceMonitorProps) {
  const { sessionId, turnId, setSilenceTime, lastLLMResponseTime } = useSession();
  const lastInputAtRef = useRef(Date.now());
  const previousTextRef = useRef('');
  const silenceFiredRef = useRef(false);
  const [lastSilenceEvent, setLastSilenceEvent] = useState<SilenceEvent | null>(null);

  useEffect(() => {
    const interval = setInterval(() => {
      const element = inputRef.current;
      if (!element) return;

      const currentText = element.value;
      const now = Date.now();

      if (currentText !== previousTextRef.current) {
        // 입력 변화 있음 — 타이머 리셋
        lastInputAtRef.current = now;
        silenceFiredRef.current = false;
        previousTextRef.current = currentText;
        setSilenceTime(0);
      } else {
        // 입력 변화 없음 — 침묵 시간 누적
        const silenceSec = (now - lastInputAtRef.current) / 1000;
        setSilenceTime(silenceSec);

        if (silenceSec >= SILENCE_THRESHOLD_SEC && !silenceFiredRef.current) {
          silenceFiredRef.current = true; // 중복 발송 방지

          // context 판단
          let context: 'after_llm_response' | 'mid_typing' = 'mid_typing';

          if (lastLLMResponseTime !== null) {
            const timeSinceResponse = (now - lastLLMResponseTime) / 1000;
            // LLM 응답 후 10초 이내라면 after_llm_response로 판단
            if (timeSinceResponse < 10) {
              context = 'after_llm_response';
            }
          }

          const silenceEvent: SilenceEvent = {
            session_id: sessionId,
            turn_id: turnId,
            type: 'silence_event',
            silence_duration_sec: silenceSec,
            last_keystroke_at: lastInputAtRef.current / 1000,
            context,
            timestamp: now / 1000,
          };

          setLastSilenceEvent(silenceEvent);

          // 콜백 호출
          if (onSilenceDetected) {
            onSilenceDetected(silenceEvent);
          }

          console.log('[Silence Monitor] Silence detected:', silenceEvent);
        }
      }
    }, POLL_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [inputRef, sessionId, turnId, setSilenceTime, lastLLMResponseTime, onSilenceDetected]);

  return { lastSilenceEvent };
}
