import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { createSession, createTurn } from '../utils/api';

interface DeletedSegment {
  text: string;
  deleted_at: number;
}

interface KeystrokeEvent {
  type: 'keydown' | 'keyup';
  key: string;
  timestamp: number;
  is_delete: boolean;
}

interface SessionContextType {
  sessionId: string | null;
  turnId: number;
  incrementTurn: () => Promise<void>;
  silenceTime: number;
  setSilenceTime: (time: number) => void;
  deletedSegments: DeletedSegment[];
  addDeletedSegment: (segment: DeletedSegment) => void;
  keystrokeEvents: KeystrokeEvent[];
  addKeystrokeEvent: (event: KeystrokeEvent) => void;
  clearKeystrokeEvents: () => void;
  typingRhythm: number[];
  updateTypingRhythm: (rhythm: number[]) => void;
  lastLLMResponseTime: number | null;
  setLastLLMResponseTime: (time: number | null) => void;
  isSessionReady: boolean;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

export function SessionProvider({ children }: { children: ReactNode }) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [turnId, setTurnId] = useState(1);
  const [silenceTime, setSilenceTime] = useState(0);
  const [deletedSegments, setDeletedSegments] = useState<DeletedSegment[]>([]);
  const [keystrokeEvents, setKeystrokeEvents] = useState<KeystrokeEvent[]>([]);
  const [typingRhythm, setTypingRhythm] = useState<number[]>([3, 5, 4, 6, 3, 7, 4]);
  const [lastLLMResponseTime, setLastLLMResponseTime] = useState<number | null>(null);
  const [isSessionReady, setIsSessionReady] = useState(false);

  // Initialize session with backend on mount
  useEffect(() => {
    async function initSession() {
      try {
        const response = await createSession();
        setSessionId(response.session_id);
        setTurnId(response.turn_index);
        setIsSessionReady(true);
        console.log('[Session] Backend session created:', response);
      } catch (error) {
        console.error('[Session] Failed to create backend session:', error);
        // Fallback to local UUID if backend is not available
        const fallbackId = 'local-' + Date.now();
        setSessionId(fallbackId);
        setIsSessionReady(true);
        console.warn('[Session] Using local session ID (backend unavailable)');
      }
    }

    initSession();
  }, []);

  const incrementTurn = async () => {
    if (!sessionId) return;

    try {
      // Only call backend if we have a real session (not fallback)
      if (!sessionId.startsWith('local-')) {
        const response = await createTurn(sessionId);
        setTurnId(response.turn_index);
        console.log('[Session] New turn created:', response);
      } else {
        // Fallback: just increment locally
        setTurnId((prev) => prev + 1);
      }
    } catch (error) {
      console.error('[Session] Failed to create turn:', error);
      // Fallback to local increment
      setTurnId((prev) => prev + 1);
    }
  };

  const addDeletedSegment = (segment: DeletedSegment) => {
    setDeletedSegments((prev) => [...prev.slice(-3), segment]);
  };

  const addKeystrokeEvent = (event: KeystrokeEvent) => {
    setKeystrokeEvents((prev) => [...prev, event]);
  };

  const clearKeystrokeEvents = () => {
    setKeystrokeEvents([]);
    setDeletedSegments([]);
  };

  const updateTypingRhythm = (rhythm: number[]) => {
    setTypingRhythm(rhythm);
  };

  // 타이핑 리듬을 키스트로크 이벤트에서 계산
  useEffect(() => {
    if (keystrokeEvents.length >= 2) {
      const recentEvents = keystrokeEvents.slice(-10).filter((e) => e.type === 'keydown');
      if (recentEvents.length >= 2) {
        const intervals = [];
        for (let i = 1; i < recentEvents.length; i++) {
          const iki = (recentEvents[i].timestamp - recentEvents[i - 1].timestamp) * 1000;
          intervals.push(Math.min(iki / 100, 10));
        }
        setTypingRhythm(intervals.slice(-7));
      }
    }
  }, [keystrokeEvents]);

  return (
    <SessionContext.Provider
      value={{
        sessionId,
        turnId,
        incrementTurn,
        silenceTime,
        setSilenceTime,
        deletedSegments,
        addDeletedSegment,
        keystrokeEvents,
        addKeystrokeEvent,
        clearKeystrokeEvents,
        typingRhythm,
        updateTypingRhythm,
        lastLLMResponseTime,
        setLastLLMResponseTime,
        isSessionReady,
      }}
    >
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const context = useContext(SessionContext);
  if (context === undefined) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return context;
}
