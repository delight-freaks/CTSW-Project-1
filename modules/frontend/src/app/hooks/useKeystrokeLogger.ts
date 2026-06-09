import { useEffect, useCallback } from 'react';
import { useSession } from '../context/SessionContext';

interface KeystrokeEvent {
  type: 'keydown' | 'keyup';
  key: string;
  timestamp: number;
  is_delete: boolean;
}

interface RawKeystrokeOutput {
  session_id: string;
  turn_id: number;
  events: KeystrokeEvent[];
}

export function useKeystrokeLogger(inputRef: React.RefObject<HTMLTextAreaElement>) {
  const { sessionId, turnId, addKeystrokeEvent, keystrokeEvents } = useSession();

  useEffect(() => {
    const element = inputRef.current;
    if (!element) return;

    const handleKeyDown = (e: KeyboardEvent) => {
      const isDelete = e.key === 'Backspace' || e.key === 'Delete';

      const event: KeystrokeEvent = {
        type: 'keydown',
        key: e.key,
        timestamp: Date.now() / 1000, // Unix timestamp in seconds
        is_delete: isDelete,
      };

      addKeystrokeEvent(event);
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      const isDelete = e.key === 'Backspace' || e.key === 'Delete';

      const event: KeystrokeEvent = {
        type: 'keyup',
        key: e.key,
        timestamp: Date.now() / 1000,
        is_delete: isDelete,
      };

      addKeystrokeEvent(event);
    };

    element.addEventListener('keydown', handleKeyDown);
    element.addEventListener('keyup', handleKeyUp);

    return () => {
      element.removeEventListener('keydown', handleKeyDown);
      element.removeEventListener('keyup', handleKeyUp);
    };
  }, [inputRef, addKeystrokeEvent]);

  const getKeystrokeOutput = useCallback((): RawKeystrokeOutput => {
    return {
      session_id: sessionId,
      turn_id: turnId,
      events: keystrokeEvents,
    };
  }, [sessionId, turnId, keystrokeEvents]);

  return { getKeystrokeOutput };
}
