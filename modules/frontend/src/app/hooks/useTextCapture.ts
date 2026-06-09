import { useEffect, useRef, useCallback } from 'react';
import { useSession } from '../context/SessionContext';

interface DeletedSegment {
  text: string;
  deleted_at: number;
}

interface TextOutput {
  session_id: string;
  turn_id: number;
  final_text: string;
  deleted_segments: DeletedSegment[];
}

export function useTextCapture(inputRef: React.RefObject<HTMLTextAreaElement>) {
  const { sessionId, turnId, deletedSegments, addDeletedSegment } = useSession();
  const previousTextRef = useRef('');

  useEffect(() => {
    const element = inputRef.current;
    if (!element) return;

    const handleInput = () => {
      const currentText = element.value;
      const previousText = previousTextRef.current;

      // Detect deletion
      if (currentText.length < previousText.length) {
        const deletedText = previousText.slice(currentText.length);

        // Only track meaningful deletions (more than just a single character)
        if (deletedText.trim().length > 0) {
          const segment: DeletedSegment = {
            text: deletedText,
            deleted_at: Date.now() / 1000,
          };
          addDeletedSegment(segment);
        }
      }

      previousTextRef.current = currentText;
    };

    element.addEventListener('input', handleInput);

    return () => {
      element.removeEventListener('input', handleInput);
    };
  }, [inputRef, addDeletedSegment]);

  const getTextOutput = useCallback(
    (finalText: string): TextOutput => {
      return {
        session_id: sessionId,
        turn_id: turnId,
        final_text: finalText,
        deleted_segments: deletedSegments,
      };
    },
    [sessionId, turnId, deletedSegments]
  );

  return { getTextOutput };
}
