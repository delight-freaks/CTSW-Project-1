import { useState, useRef, useCallback } from 'react';

interface UseWebcamResult {
  stream: MediaStream | null;
  error: string | null;
  isLoading: boolean;
  isEnabled: boolean;
  enableWebcam: () => Promise<void>;
  disableWebcam: () => void;
}

export function useWebcam(): UseWebcamResult {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isEnabled, setIsEnabled] = useState(false);
  const mounted = useRef(true);

  const enableWebcam = useCallback(async () => {
    if (stream) {
      console.log('[Webcam] Already enabled');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      });

      if (mounted.current) {
        setStream(mediaStream);
        setError(null);
        setIsEnabled(true);
        console.log('[Webcam] Stream initialized successfully');
      } else {
        mediaStream.getTracks().forEach((track) => track.stop());
      }
    } catch (err) {
      if (mounted.current) {
        const errorMessage =
          err instanceof Error ? err.message : 'Failed to access webcam';
        setError(errorMessage);
        setIsEnabled(false);
        console.error('[Webcam] Failed to initialize:', err);
      }
    } finally {
      if (mounted.current) {
        setIsLoading(false);
      }
    }
  }, [stream]);

  const disableWebcam = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
      setIsEnabled(false);
      console.log('[Webcam] Stream stopped');
    }
  }, [stream]);

  return { stream, error, isLoading, isEnabled, enableWebcam, disableWebcam };
}
