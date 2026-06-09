import { createContext, useContext, ReactNode } from 'react';
import { useWebcam } from '../hooks/useWebcam';

/**
 * 하나의 웹캠 스트림을 앱 전체에서 공유한다.
 * ChatInterface(활성화 버튼)와 LiveAnalysis(영상 표시 + 프레임 추론)가
 * 같은 MediaStream을 쓰도록 하기 위함 (getUserMedia 중복 방지).
 */
type WebcamContextValue = ReturnType<typeof useWebcam>;

const WebcamContext = createContext<WebcamContextValue | null>(null);

export function WebcamProvider({ children }: { children: ReactNode }) {
  const webcam = useWebcam();
  return <WebcamContext.Provider value={webcam}>{children}</WebcamContext.Provider>;
}

export function useWebcamContext(): WebcamContextValue {
  const ctx = useContext(WebcamContext);
  if (!ctx) {
    throw new Error('useWebcamContext must be used within a WebcamProvider');
  }
  return ctx;
}
