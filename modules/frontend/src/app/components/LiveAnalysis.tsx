import { useState, useEffect, useRef, useCallback } from 'react';
import { Video, VideoOff } from 'lucide-react';
import { useSession } from '../context/SessionContext';
import { useWebcamContext } from '../context/WebcamContext';
import { API_BASE_URL } from '../utils/api';

interface EmotionScore {
  label: string;
  value: number;
  color: string;
}

interface VisionData {
  face_detected: boolean;
  emotion: string | null;
  confidence: number | null;
  emotion_scores: Record<string, number> | null;
  peak_emotion: string | null;
  peak_confidence: number | null;
  timestamp: number | null;
}

const EMOTION_COLORS: Record<string, string> = {
  sad: 'bg-blue-500',
  angry: 'bg-red-500',
  happy: 'bg-green-500',
  neutral: 'bg-gray-500',
  surprised: 'bg-purple-500',
  fearful: 'bg-orange-500',
  anxious: 'bg-yellow-500',
  disgust: 'bg-pink-500',
};

const EMOTION_LABELS: Record<string, string> = {
  sad: '슬픔',
  angry: '분노',
  happy: '기쁨',
  neutral: '중립',
  surprised: '놀람',
  fearful: '두려움',
  anxious: '불안',
  disgust: '혐오',
};

// 브라우저 프레임을 백엔드로 보내 추론하는 주기. CPU 추론이라 5fps는 무리 → ~1.25fps.
const CAPTURE_INTERVAL_MS = 800;
const CAPTURE_WIDTH = 480; // 전송 프레임 가로 해상도 (MediaPipe 얼굴 검출에 충분)

export function LiveAnalysis() {
  const { sessionId, isSessionReady } = useSession();
  const { stream, isEnabled } = useWebcamContext();

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const inFlightRef = useRef(false);

  const [visionData, setVisionData] = useState<VisionData | null>(null);
  const [error, setError] = useState<string | null>(null);

  // 웹캠 스트림을 video 엘리먼트에 연결 (카메라 화면 표시)
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.srcObject = stream ?? null;
    if (stream) video.play().catch(() => {});
  }, [stream]);

  // 한 프레임을 캡처해 백엔드 추론 엔드포인트로 전송 → 결과로 패널 갱신
  const captureAndInfer = useCallback(async () => {
    if (inFlightRef.current) return; // 이전 요청 진행 중이면 건너뜀 (적체 방지)
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !sessionId || sessionId.startsWith('local-')) return;
    if (!video.videoWidth || video.readyState < 2) return;

    const scale = CAPTURE_WIDTH / video.videoWidth;
    canvas.width = CAPTURE_WIDTH;
    canvas.height = Math.round(video.videoHeight * scale);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.7);

    inFlightRef.current = true;
    try {
      const res = await fetch(`${API_BASE_URL}/sessions/${sessionId}/vision/infer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'ngrok-skip-browser-warning': 'true' },
        body: JSON.stringify({ image_b64: dataUrl }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setVisionData(await res.json());
      setError(null);
    } catch (err) {
      setError('추론 서버 연결 실패');
      console.warn('[LiveAnalysis] infer error:', err);
    } finally {
      inFlightRef.current = false;
    }
  }, [sessionId]);

  // 웹캠이 켜져 있고 세션 준비됐을 때만 캡처 루프 동작
  useEffect(() => {
    if (!isSessionReady || !stream || !sessionId || sessionId.startsWith('local-')) {
      setVisionData(null);
      return;
    }
    const interval = setInterval(captureAndInfer, CAPTURE_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [isSessionReady, stream, sessionId, captureAndInfer]);

  // emotion_scores → 상위 3개 막대
  const topEmotions: EmotionScore[] =
    visionData?.emotion_scores && Object.keys(visionData.emotion_scores).length > 0
      ? Object.entries(visionData.emotion_scores)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 3)
          .map(([label, value]) => ({
            label,
            value: value * 100,
            color: EMOTION_COLORS[label] ?? 'bg-gray-500',
          }))
      : [];

  const hasFace = visionData?.face_detected ?? false;
  const peakLabel = visionData?.peak_emotion ?? topEmotions[0]?.label;
  const peakValue = topEmotions[0]?.value;

  return (
    <div className="bg-[#2A2A2A] rounded-lg p-4">
      <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
        <Video size={18} className="text-blue-400" />
        실시간 분석
      </h3>

      {/* Camera view */}
      <div className="relative bg-[#1A1A1A] rounded-lg aspect-video mb-3 overflow-hidden flex items-center justify-center">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className={`w-full h-full object-cover ${isEnabled ? '' : 'hidden'}`}
        />
        {!isEnabled && (
          <div className="flex flex-col items-center gap-2 text-gray-500">
            <VideoOff size={40} />
            <span className="text-xs">웹캠 비활성화 — 상단에서 활성화</span>
          </div>
        )}
        <canvas ref={canvasRef} className="hidden" />

        {/* Live / error indicator */}
        {isEnabled && (
          <div className="absolute top-2 right-2">
            {error ? (
              <div className="bg-yellow-500/80 rounded-full w-3 h-3" title={error} />
            ) : (
              <div className="bg-red-500 rounded-full w-3 h-3 animate-pulse" />
            )}
          </div>
        )}

        {/* Emotion overlay */}
        {isEnabled && (
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-3">
            {error ? (
              <div className="text-xs text-yellow-400">{error}</div>
            ) : !visionData ? (
              <div className="text-xs text-gray-400">추론 중...</div>
            ) : !hasFace ? (
              <div className="text-xs text-gray-400">얼굴 미감지</div>
            ) : peakLabel ? (
              <div className="text-xs text-gray-300">
                Peak:{' '}
                <span className="text-white font-semibold">
                  {EMOTION_LABELS[peakLabel] ?? peakLabel}
                </span>
                {peakValue != null && (
                  <> <span className="text-blue-400">{peakValue.toFixed(0)}%</span></>
                )}
              </div>
            ) : null}
          </div>
        )}
      </div>

      {/* Emotion bars */}
      <div className="space-y-2">
        <div className="text-xs text-gray-400 mb-2">
          감정 분석 (Vision Module · DDAMFN)
          {visionData?.timestamp && (
            <span className="ml-2 text-gray-600">
              {new Date(visionData.timestamp * 1000).toLocaleTimeString('ko-KR')}
            </span>
          )}
        </div>

        {topEmotions.length > 0 ? (
          topEmotions.map((emotion) => (
            <div key={emotion.label}>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-gray-300">
                  {EMOTION_LABELS[emotion.label] ?? emotion.label}
                </span>
                <span className="text-white font-semibold">{emotion.value.toFixed(0)}%</span>
              </div>
              <div className="h-2 bg-[#1A1A1A] rounded-full overflow-hidden">
                <div
                  className={`h-full ${emotion.color} transition-all duration-500`}
                  style={{ width: `${emotion.value}%` }}
                />
              </div>
            </div>
          ))
        ) : (
          <div className="text-xs text-gray-500 py-2 text-center">
            {isEnabled ? '감정 데이터 없음' : '웹캠을 활성화하면 분석이 시작됩니다'}
          </div>
        )}
      </div>
    </div>
  );
}
