import { useState, useEffect, useRef } from 'react';
import { Send, Video, VideoOff } from 'lucide-react';
import { useSession } from '../context/SessionContext';
import { useKeystrokeLogger } from '../hooks/useKeystrokeLogger';
import { useTextCapture } from '../hooks/useTextCapture';
import { useSilenceMonitor } from '../hooks/useSilenceMonitor';
import { useWebcamContext } from '../context/WebcamContext';
import { useDevMode } from '../hooks/useDevMode';
import { useWebSocket } from '../hooks/useWebSocket';
import { sendKeystrokeData, sendTextData, sendSilenceEvent } from '../utils/api';

interface Message {
  id: string;
  sender: 'KAI' | 'User';
  text: string;
  timestamp: Date;
}

export function ChatInterface() {
  const isDevMode = useDevMode();
  const {
    sessionId,
    turnId,
    incrementTurn,
    clearKeystrokeEvents,
    setLastLLMResponseTime,
    isSessionReady,
  } = useSession();

  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      sender: 'KAI',
      text: '안녕하세요! 저는 KAI입니다. 오늘 기분이 어떠신가요?',
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 커스텀 훅들 통합
  const { getKeystrokeOutput } = useKeystrokeLogger(inputRef);
  const { getTextOutput } = useTextCapture(inputRef);
  const {
    stream: webcamStream,
    error: webcamError,
    isLoading: webcamLoading,
    isEnabled: webcamEnabled,
    enableWebcam,
    disableWebcam,
  } = useWebcamContext();

  // WebSocket 연결
  const { isConnected: wsConnected, sendMessage: wsSendMessage, lastMessage } = useWebSocket(sessionId);

  // 침묵 모니터 (침묵 감지 시 자동으로 LLM 호출)
  useSilenceMonitor({
    inputRef,
    onSilenceDetected: async (silenceEvent) => {
      console.log('[Chat] Silence event detected:', silenceEvent);

      try {
        // Send silence event to backend
        await sendSilenceEvent(silenceEvent);

        // Send trigger to backend via WebSocket to invoke silence prompt
        // LLM response arrives via WebSocket as 'llm_response' message
        wsSendMessage({
          type: 'silence_trigger',
          turn_id: turnId,
        });
      } catch (error) {
        console.error('[Chat] Failed to handle silence event:', error);
      }

      setLastLLMResponseTime(Date.now());
    },
  });

  // 자동 스크롤
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // WebSocket 메시지 처리 (LLM 응답 수신)
  useEffect(() => {
    if (!lastMessage) return;

    if (lastMessage.type === 'llm_response') {
      // LLM 응답을 채팅에 추가
      const aiMessage: Message = {
        id: Date.now().toString(),
        sender: 'KAI',
        text: lastMessage.text,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, aiMessage]);
      setLastLLMResponseTime(Date.now());
    } else if ((lastMessage as any).type === 'error') {
      // 백엔드 에러 메시지 표시
      const errMessage: Message = {
        id: Date.now().toString(),
        sender: 'KAI',
        text: `[오류] ${(lastMessage as any).text}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errMessage]);
    } else if (lastMessage.type === 'assembled_data' || lastMessage.type === 'assembled_data_silence') {
      // 디버그 모드에서만 assembled data 표시
      if (isDevMode) {
        console.log('[Chat] Assembled data from backend:', lastMessage);
      }
    }
  }, [lastMessage, isDevMode, setLastLLMResponseTime]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = async () => {
    if (!inputValue.trim() || !sessionId) return;

    try {
      // 모든 멀티모달 데이터 수집
      const keystrokeOutput = getKeystrokeOutput();
      const textOutput = getTextOutput(inputValue);

      console.log('[Chat] Sending message with multimodal data:');
      console.log('- Session ID:', sessionId);
      console.log('- Turn ID:', turnId);
      console.log('- Keystroke Output:', keystrokeOutput);
      console.log('- Text Output:', textOutput);

      // Send data to backend
      await Promise.all([
        sendKeystrokeData(keystrokeOutput),
        sendTextData(textOutput),
      ]);

      // Send trigger to backend via WebSocket to invoke LLM
      wsSendMessage({
        type: 'send_trigger',
        turn_id: turnId,
      });

      // UI 업데이트
      const newMessage: Message = {
        id: Date.now().toString(),
        sender: 'User',
        text: inputValue,
        timestamp: new Date(),
      };
      setMessages([...messages, newMessage]);
      setInputValue('');
      clearKeystrokeEvents();
      await incrementTurn();
    } catch (error) {
      console.error('[Chat] Failed to send message:', error);
      // Still update UI even if backend fails
      const errorMessage: Message = {
        id: Date.now().toString(),
        sender: 'KAI',
        text: '죄송합니다. 연결에 문제가 있습니다. 잠시 후 다시 시도해주세요.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Webcam Status Indicator - Only in Dev Mode */}
      {isDevMode && (
        <div className="bg-[#252525] border-b border-gray-700 px-4 py-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {webcamEnabled ? (
                <>
                  <div className="flex items-center gap-2 text-xs">
                    <Video size={14} className="text-green-400" />
                    <span className="text-green-400">웹캠 활성화</span>
                  </div>
                  <button
                    onClick={disableWebcam}
                    className="text-xs text-gray-400 hover:text-white transition-colors"
                  >
                    비활성화
                  </button>
                </>
              ) : (
                <>
                  <div className="flex items-center gap-2 text-xs">
                    <VideoOff size={14} className="text-gray-500" />
                    <span className="text-gray-400">웹캠 비활성화</span>
                  </div>
                  <button
                    onClick={enableWebcam}
                    disabled={webcamLoading}
                    className="text-xs bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-3 py-1 rounded transition-colors"
                  >
                    {webcamLoading ? '연결 중...' : '웹캠 활성화'}
                  </button>
                  {webcamError && (
                    <span className="text-xs text-red-400">
                      (오류: 권한 필요)
                    </span>
                  )}
                </>
              )}
            </div>
            <div className="flex items-center gap-3 text-xs text-gray-500">
              <span>Session: {sessionId?.slice(0, 8)}...</span>
              <span>|</span>
              <span>Turn: {turnId}</span>
              <span>|</span>
              <div className="flex items-center gap-1">
                <div
                  className={`w-2 h-2 rounded-full ${
                    wsConnected ? 'bg-green-400' : 'bg-red-400'
                  }`}
                />
                <span>{wsConnected ? 'Connected' : 'Disconnected'}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.sender === 'User' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[70%] rounded-2xl px-4 py-3 ${
                message.sender === 'User'
                  ? 'bg-blue-600 text-white'
                  : 'bg-[#2A2A2A] text-gray-100'
              }`}
            >
              <div className="text-xs opacity-70 mb-1">{message.sender}</div>
              <div className="text-sm leading-relaxed">{message.text}</div>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-gray-700 p-4">
        {!isSessionReady ? (
          <div className="text-center text-gray-400 py-4">
            세션 초기화 중...
          </div>
        ) : (
          <div className="flex gap-2">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="메시지를 입력하세요... (Enter: 전송, Shift+Enter: 줄바꿈)"
              rows={3}
              disabled={!isSessionReady}
              className="flex-1 bg-[#2A2A2A] text-white rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 placeholder-gray-500 resize-none disabled:opacity-50"
            />
            <button
              onClick={handleSend}
              disabled={!isSessionReady || !inputValue.trim()}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg px-6 py-3 transition-colors flex items-center gap-2 self-end"
            >
              <Send size={18} />
              전송
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
