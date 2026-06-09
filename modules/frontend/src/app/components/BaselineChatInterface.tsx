import { useState, useEffect, useRef } from 'react';
import { Send } from 'lucide-react';
import { useSession } from '../context/SessionContext';
import { useWebSocket } from '../hooks/useWebSocket';
import { sendTextData } from '../utils/api';

interface Message {
  id: string;
  sender: 'KAI' | 'User';
  text: string;
  timestamp: Date;
}

export function BaselineChatInterface() {
  const {
    sessionId,
    turnId,
    incrementTurn,
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
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { sendMessage: wsSendMessage, lastMessage } = useWebSocket(sessionId);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (!lastMessage) return;

    if (lastMessage.type === 'llm_response') {
      const aiMessage: Message = {
        id: Date.now().toString(),
        sender: 'KAI',
        text: lastMessage.text,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, aiMessage]);
    } else if ((lastMessage as any).type === 'error') {
      const errMessage: Message = {
        id: Date.now().toString(),
        sender: 'KAI',
        text: `[오류] ${(lastMessage as any).text}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errMessage]);
    }
  }, [lastMessage]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = async () => {
    if (!inputValue.trim() || !sessionId) return;

    try {
      await sendTextData({
        session_id: sessionId,
        turn_id: turnId,
        final_text: inputValue,
        deleted_segments: [],
      });

      wsSendMessage({
        type: 'baseline_send_trigger',
        turn_id: turnId,
      });

      const newMessage: Message = {
        id: Date.now().toString(),
        sender: 'User',
        text: inputValue,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, newMessage]);
      setInputValue('');
      await incrementTurn();
    } catch (error) {
      console.error('[Baseline] Failed to send message:', error);
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

      <div className="border-t border-gray-700 p-4">
        {!isSessionReady ? (
          <div className="text-center text-gray-400 py-4">
            세션 초기화 중...
          </div>
        ) : (
          <div className="flex gap-2">
            <textarea
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
