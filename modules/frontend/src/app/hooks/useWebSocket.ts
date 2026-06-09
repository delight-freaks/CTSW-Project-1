import { useEffect, useRef, useState, useCallback } from 'react';
import { API_BASE_URL } from '../utils/api';

interface LLMResponse {
  type: 'llm_response';
  text: string;
}

interface AssembledData {
  type: 'assembled_data' | 'assembled_data_silence';
  note: string;
  data: any;
}

type WebSocketMessage = LLMResponse | AssembledData;

interface UseWebSocketResult {
  isConnected: boolean;
  sendMessage: (message: any) => void;
  lastMessage: WebSocketMessage | null;
  error: string | null;
}

export function useWebSocket(sessionId: string | null): UseWebSocketResult {
  const [isConnected, setIsConnected] = useState(false);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    if (!sessionId) return;

    const wsUrl = API_BASE_URL
      ? API_BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://')
      : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;
    const ws = new WebSocket(`${wsUrl}/ws/${sessionId}`);

    ws.onopen = () => {
      console.log('[WebSocket] Connected to backend');
      setIsConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WebSocketMessage;
        console.log('[WebSocket] Received message:', message);
        setLastMessage(message);
      } catch (err) {
        console.error('[WebSocket] Failed to parse message:', err);
      }
    };

    ws.onerror = (event) => {
      console.error('[WebSocket] Error:', event);
      setError('WebSocket connection error');
    };

    ws.onclose = () => {
      console.log('[WebSocket] Connection closed');
      setIsConnected(false);

      // Attempt to reconnect after 3 seconds
      reconnectTimeoutRef.current = setTimeout(() => {
        console.log('[WebSocket] Attempting to reconnect...');
        connect();
      }, 3000);
    };

    wsRef.current = ws;
  }, [sessionId]);

  useEffect(() => {
    if (sessionId) {
      connect();
    }

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [sessionId, connect]);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      console.log('[WebSocket] Sent message:', message);
    } else {
      console.warn('[WebSocket] Cannot send message - not connected');
    }
  }, []);

  return {
    isConnected,
    sendMessage,
    lastMessage,
    error,
  };
}
