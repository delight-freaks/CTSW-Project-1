import { useRef, useCallback } from 'react';

export const useKeystrokeLogger = (sessionId = "a3f2c1d4-7e8b-4c2a-9f1d-0b5e3a7c6d2e") => {
  const keyEventsBuffer = useRef([]);
  const turnIdRef = useRef(1);

  const handleKeyEvent = useCallback((e, type) => {
    if (e.key === 'AltGraph' || e.code === 'AltRight') return;

    const timestamp = parseFloat((Date.now() / 1000).toFixed(3));
    const is_delete = e.code === 'Backspace' || e.code === 'Delete';

    let parsedKey = e.key;
    if (e.code.startsWith('Key')) {
      parsedKey = e.code.replace('Key', '').toLowerCase();
    } else if (e.code.startsWith('Digit')) {
      parsedKey = e.code.replace('Digit', '');
    } else {
      parsedKey = e.code;
    }

    const buffer = keyEventsBuffer.current;
    if (type === 'keyup' && buffer.length > 0) {
      const lastEvent = buffer[buffer.length - 1];
      if (lastEvent.type === 'keyup' && lastEvent.key === parsedKey && (timestamp - lastEvent.timestamp) < 0.05) {
        return;
      }
    }

    const eventData = {
      type: type,
      key: parsedKey,
      timestamp: timestamp,
      is_delete: is_delete
    };

    keyEventsBuffer.current.push(eventData);

    const color = type === 'keydown' ? 'color: #1e90ff; font-weight: bold;' : 'color: #ff7f50; font-weight: bold;';
    console.log(`%c [${type.padEnd(7)}] Key: ${parsedKey.padEnd(10)} | Time: ${timestamp.toFixed(3)} | Delete: ${is_delete}`, color);
  }, []);

  const sendData = useCallback(async () => {
    if (keyEventsBuffer.current.length === 0) {
      console.log('[전송] 전송할 이벤트가 없습니다.');
      return;
    }

    const typedSequence = keyEventsBuffer.current
      .filter(e => e.type === 'keydown')
      .map(e => e.key)
      .join(' ');

    console.log(`[디버그] 이번 턴 타이핑 흐름: ${typedSequence}`);

    const payload = {
      session_id: sessionId,
      turn_id: turnIdRef.current,
      events: keyEventsBuffer.current
    };

    console.log(`[전송] Turn ID: ${payload.turn_id} | 수집된 이벤트: ${payload.events.length}개`);
    console.log('[전송] 페이로드 상세:', payload);

    try {
      await fetch('http://localhost:8000/keystrokes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      turnIdRef.current += 1;
    } catch (e) {
      console.error('[전송] 전송 실패:', e);
    } finally {
      keyEventsBuffer.current = [];
    }
  }, [sessionId]);

  return { handleKeyEvent, sendData };
};
