import { useRef, useCallback } from 'react';

export const useKeystrokeLogger = (backendUrl) => {
  const keyEventsBuffer = useRef([]);
  const snapshotsBuffer = useRef([]);

  const handleInput = useCallback((e) => {
    const timestamp = Math.round(performance.timeOrigin + performance.now());
    const textSnapshot = e.target.value;

    snapshotsBuffer.current.push({ timestamp, textSnapshot });
    console.log(`[Input] 스냅샷 저장: "${textSnapshot}"`);
  }, []);

  const handleKeyEvent = useCallback((e, type) => {
    const timestamp = Math.round(performance.timeOrigin + performance.now());

    const eventData = {
      type: type,
      code: e.code,
      timestamp: timestamp,
      isComposing: e.nativeEvent.isComposing
    };

    keyEventsBuffer.current.push(eventData);

    const color = type === 'keydown' ? 'color: #1e90ff; font-weight: bold;' : 'color: #ff7f50; font-weight: bold;';
    console.log(`%c [${type.padEnd(7)}] Code: ${e.code.padEnd(10)} | Time: ${timestamp} | Composing: ${e.nativeEvent.isComposing}`, color);
  }, []);

  const sendData = useCallback(async () => {
    if (keyEventsBuffer.current.length === 0 && snapshotsBuffer.current.length === 0) {
      console.log('[전송] 버퍼가 비어 전송하지 않습니다.');
      return;
    }

    const payload = {
      keyEvents: keyEventsBuffer.current,
      snapshots: snapshotsBuffer.current,
      sentAt: Math.round(performance.timeOrigin + performance.now())
    };

    const allWords = payload.snapshots.map(s => `"${s.textSnapshot}"`);
    console.log('[저장된 전체 단어 기록]\n', allWords.join(' -> '));

    try {
      await fetch(backendUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      keyEventsBuffer.current = [];
      snapshotsBuffer.current = [];
      console.log('[전송 완료] 버퍼 초기화 됨');
    } catch (e) {
      console.error('[전송 실패]', e);
    }
  }, [backendUrl]);

  return { handleKeyEvent, handleInput, sendData };
};
