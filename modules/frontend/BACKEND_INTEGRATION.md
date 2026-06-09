# 백엔드 연동 가이드


### 구현된 기능

1. **세션 관리**
   - 앱 시작 시 백엔드에서 `session_id` (UUID v4) 자동 발급
   - 턴 증가 시 백엔드에 새 턴 생성 요청

2. **멀티모달 데이터 전송**
   - 키스트로크 데이터: `POST /keystrokes`
   - 텍스트 데이터: `POST /text`
   - 침묵 이벤트: `POST /silence`

3. **실시간 통신 (WebSocket)**
   - WebSocket 연결: `/ws/{session_id}`
   - LLM 응답 실시간 수신
   - 전송 트리거/침묵 트리거 전달

---
