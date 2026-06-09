# Interface Specification

> **담당:** 박관용 (A)
> **최초 작성일:** 2026-03-15
> **상태:** v0.3.0

각 모듈이 독립적으로 개발될 수 있도록 입출력 형식을 명확히 정의한다. 모든 모듈은 이 문서의 스펙을 준수하여 출력해야 하며, 스펙 변경 시 반드시 이 문서를 먼저 수정하고 팀에 공유한다.

---

## 시스템 데이터 흐름

```
[Frontend (E)]
      |
      |-- 웹캠 스트림 ---------> [Vision Module (C)]      --> vision_output JSON
      |-- 키스트로크 이벤트 ---> [Keystroke Logger (D)]   --> raw keystroke JSON
      |                                                           |
      |                                              [Keystroke Classifier (A)]
      |                                                           |
      |                                              keystroke_output JSON
      |-- 텍스트 폴링 (0.2초) --> [Silence Monitor (E)]  --> silence_event JSON
      |-- 텍스트 이벤트 -------> [Text Capture (A)]      --> text_output JSON

vision_output + keystroke_output + text_output
      |
[Trigger Evaluator (A)] --- silence_event
      |
  트리거 조건 판단
  ├── 전송 버튼 눌림  --> 일반 프롬프트
  └── 침묵 8초 초과  --> 침묵 프롬프트
      |
[Prompt Assembler (A)] --> modules/pipeline/prompt_assembler.py
      |
[LLM Client (A)]       --> modules/pipeline/llm_client.py
      |
[Claude API]           --> 상담 응답 텍스트
```

---

## 세션 메타데이터 — session_id 가명처리 (이재철, D / 이고은, E)

### session_id 생성 규칙

`session_id`는 **세션 시작 시 백엔드(이재철)가 생성·발급**하며, 프론트엔드(이고은)는 발급받은 값을 모든 모듈에 전파한다.

날짜 기반 형식(`sess_20260315_001`)은 타임스탬프로 실제 참가자를 역추적할 수 있어 완전한 비식별화가 아니다. 이를 방지하기 위해 UUID v4를 사용한다.

```python
import uuid
session_id = str(uuid.uuid4())  # 예: "a3f2c1d4-7e8b-4c2a-9f1d-0b5e3a7c6d2e"
```

UUID v4는 완전히 무작위로 생성되므로 생성 시점·장소 등 어떠한 메타데이터도 포함하지 않는다. 이것이 **가명처리(Pseudonymization)**의 표준 방식이다.

### 참가자-세션 매핑 테이블 분리 보관

실제 참가자와 `session_id` 간의 매핑은 별도 접근 제한 파일로 분리 보관한다. 데이터 분석 및 모듈 간 통신에서는 `session_id`만 사용하며, 매핑 테이블은 절대 레포에 포함하지 않는다.

```gitignore
# .gitignore에 등록
participant_session_mapping.csv
participant_session_mapping.json
```

---

## Module 1 — Vision Output (심인영, C)

### 책임 범위

OpenCV로 캡처한 웹캠 프레임에서 MediaPipe로 얼굴을 감지·크롭하고, ResNet으로 감정을 분류하여 아래 형식의 JSON을 출력한다. 출력 주기는 0.2초(5fps)로 한다.

### 출력 스펙

```json
{
  "timestamp": 1710234567.123,
  "face_detected": true,
  "emotion": "sad",
  "confidence": 0.72,
  "emotion_scores": {
    "sad": 0.72,
    "angry": 0.10,
    "happy": 0.05,
    "neutral": 0.08,
    "surprised": 0.03,
    "fearful": 0.02
  },
  "head_pose": {
    "yaw": -12.3,
    "pitch": 5.1,
    "roll": 2.0
  },
  "peak_emotion": "fearful",
  "peak_confidence": 0.74,
  "peak_detected_at": 1710234564.001
}
```

### 필드 정의

| 필드 | 타입 | 설명 |
|---|---|---|
| `timestamp` | float | Unix timestamp (초 단위, 소수점 3자리) |
| `face_detected` | bool | 얼굴 감지 여부 |
| `emotion` | string | 최고 확률 감정 레이블 |
| `confidence` | float | `emotion` 필드 감정의 확률값 (0.0 ~ 1.0) |
| `emotion_scores` | object | 전체 감정 클래스별 확률 분포. 합계는 1.0 |
| `head_pose.yaw` | float | 좌우 회전 (도). 양수: 오른쪽, 음수: 왼쪽 |
| `head_pose.pitch` | float | 상하 회전 (도). 양수: 위, 음수: 아래 |
| `head_pose.roll` | float | 기울기 (도) |
| `peak_emotion` | string | 해당 턴에서 confidence 최고값을 기록한 감정 레이블 |
| `peak_confidence` | float | 해당 감정의 최고 confidence값 |
| `peak_detected_at` | float | peak 감정이 포착된 시점 Unix timestamp |

### 감정 레이블 목록

`sad`, `angry`, `happy`, `neutral`, `surprised`, `fearful`

### face_detected: false 처리

얼굴이 감지되지 않은 경우 나머지 필드는 모두 `null`로 설정한다.

```json
{
  "timestamp": 1710234567.123,
  "face_detected": false,
  "emotion": null,
  "confidence": null,
  "emotion_scores": null,
  "head_pose": null
}
```

프롬프트 조립 단계에서 `face_detected: false`인 경우 "표정 데이터 없음"으로 처리한다.

---

## Module 2 — Keystroke Logger Raw Output (조재현, B)

### 책임 범위

브라우저에서 사용자의 키 입력 이벤트를 실시간으로 수집하여 아래 형식의 이벤트 배열을 생성한다. 한 턴(사용자가 전송 버튼을 누르기까지)의 이벤트를 하나의 배열로 묶어 박관용(A)의 분류기로 전달한다.

### 출력 스펙

```json
{
  "session_id": "a3f2c1d4-7e8b-4c2a-9f1d-0b5e3a7c6d2e",
  "turn_id": 3,
  "events": [
    {
      "type": "keydown",
      "key": "a",
      "timestamp": 1710234560.001,
      "is_delete": false
    },
    {
      "type": "keyup",
      "key": "a",
      "timestamp": 1710234560.087,
      "is_delete": false
    },
    {
      "type": "keydown",
      "key": "Backspace",
      "timestamp": 1710234562.340,
      "is_delete": true
    }
  ]
}
```

### 필드 정의

| 필드 | 타입 | 설명 |
|---|---|---|
| `session_id` | string | 상담 세션 식별자 |
| `turn_id` | int | 해당 세션 내 턴 번호 (1부터 시작) |
| `events[].type` | string | `"keydown"` 또는 `"keyup"` |
| `events[].key` | string | 입력된 키 값 (KeyboardEvent.key 기준) |
| `events[].timestamp` | float | Unix timestamp (초 단위, 소수점 3자리) |
| `events[].is_delete` | bool | Backspace 또는 Delete 키 여부 |

### 파생 피처 (박관용이 분류기 입력으로 계산)

D는 raw 이벤트만 제공하면 된다. 아래 피처 계산은 A의 분류기 전처리 단계에서 수행한다.

| 피처 | 설명 |
|---|---|
| IKI (Inter-Key Interval) | 연속된 keydown 이벤트 사이의 시간 간격 |
| D1U1 | 하나의 키에 대한 keydown → keyup 시간 (dwell time) |
| D1D2 | 연속된 두 키의 keydown → keydown 시간 |
| U1D2 | 이전 키 keyup → 다음 키 keydown 시간 |
| backspace_rate | 전체 키 입력 중 삭제 키 비율 |
| pause_count | IKI > 2초인 구간의 횟수 |

---

## Module 2 — Keystroke Classifier Output (박관용, A)

### 책임 범위

D의 raw keystroke JSON을 입력받아 EmoSurv 학습 분류기로 감정을 추론하고 아래 형식으로 출력한다.

### 출력 스펙

```json
{
  "session_id": "a3f2c1d4-7e8b-4c2a-9f1d-0b5e3a7c6d2e",
  "turn_id": 3,
  "emotion": "anxious",
  "confidence": 0.61,
  "avg_iki_ms": 2300.0,
  "backspace_rate": 0.14
}
```

### 필드 정의

| 필드 | 타입 | 설명 |
|---|---|---|
| `session_id` | string | D로부터 전달받은 세션 식별자 |
| `turn_id` | int | D로부터 전달받은 턴 번호 |
| `emotion` | string | 분류기가 예측한 감정 레이블 |
| `confidence` | float | 예측 확률값 (0.0 ~ 1.0) |
| `avg_iki_ms` | float | 평균 IKI (밀리초). 프롬프트에서 "입력 지연 X초"로 변환 |
| `backspace_rate` | float | 삭제 키 비율. 프롬프트에서 망설임 정도 표현에 활용 |

### 감정 레이블 목록

EmoSurv 레이블 기준: `angry`, `happy`, `calm`, `sad`, `neutral`

> **참고:** 비전 모듈(C)의 레이블(`fearful`, `surprised` 포함)과 다르다. 프롬프트 조립 단계에서 A가 통합 처리한다.

---

## Module 3 — Text Output (박관용, A)

### 책임 범위

프론트엔드(E)로부터 실시간 텍스트 이벤트를 수신하여 삭제된 텍스트와 최종 전송 텍스트를 분리 캡처한다.

### 출력 스펙

```json
{
  "session_id": "a3f2c1d4-7e8b-4c2a-9f1d-0b5e3a7c6d2e",
  "turn_id": 3,
  "final_text": "그냥 힘들어요",
  "deleted_segments": [
    {
      "text": "죽고 싶어요",
      "deleted_at": 1710234561.200
    },
    {
      "text": "너무 지쳐서",
      "deleted_at": 1710234563.800
    }
  ]
}
```

### 필드 정의

| 필드 | 타입 | 설명 |
|---|---|---|
| `session_id` | string | 세션 식별자 |
| `turn_id` | int | 턴 번호 |
| `final_text` | string | 사용자가 최종 전송한 텍스트 |
| `deleted_segments` | array | 입력 도중 삭제된 텍스트 목록. 없으면 빈 배열 `[]` |
| `deleted_segments[].text` | string | 삭제된 텍스트 내용 |
| `deleted_segments[].deleted_at` | float | 삭제 시점 Unix timestamp |

---

## Module 4 — Silence Monitor (이고은, E)

### 책임 범위

프론트엔드에서 0.2초마다 텍스트 입력창 상태를 폴링하여 마지막 입력 시점을 추적한다. 입력 없이 8초가 경과하면 침묵 이벤트를 생성하여 박관용(A)의 파이프라인으로 전달한다.

**폴링 방식을 선택한 이유:** 침묵은 키 이벤트가 발생하지 않는 상태이므로 이벤트 기반으로는 감지할 수 없다. 일정 주기로 입력창 상태를 확인하는 폴링 방식이 유일한 선택지다.

**침묵 임계값 근거:** 일상 대화에서 3초를 넘는 침묵은 심리적으로 유의미한 것으로 간주되며 (Heldner & Edlund, 2010), 실제 심리치료 세션에서 치료사가 개입하는 시점은 평균 10초 전후로 관찰된다 (Soma et al., 2022). 이를 절충하여 8초를 기본 임계값으로 설정하며, 파일럿 테스트에서 검증 후 확정한다.

### 프론트엔드 구현 가이드 (이고은)

```javascript
// React 커스텀 훅 예시
const POLL_INTERVAL_MS = 200   // 0.2초마다 입력창 상태 체크
const SILENCE_THRESHOLD_SEC = 8  // 8초 이상 입력 없으면 침묵으로 판단

useEffect(() => {
  let lastInputAt = Date.now()
  let silenceFired = false

  const interval = setInterval(() => {
    const currentText = inputRef.current?.value ?? ""
    const now = Date.now()

    if (currentText !== prevText) {
      // 입력 변화 있음 — 타이머 리셋
      lastInputAt = now
      silenceFired = false
      prevText = currentText
    } else {
      // 입력 변화 없음 — 침묵 시간 누적
      const silenceSec = (now - lastInputAt) / 1000
      if (silenceSec >= SILENCE_THRESHOLD_SEC && !silenceFired) {
        silenceFired = true  // 중복 발송 방지
        sendSilenceEvent(silenceSec)
      }
    }
  }, POLL_INTERVAL_MS)

  return () => clearInterval(interval)
}, [])
```

> **주의:** `silenceFired` 플래그로 중복 발송을 방지한다. 사용자가 다시 타이핑을 시작하면 플래그를 리셋한다.

### 출력 스펙

```json
{
  "session_id": "a3f2c1d4-7e8b-4c2a-9f1d-0b5e3a7c6d2e",
  "turn_id": 3,
  "type": "silence_event",
  "silence_duration_sec": 12.4,
  "last_keystroke_at": 1710234560.001,
  "context": "after_llm_response",
  "timestamp": 1710234572.401
}
```

### 필드 정의

| 필드 | 타입 | 설명 |
|---|---|---|
| `session_id` | string | 세션 식별자 |
| `turn_id` | int | 현재 턴 번호 |
| `type` | string | 항상 `"silence_event"` |
| `silence_duration_sec` | float | 마지막 키 입력으로부터 경과한 시간 (초) |
| `last_keystroke_at` | float | 마지막 키 입력 시점 Unix timestamp |
| `context` | string | 침묵 발생 맥락. 아래 참고 |
| `timestamp` | float | 침묵 이벤트 생성 시점 Unix timestamp |

### context 필드 값

| 값 | 설명 |
|---|---|
| `"after_llm_response"` | LLM이 응답한 직후 사용자가 입력하지 않는 상태 |
| `"mid_typing"` | 사용자가 타이핑 도중 멈춘 상태 |

`context` 구분이 중요한 이유: LLM 응답 직후의 침묵은 사용자가 내용을 읽거나 생각하는 중일 수 있고, 타이핑 도중의 침묵은 말하기 어려운 내용을 망설이는 신호일 수 있다. 프롬프트 조립 단계에서 다르게 처리한다.

---

## Trigger Evaluator — LLM 호출 조건 (박관용, A)

### 책임 범위

비전·키스트로크·텍스트·침묵 데이터를 버퍼에 누적하다가 아래 트리거 조건이 충족되면 Prompt Assembler를 호출한다. 트리거 조건이 발생하지 않는 한 LLM을 호출하지 않는다.

### 트리거 조건

| 트리거 | 조건 | 프롬프트 유형 |
|---|---|---|
| 전송 | 사용자가 전송 버튼을 누름 | 일반 프롬프트 |
| 침묵 | 복합 조건 충족 시 (아래 참고) | 침묵 프롬프트 |

### 침묵 트리거 — 복합 조건 (Composite Trigger)

8초 단일 시간 조건만으로는 "읽는 중"과 "망설이는 중"을 구분할 수 없어 임상적으로 불충분하다. 시간 조건에 감정 상태 조건을 AND로 결합하는 복합 트리거를 적용한다.

**개입 조건 (모두 충족 시 침묵 프롬프트 호출):**

1. `silence_duration_sec >= 8.0`
2. 다음 중 하나 이상 해당:
   - 비전 모듈의 `peak_emotion`이 부정 감정(`sad`, `fearful`, `angry`)이고 `peak_confidence >= θ_vision`
   - 키스트로크 분류기의 `emotion`이 부정 감정(`anxious`, `sad`, `angry`)이고 `confidence >= θ_keystroke`
   - `context == "mid_typing"` (타이핑 중 멈춘 경우 — 망설임 신호로 가중)

**개입 보류 조건 (해당 시 호출 안 함):**

- `context == "after_llm_response"` AND 비전·키스트로크 감정 신호 모두 임계값 미만
- 즉, LLM이 방금 응답했고 사용자 감정도 안정적이라면 → 읽고 생각하는 중으로 간주

**임계값(θ) 처리:**

`θ_vision`, `θ_keystroke`는 현재 미확정이다. 파일럿 테스트(N=5 팀원)에서 수집된 비전·키스트로크 감정 신뢰도 분포를 기반으로 확정한다. 코드에서는 `THETA_VISION`, `THETA_KEYSTROKE` 상수로 관리하며, 파일럿 결과에 따라 교체한다.

```python
# modules/pipeline/prompt_assembler.py
THETA_VISION     = 0.60  # 파일럿 테스트 후 확정 — 현재 임시값
THETA_KEYSTROKE  = 0.55  # 파일럿 테스트 후 확정 — 현재 임시값
```

### 침묵 프롬프트 구조

```
[사용자 상태 분석]
표정: {emotion} (신뢰도 {confidence})
시선: {head_pose_label}
침묵 지속: {silence_duration_sec}초
맥락: {context}

사용자가 {silence_duration_sec}초간 입력하지 않고 있습니다.
말하기 어렵거나 정리가 필요한 상황일 수 있습니다.
강요하지 말고, 공간을 주는 방식으로 부드럽게 말을 건네세요.
```

---

## Prompt Assembler — Input & Output (박관용, A)

### 입력

위 모듈들의 출력 JSON을 조합하여 프롬프트를 조립한다.

```python
def assemble_prompt(
    vision: dict,       # Module 1 output
    keystroke: dict,    # Module 2 classifier output
    text: dict,         # Module 3 output
    silence: dict | None = None  # Module 4 output. 침묵 트리거 시에만 전달
) -> str:
    ...
```

### Semantic Mapping 규칙

raw 수치를 LLM에 직접 전달하지 않고 심리적 의미 레이블로 변환한다.

**Head pose 변환:**

```python
def interpret_head_pose(yaw: float, pitch: float) -> str:
    if abs(yaw) > 20:
        return "시선 회피 (고개가 옆으로 돌아있음)"
    elif pitch < -15:
        return "고개 숙임 (위축된 자세)"
    else:
        return "정면 응시"
```

임계값(yaw 20도, pitch -15도)은 파일럿 테스트에서 검증 후 보고서에 근거를 명시한다.

**IKI 변환:**

```python
def interpret_iki(avg_iki_ms: float) -> str:
    if avg_iki_ms > 2000:
        return f"입력 지연 {avg_iki_ms / 1000:.1f}초 (긴 망설임)"
    elif avg_iki_ms > 800:
        return f"입력 지연 {avg_iki_ms / 1000:.1f}초 (보통)"
    else:
        return "빠른 입력"
```

임계값(2000ms, 800ms)은 파일럿 테스트에서 검증 후 보고서에 근거를 명시한다.

### 일반 프롬프트 구조 (전송 트리거)

```
[사용자 상태 분석]
표정(전송 시점): {emotion} (신뢰도 {confidence})
표정(턴 중 최고점): {peak_emotion} (신뢰도 {peak_confidence}, 전송 {elapsed}초 전)
시선: {head_pose_label}
타이핑 패턴: {keystroke_emotion} (신뢰도 {keystroke_confidence}), {iki_label}
삭제된 텍스트: "{deleted_text}"
최종 입력: "{final_text}"

사용자가 말하지 못한 감정이 있을 수 있다.
위 신호들을 종합하여 판단하되, 단정하지 말고
공감적으로 탐색하는 방식으로 응답하라.
```

**삭제된 텍스트가 없는 경우:** 해당 줄을 생략한다.  
**face_detected: false인 경우:** "표정: 데이터 없음"으로 대체한다.

### 특수 토큰 (선택적 확장)

현재는 위 텍스트 기반 프롬프트를 기본 방식으로 사용한다. 추후 LoRA 파인튜닝 단계로 전환할 경우 아래 특수 토큰 방식으로 확장을 검토한다.

| 토큰 | 의미 |
|---|---|
| `[PAUSE_2s]` | 2초 이상의 입력 지연 |
| `[BACKSPACE]` | 삭제 이벤트 발생 |
| `[EMOTION:SAD]` | 감정 레이블 삽입 |
| `[GAZE:AVERTED]` | 시선 회피 감지 |
| `[SILENCE_8s]` | 8초 이상 침묵 감지 |

---

## 버전 히스토리

| 버전 | 날짜 | 변경 내용 | 작성자 |
|---|---|---|---|
| v0.1 | 2026-03-15 | 초안 작성 | 박관용 |
| v0.2 | 2026-03-19 | Module 4 침묵 모니터 추가, Trigger Evaluator 추가, 비전 출력 주기 명시 (0.2초), 침묵 프롬프트 구조 추가, 특수 토큰 `[SILENCE_8s]` 추가 | 박관용 |
| v0.2.1 | 2026-03-19 | Module 2 담당자 오기 수정: 이재철(D) → 조재현(B) | 박관용 |
| v0.2.2 | 2026-03-20 | Vision Output에 peak_emotion, peak_confidence, peak_detected_at 필드 추가. Prompt Assembler 일반 프롬프트 구조에 peak emotion 라인 추가 | 박관용 |
| v0.2.3 | 2026-03-24 | 세션 메타데이터 섹션 추가: session_id를 날짜 기반 형식에서 UUID v4로 교체, 참가자-세션 매핑 테이블 분리 보관 원칙 명시. 전 모듈 JSON 예시의 session_id 값 업데이트 | 박관용 |
| v0.3.0 | 2026-03-24 | Trigger Evaluator 침묵 트리거를 단일 시간 조건(8초)에서 복합 조건(Composite Trigger)으로 고도화. 개입 조건(시간 + 감정 신호 AND), 개입 보류 조건(after_llm_response + 중립 감정), θ 임계값 상수 및 파일럿 확정 방침 명시 | 박관용 |
