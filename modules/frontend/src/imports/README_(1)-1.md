# 키스트로크 과정 데이터 기반 멀티모달 심리 상담 LLM

---

## 1. 프로젝트 개요

사용자가 텍스트를 입력하는 과정에서 발생하는 데이터(입력 지연, 삭제, 커서 움직임 등)와 안면 표정(영상)을 동시에 분석하여, 최종 전송된 텍스트만으로는 포착하기 어려운 내면의 감정 상태를 추론하고 공감적으로 반응하는 상담용 LLM 플랫폼 개발.

**핵심 차별점:** 기존 상담 챗봇은 최종 전송된 텍스트만 분석하지만, 이 프로젝트는 다음 세 가지를 추가로 활용한다.

1. **삭제된 텍스트:** 사용자가 "불안해요"라고 썼다가 지우고 "괜찮아요"라고 전송했을 때, 지워진 텍스트까지 모델이 맥락으로 인식한다. 삭제된 텍스트를 심리 상태 추론에 직접 활용한 NLP 연구는 현재 초기 단계에 머물러 있어 학술적 차별점이 명확하다.
2. **키스트로크 패턴:** D1U1, D1D2, U1D2 등 타이밍 피처를 EmoSurv 학습 분류기로 감정 추론에 활용한다.
3. **침묵 감지:** 사용자가 8초 이상 입력하지 않으면 LLM이 먼저 말을 건다. 말하기 어려운 상황을 텍스트 전송 없이도 감지하는 Proactive 상담 기능이다.

---

## 2. 팀 구성 및 역할 분담

| 팀원 | 배경 | 역할 |
|---|---|---|
| 박관용 (A) | LLM 연구, AI 에이전트 해커톤 경험 | 키스트로크 감정 분류기 학습, 멀티모달 신호의 프롬프트 변환 설계, 프롬프트 조립 파이프라인, Trigger Evaluator, Claude API 통신 |
| 조재현 (B) | 의료 AI 기획, 바이브 코딩 | 브라우저 키스트로크 로거 개발 (React 커스텀 훅), 사용자 평가 설계(PETS), 사용자 테스트 운영 및 분석 |
| 심인영 (C) | 비전 AI (인턴십 포함, 다수 프로젝트) | 안면 표정 비전 파이프라인 개발 (ResNet 계열), 감정 레이블+확률값 JSON 출력 (0.2초 주기) |
| 이재철 (D) | ResNet, 백엔드, 보안 | TimescaleDB 기반 세션 데이터 저장·관리 백엔드, 키스트로크 데이터 및 침묵 이벤트 수신·저장 인터페이스 설계 |
| 이고은 (E) | 피그마, 프론트엔드 | 상담 채팅 UI 개발, 피그마 와이어프레임 설계, 키스트로크 로거 통합, 웹캠 스트림 전달, 침묵 모니터(Silence Monitor) 구현 |

**병렬 개발 전략:** A가 C, D, E의 출력 형식(JSON 인터페이스)을 먼저 정의하면, 각 모듈이 충돌 없이 독립적으로 개발 가능하다. 인터페이스 명세는 `interface_spec.md`를 참고한다.

---

## 3. 저장소 구조

```
keystroke-multimodal-counselor/
├── modules/
│   ├── classifier/          # 키스트로크 감정 분류기 (박관용)
│   │   ├── preprocessing.py
│   │   ├── classifier.py
│   │   └── data/
│   │       └── emosurv/     # EmoSurv CSV 4개 (.gitignore 적용)
│   ├── pipeline/            # 프롬프트 조립 및 LLM 통신 (박관용)
│   │   ├── prompt_assembler.py
│   │   └── llm_client.py
│   ├── keystroke/           # 키스트로크 로거 및 백엔드 (조재현, 이재철)
│   ├── vision/              # 비전 파이프라인 (심인영)
│   ├── frontend/            # 채팅 UI 및 침묵 모니터 (이고은)
│   └── evaluation/          # 사용자 평가 (조재현)
├── interface_spec.md        # 모듈 간 인터페이스 명세 (박관용)
├── .gitignore
└── README.md
```

> **주의:** EmoSurv CSV 파일은 IEEE DataPort 라이선스상 재배포가 금지되어 있으므로 `.gitignore`에 등록하여 레포에 포함하지 않는다. 팀원 각자가 아래 데이터 세팅 절차에 따라 로컬에서 직접 다운로드한다.

---

## 4. 개발 환경 세팅

### 저장소 클론

```bash
cd ~/projects
git clone https://github.com/gwanyong02/keystroke-multimodal-counselor.git
cd keystroke-multimodal-counselor
```

### EmoSurv 데이터셋 다운로드

1. [IEEE DataPort EmoSurv 페이지](https://ieee-dataport.org/open-access/emosurv-typing-biometric-keystroke-dynamics-dataset-emotion-labels-created-using)에서 IEEE 계정(무료)으로 로그인
2. 아래 4개 파일을 다운로드

   | 파일명 | 크기 |
   |---|---|
   | Fixed Text Typing Dataset.csv | 2.77 MB |
   | Free Text Typing Dataset.csv | 2.55 MB |
   | Frequency Dataset.csv | 8.23 KB |
   | Participants Information.csv | 12.61 KB |

3. `modules/classifier/data/emosurv/` 폴더에 위치시킨다

```bash
mkdir -p modules/classifier/data/emosurv
# 다운로드한 CSV 4개를 위 경로에 복사
```

### 분류기 실행

```bash
cd modules/classifier
pip install xgboost scikit-learn pandas numpy matplotlib
python classifier.py ./data/emosurv
# 옵션: --model svm (SVM으로 비교 실험)
```

### 프롬프트 조립 모듈 테스트 (mock 데이터)

```bash
cd modules/pipeline
python prompt_assembler.py          # mock JSON으로 프롬프트 생성 확인
python prompt_assembler.py --claude # Claude API 실제 호출 (ANTHROPIC_API_KEY 필요)
```

### .gitignore

```
modules/classifier/data/
*.pkl
*.pdf
eval_summary.json
prompt_payload_sample.json
.env
```

---

## 5. 시스템 아키텍처: Late Fusion

세 모달리티를 하나의 통합 모델로 end-to-end 학습시키는 Early/Intermediate Fusion 대신, **Late Fusion** 방식을 채택한다.

**Late Fusion을 선택한 이유:**
- 이질적인 데이터(텍스트·키스트로크·영상) 간 정렬(alignment) 문제와 역전파(backpropagation) 설계가 필요한 통합 아키텍처는 구현이 어려움
- Late Fusion은 각 모달리티를 독립 모듈로 분리하고 출력 결과만 마지막 단계에서 합치므로 대학생 팀이 병렬 개발 가능
- 각 모달리티의 해석 가능성을 유지할 수 있어 설명가능한 AI(XAI) 방향성과 일치하며 보고서의 강점으로 활용 가능

### 데이터 흐름

```
[Frontend (E)]
      |
      |-- 웹캠 스트림 ---------> [Vision Module (C)]      --> vision_output JSON (0.2초 주기)
      |-- 키스트로크 이벤트 ---> [Keystroke Logger (B/D)] --> raw keystroke JSON
      |                                                           |
      |                                              [Keystroke Classifier (A)]
      |                                                           |
      |                                              keystroke_output JSON
      |-- 텍스트 폴링 (0.2초) --> [Silence Monitor (E)]  --> silence_event JSON
      |-- 텍스트 이벤트 -------> [Text Capture (A)]      --> text_output JSON

vision_output + keystroke_output + text_output
      |
[Trigger Evaluator (A)] <--- silence_event
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

### 모달리티별 모듈 구성

**모듈 1 — 비전 (심인영 담당)**

파이프라인: OpenCV → MediaPipe → ResNet → JSON 출력 (0.2초 주기)

- **OpenCV:** 웹캠 프레임 캡처
- **MediaPipe:** 얼굴 영역 감지 및 랜드마크 추출, 크롭된 얼굴 이미지와 head pose 전달. YOLO는 객체 탐지 특화 모델로 표정 분류에 부적합하여 제외
- **ResNet:** 크롭된 얼굴 이미지로 감정 분류, 클래스별 확률 분포 출력

비전 모듈은 턴 전체의 vision_output 버퍼에서 두 가지 값을 추출하여 프롬프트에 포함한다. 전송 시점의 최신 감정(current emotion)과 턴 중 confidence 최고값을 기록한 감정(peak emotion)을 함께 전달한다. 표정은 잠깐 드러났다가 억제될 수 있으므로 최신값만 사용하면 일시적으로 노출된 감정 신호를 놓칠 수 있다. LLM은 두 값을 비교하여 "현재는 감정을 숨기고 있으나 턴 중 두려움이 포착됨" 같은 맥락을 스스로 해석한다.

**모듈 2 — 키스트로크 (조재현 수집 / 박관용 분류)**
- 조재현: 브라우저에서 key down, key up, 타임스탬프, 삭제 여부를 실시간 수집하는 React 커스텀 훅(useKeystrokeLogger) 개발, 메시지 전송 시 버퍼 일괄 전송
- 이재철: 전달된 데이터를 TimescaleDB에 수신·저장
- 박관용: EmoSurv 데이터셋으로 학습한 XGBoost 분류기가 D1U1, D1D2, U1D2 등 키스트로크 피처를 받아 감정 레이블+확률값 출력

**모듈 3 — 텍스트 (박관용 담당)**
- 삭제된 텍스트(Counterfactual Text)와 최종 전송 텍스트를 모두 캡처하여 전달

**모듈 4 — 침묵 모니터 (이고은 담당)**

0.2초마다 텍스트 입력창 상태를 폴링하여 마지막 입력 시점을 추적한다. 입력 없이 8초가 경과하면 침묵 이벤트를 생성하여 박관용의 파이프라인으로 전달한다.

**침묵 임계값 근거:** 일상 대화에서 3초를 넘는 침묵은 심리적으로 유의미한 것으로 간주되며(Heldner & Edlund, 2010), 실제 심리치료 세션에서 치료사가 개입하는 시점은 평균 10초 전후로 관찰된다(Soma et al., 2022). 이를 절충하여 8초를 기본 시간 조건으로 설정한다. 단, 8초 단일 시간 조건만으로는 "읽는 중"과 "망설이는 중"을 구분할 수 없어 임상적으로 불충분하다. 이를 보완하기 위해 비전·키스트로크 감정 신호 조건을 AND로 결합한 복합 트리거(Composite Trigger)를 적용하며, 감정 신뢰도 임계값(θ_vision, θ_keystroke)은 파일럿 테스트(N=5)에서 수집된 신뢰도 분포를 기반으로 확정한다.

**프롬프트 조립 및 LLM 통신 (박관용 담당)**

세 모듈의 JSON 출력을 그대로 LLM에 전달하지 않고, **semantic mapping** 과정을 거쳐 심리적 의미 레이블로 변환한 뒤 구조화된 프롬프트로 조립한다. raw 수치를 그대로 주면 LLM의 해석이 일관되지 않을 수 있기 때문이다.

역할 분리:
- `prompt_assembler.py`: 모달리티 출력 → 프롬프트 문자열 생성 전담
- `llm_client.py`: Claude API 호출, 응답 수신, 오류 재시도 전담. 추후 모델 교체(Claude → 로컬 모델 등) 시 이 파일만 수정하면 된다.

**Trigger Evaluator (박관용 담당)**

버퍼에 누적된 멀티모달 데이터를 보다가 아래 두 가지 조건 중 하나가 충족되면 LLM을 호출한다.

| 트리거 | 조건 | 프롬프트 유형 |
|---|---|---|
| 전송 | 사용자가 전송 버튼을 누름 | 일반 프롬프트 |
| 침묵 | 복합 조건 충족 시 (시간 + 감정 신호) | 침묵 프롬프트 |

침묵 트리거는 8초 단일 시간 조건에서 복합 트리거(Composite Trigger)로 고도화한다. 개입 조건은 `silence_duration_sec >= 8.0` AND (비전 peak_emotion 부정 감정 ≥ θ_vision / 키스트로크 emotion 부정 감정 ≥ θ_keystroke / `context == "mid_typing"` 중 하나 이상)이다. 반대로 `context == "after_llm_response"`이고 감정 신호가 모두 임계값 미만이면 개입을 보류한다(읽는 중으로 간주). θ값은 파일럿 테스트(N=5)에서 확정하며, 코드에서 `THETA_VISION`, `THETA_KEYSTROKE` 상수로 관리한다.

일반 프롬프트 예시:

```
[사용자 상태 분석]
표정: 슬픔 (신뢰도 0.72)
시선: 시선 회피 (고개가 옆으로 돌아있음)
타이핑 패턴: 불안 (신뢰도 0.61), 입력 지연 2.3초
삭제된 텍스트: "죽고 싶어요"
최종 입력: "그냥 힘들어요"

사용자가 말하지 못한 감정이 있을 수 있다.
위 신호들을 종합하여 판단하되, 단정하지 말고
공감적으로 탐색하는 방식으로 응답하라.
```

침묵 프롬프트 예시:

```
[사용자 상태 분석]
표정: 슬픔 (신뢰도 0.68)
시선: 고개 숙임 (위축된 자세)
침묵 지속: 12.4초
맥락: after_llm_response

사용자가 12.4초간 입력하지 않고 있습니다.
말하기 어렵거나 정리가 필요한 상황일 수 있습니다.
강요하지 말고, 공간을 주는 방식으로 부드럽게 말을 건네세요.
```

특수 토큰 예시: `[PAUSE_2s]`, `[BACKSPACE]`, `[EMOTION:SAD]`, `[GAZE:AVERTED]`, `[SILENCE_8s]`

---

## 6. 데이터셋

### 키스트로크 감정 분류기 학습용: EmoSurv
- 출처: [IEEE DataPort](https://ieee-dataport.org/open-access/emosurv-typing-biometric-keystroke-dynamics-dataset-emotion-labels-created-using)
- 구성: 124명 참가자, 감정 레이블(분노·행복·평온·슬픔·중립) 포함
- 수집 방식: 참가자가 특정 감정 유도 영상 시청 후 자유 텍스트 및 고정 텍스트 타이핑
- 제공 피처: key down, key up, D1U1, D1D2, U1D2 등 키 입력 타이밍 시계열 데이터
- 접근 방법: IEEE 계정(무료)으로 로그인 후 오픈 액세스 다운로드 가능
- **수집 환경:** 웹 애플리케이션 기반으로 수집됨. 논문 원문에서 의도적으로 웹 환경을 선택했음을 명시. 본 프로젝트의 React 브라우저 기반 수집 방식과 훈련-추론 환경이 일치하며, 이를 EmoSurv 선택 근거로 보고서에 명시할 수 있음
- **한계 (보고서에 명시 필요):** 감정 유도 방식이 영상 시청이므로 실제 상담 맥락과 다소 거리가 있음

### 상담용 LLM 파인튜닝용
- **EmpatheticDialogues** (Facebook Research): 25,000개 공감 대화 쌍
- **Counsel Chat**: 실제 상담사와 내담자의 Q&A 데이터셋

---

## 7. LLM 전략: 프롬프트 엔지니어링 중심

**핵심 전략 — API 기반 프롬프트 엔지니어링**
- Claude API를 호출하여 전체 파이프라인을 구성한다. 공감 응답 품질 벤치마크에서 Claude가 GPT-4o 대비 우위를 보인 점을 근거로 선택했다.
- 별도 모델 학습 없이 프롬프트 엔지니어링만으로 상담 품질을 확보한다.
- 상담 성능 자체는 Claude에 위임하고, 본 프로젝트의 기여는 멀티모달 입력을 프롬프트에 주입하는 파이프라인에 집중한다.

별도 모델 학습(LoRA 파인튜닝 등)은 필요 시에만 선택적으로 진행한다.

---

## 8. 하드웨어

- **GPU:** RTX 5070 Ti (VRAM 16GB, Blackwell 아키텍처)
- **작업별 가능 여부:**
  - 키스트로크 분류기 학습: 가능 (CPU만으로도 충분, VRAM 불필요)
  - 비전 모듈 추론: 가능 (16GB에서 여유롭게 수행)
  - Mistral 7B / LLaMA 3 8B LoRA 파인튜닝: 가능 (4bit 양자화 적용 시 약 8~10GB)
  - LLaMA 3 70B 이상: 불가 (최소 80GB 이상 필요)

---

## 9. 성능 평가 방법론

### 평가 목적

멀티모달 입력(키스트로크 감정, 표정, 삭제된 텍스트, 침묵 감지)이 LLM의 공감 응답 품질을 실제로 향상시키는지 측정한다. Claude의 상담 성능 자체가 아니라, 멀티모달 입력 주입 유무에 따른 응답 품질 변화가 평가 대상이다.

### 비교 조건 (Ablation)

- **조건 A (Baseline):** 최종 전송 텍스트만 LLM에 입력 — 기존 상담 챗봇과 동일한 조건
- **조건 B (제안 시스템):** 최종 텍스트 + 키스트로크 감정 + 표정 + 삭제된 텍스트 + 침묵 감지 전부 포함

### 실험 설계

Within-Subject Design을 채택한다. 동일한 참가자가 조건 A와 조건 B를 모두 경험하고 비교 평가한다. 순서 효과(Order Effect)를 통제하기 위해 참가자의 절반은 A→B, 나머지 절반은 B→A 순서로 진행한다.

### 참가자 구성

평가는 파일럿과 본 평가 두 단계로 분리하여 진행한다.

- **파일럿 테스트 (N=5, 개발 연구자):** 본 평가 전 팀원 5명이 평가 절차를 검증한다. 상담 시나리오 흐름의 자연스러움, 소요 시간 적절성, PETS 문항의 이해 가능성을 점검하고 필요 시 평가 설계를 개선한다. 팀원은 가설 및 조건 설계를 사전에 인지한 비맹검(non-blind) 평가자이므로 본 평가에서는 제외한다.
- **본 평가 (N=10, 교내 일반 참가자):** 가설을 사전에 인지하지 않은 교내 일반 참가자 10명을 대상으로 본 평가를 진행한다.

### 평가 척도

**PETS (Perceived Empathy of Technology Scale)** 단일 척도를 적용한다. 본 프로젝트의 핵심 주장이 "멀티모달 입력을 통해 더 공감적인 응답을 생성한다"이므로, 공감 반응 품질을 사용자 관점에서 직접 측정하는 PETS가 가장 적합하다. 조건 A와 조건 B의 PETS 점수 차이를 비교해 멀티모달 입력의 효과를 측정한다.

**조재현의 역할:** 상담 시나리오 설계, PETS 적용, 사용자 테스트 운영 및 결과 분석

### 한계 (보고서에 명시 필요)

- 본 평가 표본 크기(N=10)가 작아 결과의 통계적 일반화에 한계가 있음
- 참가자가 교내로 한정되어 다양한 연령·배경을 대표하지 못함
- 파일럿 참가자(팀원)는 비맹검 평가자로서 파일럿 결과 해석 시 편향 가능성에 유의 필요
- 침묵 임계값(8초)은 파일럿 테스트 기반 고정값으로, 개인차를 반영하지 못함

---

## 10. 윤리적 고려사항 (보고서에 반드시 포함)

- **투명한 동의(Informed Consent):** 키 입력 시간, 삭제 내용, 침묵 시간, 표정 등 수집 데이터와 활용 방식을 명확히 고지 후 명시적 동의 획득
- **강력한 익명화(Anonymization):** 삭제된 텍스트에 포함될 수 있는 개인정보 탐지 및 마스킹
- **데이터 보안:** 저장·전송 과정 암호화, 데이터 유출 시 책임 소재 명확화
- **면책 범위:** 이 시스템은 "진단"이 아닌 "상담 보조" 목적임을 명시

### 데이터 비식별화 구현 원칙

#### 1. 안면 영상 즉시 폐기 (비전 모듈 담당: 심인영)

- OpenCV로 캡처한 원본 프레임은 MediaPipe → ResNet 추론에만 사용하고 메모리에서 즉시 폐기한다.
- 디스크 및 DB에 기록되는 것은 vision_output JSON(감정 레이블, 확률값)뿐이며, 원본 프레임은 어떤 형태로도 저장하지 않는다.
- 비전 모듈(심인영) 코드 리뷰 시 OpenCV 버퍼가 파일로 flush되지 않는지 반드시 확인한다.

#### 2. 텍스트 PII 마스킹 (프롬프트 조립 담당: 박관용 / 백엔드 담당: 이재철)

- `final_text` 및 `deleted_segments`의 텍스트는 TimescaleDB 저장 및 LLM 전달 전에 정규식 기반 PII 마스킹을 적용한다.
- 마스킹 대상: 한국 전화번호, 주민등록번호, 이메일 주소.
- 마스킹 함수는 `modules/pipeline/prompt_assembler.py`에 `mask_pii(text: str) -> str`로 구현하며, `assemble_prompt` 내부에서 호출한다.
- 백엔드(이재철)는 텍스트 수신 시 마스킹된 값만 저장하도록 박관용과 인터페이스를 맞춘다.

#### 3. 세션 식별자 가명처리 (파이프라인 담당: 박관용)

- `session_id`는 세션 시작 시 백엔드(이재철)가 생성·발급하며, 프론트엔드(이고은)는 발급받은 값을 모든 모듈에 전파한다.
- `session_id`는 날짜 기반 형식(`sess_20260315_001`) 대신 UUID v4(`uuid.uuid4()`)를 사용한다.
- 실제 참가자와 `session_id` 간의 매핑 테이블은 별도 파일로 분리 보관하며, 분석 시에는 `session_id`만 사용한다.
- 매핑 테이블 파일은 `.gitignore`에 등록하여 레포에 포함하지 않는다.

---

## 11. 선행연구 근거 요약

- **키스트로크 동역학(Keystroke Dynamics):** 타이핑 속도, 키 입력 간 지연, 백스페이스 사용 빈도 등이 스트레스·불안·우울·인지 부하와 유의미한 상관관계를 보임이 학술적으로 증명됨 (디지털 표현형, Digital Phenotyping 분야)
- **TypeFormer:** 키보드 입력 이벤트와 시간 간격을 시퀀스 데이터로 간주하여 트랜스포머 아키텍처로 처리하는 선행 연구
- **MARS 모델:** 음성·제스처 같은 비언어적 신호를 이산적인 잠재 토큰(discrete latent tokens)으로 변환하여 텍스트 토큰과 함께 LLM을 학습시키는 방법 제안 → 본 프로젝트의 특수 토큰 설계 방식의 근거
- **반사실적 텍스트(Counterfactual Text) 분석:** 삭제된 텍스트를 심리 상태 추론에 활용하는 NLP 연구는 아직 초기 단계 → 본 프로젝트의 학술적 차별점
- **상담 침묵 연구:** 일상 대화에서 3초 이상의 침묵은 심리적으로 유의미하며(Heldner & Edlund, 2010), 심리치료 세션에서 치료사의 개입 시점은 평균 10초 전후로 관찰됨(Soma et al., 2022) → 침묵 임계값 8초 설정의 근거

---

## 12. 한계 (보고서에 명시 필요)

1. EmoSurv의 감정 유도 방식(영상 시청)이 실제 상담 맥락과 다소 거리가 있음
2. 124명 규모의 EmoSurv 데이터셋이 모델 학습에 충분한지 검증 필요
3. 텍스트·키스트로크·영상 이질적 데이터의 통합 아키텍처를 Late Fusion으로 단순화했으므로 end-to-end 학습 대비 최적 성능에 제한이 있을 수 있음
4. 침묵 임계값(8초)이 고정값으로 개인차 및 상담 단계를 반영하지 못함

---

## References

- [EmoSurv Dataset (IEEE DataPort)](https://ieee-dataport.org/open-access/emosurv-typing-biometric-keystroke-dynamics-dataset-emotion-labels-created-using)
- [Keystroke Feature Calculation (PDF)](https://ieee-dataport.s3.amazonaws.com/docs/12722/Keystroke%20feature%20calculation.pdf)
- Maalej, A. and Kallel, I., "Does Keystroke Dynamics tell us about Emotions? A Systematic Literature Review and Dataset Construction," 2020 16th International Conference on Intelligent Environments (IE), Madrid, Spain, 2020, pp. 60-67, doi: 10.1109/IE49459.2020.9155004
- Heldner, M. & Edlund, J., "Pauses, gaps and overlaps in conversations," Journal of Phonetics, 38(4), 555-568, 2010
- Soma, C. S., Wampold, B. E., et al., "The silent treatment?: Changes in patient emotional expression after silence," Counselling and Psychotherapy Research, 2022, doi: 10.1002/capr.12560
