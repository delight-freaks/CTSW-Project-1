"""
Pipeline: OpenCV → MediaPipe (Face Detaction + Head Pose) → ResNet → return JSON
출력 주기: 0.2초 (1초에 5번 json이 출력)
출력 스펙: interface_spec.md Module 1 참고

- 데이터 비식별화를 위한 원본 프레임은 추론 직후 메모리에서 폐기
- 현재 사용하는 Resnet 모델은 대충 kaggle에 있는 데이터를 통해서 파인튜닝한 모델 가중치를 사용 중

체크포인트 자동 감지 지원:
  - 기존 6-class 형식 (backbone. prefix 유무 무관)
  - model.pt: encoder + classification_head 구조 (표준 ResNet50, AffectNet 7-class)
  - FER_static_ResNet50_AffectNet: 커스텀 명명 (conv_layer_s2_same, fc1+fc2, AffectNet 7-class)
  AffectNet 7-class → spec 6-class 자동 매핑 (disgust 확률은 angry에 합산)
"""

import re
import time
import json
import threading
import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# spec 6-class 레이블 (interface_spec.md)
EMOTION_LABELS = ["angry", "fearful", "happy", "neutral", "sad", "surprised"]

# AffectNet 7-class → spec 6-class 인덱스 매핑
# AffectNet 순서: neutral(0) happy(1) sad(2) surprise(3) fear(4) disgust(5) anger(6)
# disgust(5)는 angry(0)에 합산
_AFFECTNET_TO_SPEC = [3, 2, 4, 5, 1, 0, 0]

# DDAMFN++ 7-class → spec 6-class 인덱스 매핑 (eval_ddamfn.py에서 검증된 라벨 순서)
# DDAMFN 순서: neutral(0) happy(1) sad(2) surprise(3) fear(4) disgust(5) angry(6)
# disgust(5)는 angry(0)에 합산
_DDAMFN_TO_SPEC = [3, 2, 4, 5, 1, 0, 0]

# Head Pose 추정용 3D 얼굴 모델 포인트
_FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0],     # landmark 1
    [0.0,  -330.0,  -65.0],    # landmark 152
    [-225.0, 170.0, -135.0],   # landmark 226
    [225.0,  170.0, -135.0],   # landmark 446
    [-150.0,-150.0, -125.0],   # landmark 57
    [150.0, -150.0, -125.0],   # landmark 287
], dtype=np.float64)

_LANDMARK_INDICES = [1, 152, 226, 446, 57, 287]


# ResNet 50

class EmotionResNet(nn.Module):
    """
    ResNet50 기반 6-class 안면 감정 분류기
    """

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# Vision Pipeline

class VisionPipeline:
    """
    웹캠 스트림을 5fps 주기로 처리하여 interface_spec.md Module 1 형식의 JSON을 출력

    사용 예:
        pipeline = VisionPipeline(model_path="emotion_resnet.pth")
        pipeline.run()                          # 웹캠 실시간
        output = pipeline.process_frame(frame)  # 단일 프레임 처리
    """

    OUTPUT_FPS = 5
    FRAME_INTERVAL = 1.0 / OUTPUT_FPS

    def __init__(self, model_path: str | None = None, device: str | None = None):
        """
        Args:
            model_path: ResNet 가중치 경로 (.pth / .pt). None이면 랜덤 초기화 (테스트용).
                체크포인트 형식을 자동 감지한다:
                  - encoder+classification_head 구조 (model.pt, AffectNet 7-class)
                  - fc1/fc2 + 커스텀 명명 (FER_static_ResNet50_AffectNet, 7-class)
                  - backbone. prefix 유무 무관한 기존 6-class 형식
            device: 'cuda' | 'cpu'. None이면 자동 선택.
                cuda를 골라도 설치된 torch 빌드에 호스트 GPU 아키텍처용 커널이
                없으면(예: RTX 50시리즈 sm_120 + cu121 torch) 초기화 시 더미
                forward로 감지해 자동으로 CPU로 폴백한다.
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # MediaPipe Face Mesh
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # ResNet 감정 분류 — 체크포인트 형식에 따라 모델·전처리 구성
        # _affectnet_mode: AffectNet 7-class → spec 6-class 매핑 필요 여부
        # _use_vgg_norm:   VGGFace2 정규화 사용 여부 (FER_static 전용)
        self._affectnet_mode = False
        self._use_vgg_norm = False
        self._ddamfn_mode = False
        if model_path and "ddamfn" in model_path.lower():
            # 벤더링한 DDAMFN++ (순수 PyTorch state_dict, modules/vision/ddamfn/)
            # 파일명에 'ddamfn'이 포함되면 이 경로로 로드 (예: ddamfn_rafdb_acc0.9204.pth)
            self._model = self._load_ddamfn(model_path)
            self._ddamfn_mode = True
            print(f"[VisionPipeline] DDAMFN++ 가중치 로드 완료: {model_path}")
        elif model_path:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
            if isinstance(ckpt, dict) and "encoder" in ckpt:
                # model.pt 형식: standard ResNet50, AffectNet 7-class, ImageNet 정규화
                self._model = self._load_model_pt_format(ckpt)
                self._affectnet_mode = True
                print(f"[VisionPipeline] AffectNet 가중치 (encoder/head 형식) 로드 완료: {model_path}")
            elif isinstance(ckpt, dict) and "fc1.weight" in ckpt:
                # FER_static_ResNet50_AffectNet 형식: fc1+fc2, 7-class, VGGFace2 정규화
                self._model = self._load_fer_static_format(ckpt)
                self._affectnet_mode = True
                self._use_vgg_norm = True
                print(f"[VisionPipeline] AffectNet 가중치 (FER_static 형식) 로드 완료: {model_path}")
            else:
                # 기존 6-class 형식 (train.py 체크포인트)
                self._model = EmotionResNet(num_classes=len(EMOTION_LABELS)).to(self.device)
                state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
                # train.py는 flat resnet50을 저장 (backbone. prefix 없음) → 추가
                if not any(k.startswith("backbone.") for k in state):
                    state = {"backbone." + k: v for k, v in state.items()}
                self._model.load_state_dict(state)
                print(f"[VisionPipeline] 가중치 로드 완료: {model_path}")
        else:
            self._model = EmotionResNet(num_classes=len(EMOTION_LABELS)).to(self.device)
            print("[VisionPipeline] 경고: model_path 없음 — 랜덤 가중치 (테스트 전용)")
        self._model.eval()

        # cuda를 골랐더라도 호스트 GPU 아키텍처용 커널이 없으면 forward가
        # 'CUDA error: no kernel image is available...'로 터진다. 매 추론마다
        # 500을 내지 않도록, 초기화 시 1회 더미 forward로 검증하고 실패하면 CPU로 폴백한다.
        self._warmup_or_fallback()

        # FER_static은 VGGFace2 학습 통계로 mean-subtraction만 수행 (std=1)
        # model.pt와 6-class 모델은 표준 ImageNet 정규화 사용
        if self._ddamfn_mode:
            # DDAMFN++는 112x112 입력 + 표준 ImageNet 정규화 (eval_ddamfn.py와 동일)
            self._transform = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        elif self._use_vgg_norm:
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255.0),
                transforms.Normalize(mean=[131.0912, 103.8827, 91.4953],
                                     std=[1.0, 1.0, 1.0]),
            ])
        else:
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        # 턴 내 peak 감정 추적 (thread-safe)
        self._peak_emotion: str | None = None
        self._peak_confidence: float = 0.0
        self._peak_detected_at: float | None = None
        self._peak_lock = threading.Lock()

        # 최신 출력 캐시
        self._latest_output: dict | None = None
        self._output_lock = threading.Lock()

    # External API

    def reset_peak(self) -> None:
        """새 턴 시작 시 peak 초기화. 박관용(A) Trigger Evaluator에서 호출."""
        with self._peak_lock:
            self._peak_emotion = None
            self._peak_confidence = 0.0
            self._peak_detected_at = None

    def get_latest(self) -> dict | None:
        """가장 최근 vision_output 반환. 박관용(A) 파이프라인에서 폴링."""
        with self._output_lock:
            return self._latest_output

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        단일 BGR 프레임을 처리하여 vision_output dict를 반환한다.
        원본 프레임은 함수 종료 시 참조 해제된다
        """
        timestamp = round(time.time(), 3)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 원본 프레임 즉시 참조 해제
        del frame

        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            output = self._make_no_face_output(timestamp)
        else:
            landmarks = results.multi_face_landmarks[0]
            face_crop = self._crop_face(rgb, landmarks, h, w)
            emotion, confidence, emotion_scores = self._classify_emotion(face_crop)
            head_pose = self._estimate_head_pose(landmarks, h, w)
            self._update_peak(emotion, confidence, timestamp)

            output = {
                "timestamp": timestamp,
                "face_detected": True,
                "emotion": emotion,
                "confidence": round(confidence, 4),
                "emotion_scores": {k: round(v, 4) for k, v in emotion_scores.items()},
                "head_pose": head_pose,
                **self._peak_snapshot(),
            }

        del rgb

        with self._output_lock:
            self._latest_output = output

        return output

    def run(self, camera_index: int = 0, output_callback=None) -> None:
        """
        웹캠 메인 루프. 0.2초마다 process_frame을 호출한다.

        Args:
            camera_index: OpenCV 카메라 인덱스 (기본 0)
            output_callback: vision_output dict를 받는 콜백. None이면 stdout JSON 출력.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"카메라를 열 수 없습니다 (index={camera_index})")

        print(f"[VisionPipeline] 시작 — device={self.device}, fps={self.OUTPUT_FPS}")
        try:
            while True:
                t0 = time.time()

                ret, frame = cap.read()
                if not ret:
                    print("[VisionPipeline] 프레임 캡처 실패, 재시도...")
                    time.sleep(0.05)
                    continue

                output = self.process_frame(frame)

                if output_callback:
                    output_callback(output) # vision_output dict를 콜백으로 전달 + 필요에 따라 서버로 전송 코드 추가도 가능
                else:
                    print(json.dumps(output, ensure_ascii=False))

                sleep_sec = self.FRAME_INTERVAL - (time.time() - t0)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)

        except KeyboardInterrupt:
            print("[VisionPipeline] 종료")
        finally:
            cap.release()
            self._face_mesh.close()

    # Internal Methods

    def _warmup_or_fallback(self) -> None:
        """
        cuda 선택 시 더미 forward 1회로 커널 가용성을 검증하고, 실패하면 CPU로 폴백한다.

        설치된 torch 빌드에 호스트 GPU 아키텍처(sm_XX)용 커널이 없으면 forward가
        'CUDA error: no kernel image is available for execution on the device'로
        터진다(예: RTX 50시리즈 sm_120 + cu121 torch). CUDA 에러는 비동기로 보고되므로
        synchronize()로 강제로 표면화한 뒤, 발생 시 모델을 CPU로 옮겨 재검증한다.
        """
        if self.device.type != "cuda":
            return
        size = 112 if self._ddamfn_mode else 224
        try:
            with torch.no_grad():
                self._model(torch.zeros(1, 3, size, size, device=self.device))
            torch.cuda.synchronize()   # 비동기 CUDA 커널 에러를 여기서 동기적으로 표면화
        except RuntimeError as e:
            print(f"[VisionPipeline] CUDA forward 실패 — CPU로 폴백합니다: {e}")
            self.device = torch.device("cpu")
            self._model.to(self.device)
            with torch.no_grad():
                self._model(torch.zeros(1, 3, size, size, device=self.device))

    def _crop_face(self, rgb: np.ndarray, landmarks, h: int, w: int) -> Image.Image:
        """MediaPipe 랜드마크 bbox + padding으로 얼굴 크롭 → PIL RGB Image"""
        xs = [lm.x * w for lm in landmarks.landmark]
        ys = [lm.y * h for lm in landmarks.landmark]
        pad = 20
        x1 = int(max(min(xs) - pad, 0))
        x2 = int(min(max(xs) + pad, w))
        y1 = int(max(min(ys) - pad, 0))
        y2 = int(min(max(ys) + pad, h))
        return Image.fromarray(rgb[y1:y2, x1:x2])

    def _classify_emotion(self, face_img: Image.Image) -> tuple[str, float, dict]:
        """ResNet으로 감정 분류 → (top_label, top_confidence, scores_dict)"""
        tensor = self._transform(face_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(tensor)
            if self._ddamfn_mode:
                logits = logits[0]   # DDAMNet.forward는 (out, feat, heads) 튜플 반환
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        if self._affectnet_mode:
            probs = self._map_affectnet_probs(probs)
        elif self._ddamfn_mode:
            probs = self._map_ddamfn_probs(probs)

        scores = {label: float(probs[i]) for i, label in enumerate(EMOTION_LABELS)}
        top_idx = int(np.argmax(probs))
        return EMOTION_LABELS[top_idx], float(probs[top_idx]), scores

    def _load_model_pt_format(self, ckpt: dict) -> nn.Module:
        """encoder + classification_head 구조 로드 (표준 ResNet50, AffectNet 7-class)"""
        backbone = models.resnet50(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, 7)
        state = dict(ckpt["encoder"])
        state.update(ckpt["classification_head"])  # fc.weight, fc.bias
        backbone.load_state_dict(state)
        return backbone.to(self.device)

    def _load_fer_static_format(self, ckpt: dict) -> nn.Module:
        """FER_static_ResNet50_AffectNet 형식 로드 (커스텀 명명, fc1+fc2, 7-class)"""
        backbone = models.resnet50(weights=None)
        backbone.fc = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 7),
        )
        backbone.load_state_dict(self._remap_fer_static_keys(ckpt))
        return backbone.to(self.device)

    @staticmethod
    def _remap_fer_static_keys(state_dict: dict) -> dict:
        """
        FER_static_ResNet50_AffectNet 키 → torchvision ResNet50 키 변환
          conv_layer_s2_same → conv1
          batch_norm1 (최상위) → bn1
          layer*.*.batch_normX → layer*.*.bnX
          i_downsample → downsample
          fc1 → fc.0 (Sequential index 0)
          fc2 → fc.2 (Sequential index 2, ReLU가 index 1)
        """
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("fc1."):
                k = "fc.0." + k[4:]
            elif k.startswith("fc2."):
                k = "fc.2." + k[4:]
            else:
                k = k.replace("conv_layer_s2_same.", "conv1.")
                if k.startswith("batch_norm1."):
                    k = "bn1." + k[12:]
                k = re.sub(r"(layer\d+\.\d+\.)batch_norm(\d+)\.", r"\1bn\2.", k)
                k = k.replace(".i_downsample.", ".downsample.")
            new_state[k] = v
        return new_state

    @staticmethod
    def _map_affectnet_probs(probs: np.ndarray) -> np.ndarray:
        """AffectNet 7-class 확률 → spec 6-class 확률 (disgust는 angry에 합산)"""
        spec = np.zeros(len(EMOTION_LABELS))
        for affectnet_idx, spec_idx in enumerate(_AFFECTNET_TO_SPEC):
            spec[spec_idx] += probs[affectnet_idx]
        return spec

    def _load_ddamfn(self, model_path: str) -> nn.Module:
        """
        벤더링한 DDAMFN++ 로드 (modules/vision/ddamfn/, 순수 PyTorch).
        체크포인트는 {'model_state_dict': ...} 형식. pretrained=False로 빈 백본을 만든 뒤
        state_dict만 덮어쓰므로 MFN_msceleb.pth 사전학습 파일은 필요 없다.
        """
        import sys
        from pathlib import Path
        ddamfn_dir = Path(__file__).parent / "ddamfn"
        if str(ddamfn_dir) not in sys.path:
            sys.path.insert(0, str(ddamfn_dir))
        from DDAM import DDAMNet

        model = DDAMNet(num_class=7, num_head=2, pretrained=False)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        return model.to(self.device)

    @staticmethod
    def _map_ddamfn_probs(probs: np.ndarray) -> np.ndarray:
        """DDAMFN++ 7-class 확률 → spec 6-class 확률 (disgust는 angry에 합산)"""
        spec = np.zeros(len(EMOTION_LABELS))
        for ddamfn_idx, spec_idx in enumerate(_DDAMFN_TO_SPEC):
            spec[spec_idx] += probs[ddamfn_idx]
        return spec

    def _estimate_head_pose(self, landmarks, h: int, w: int) -> dict:
        """
        MediaPipe 6개 랜드마크 + solvePnP → yaw / pitch / roll (도 단위)
        yaw 양수: 오른쪽, 음수: 왼쪽 / pitch 양수: 위, 음수: 아래
        """
        image_pts = np.array([
            [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
            for i in _LANDMARK_INDICES
        ], dtype=np.float64)

        fl = float(w)
        cam = np.array([[fl, 0, w / 2], [0, fl, h / 2], [0, 0, 1]], dtype=np.float64)

        ok, rvec, _ = cv2.solvePnP(
            _FACE_3D_MODEL, image_pts, cam, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        rmat, _ = cv2.Rodrigues(rvec)
        pitch = round(float(np.degrees(np.arcsin(-rmat[2, 0]))), 2)
        yaw   = round(float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))), 2)
        roll  = round(float(np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))), 2)
        return {"yaw": yaw, "pitch": pitch, "roll": roll}

    def _update_peak(self, emotion: str, confidence: float, timestamp: float) -> None:
        with self._peak_lock:
            if confidence > self._peak_confidence:
                self._peak_emotion = emotion
                self._peak_confidence = round(confidence, 4)
                self._peak_detected_at = timestamp

    def _peak_snapshot(self) -> dict:
        with self._peak_lock:
            return {
                "peak_emotion": self._peak_emotion,
                "peak_confidence": self._peak_confidence,
                "peak_detected_at": self._peak_detected_at,
            }

    def _make_no_face_output(self, timestamp: float) -> dict:
        return {
            "timestamp": timestamp,
            "face_detected": False,
            "emotion": None,
            "confidence": None,
            "emotion_scores": None,
            "head_pose": None,
            **self._peak_snapshot(),
        }


if __name__ == "__main__":
    import argparse
    from pathlib import Path as _Path

    parser = argparse.ArgumentParser(description="Vision Pipeline 실행")
    parser.add_argument("--model", type=str, default=None, help="ResNet 가중치 .pth 경로")
    parser.add_argument("--camera", type=int, default=0, help="카메라 인덱스 (기본 0)")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="vision_output JSON을 저장할 디렉토리 (기본: stdout). "
             "백엔드 연동 시 backend/vision_raw 경로를 지정한다."
    )
    args = parser.parse_args()

    callback = None
    if args.output_dir:
        _out_dir = _Path(args.output_dir)
        _out_dir.mkdir(parents=True, exist_ok=True)

        def callback(output: dict) -> None:
            """vision_output을 타임스탬프 기반 JSON 파일로 저장한다."""
            fname = _out_dir / f"vision_{output['timestamp']:.3f}.json"
            fname.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")

    VisionPipeline(model_path=args.model).run(camera_index=args.camera, output_callback=callback)
