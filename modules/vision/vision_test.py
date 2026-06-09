"""
사용법:
    python vision_test.py                        # 웹캠 실시간 (기본)
    python vision_test.py --debug                # 랜드마크·head pose·감정 시각화 (창 표시)
    python vision_test.py --image face.jpg       # 정적 이미지 1회 처리
    python vision_test.py --image face.jpg --debug  # 이미지 시각화 후 창 대기
    python vision_test.py --mock                 # 카메라 없이 출력 형식만 검증
    python vision_test.py --model emotion.pth    # 가중치 지정
"""

import argparse
import json
import time
import numpy as np
import cv2
import mediapipe as mp

from vision_pipeline import VisionPipeline, EMOTION_LABELS, _LANDMARK_INDICES, _FACE_3D_MODEL


def test_webcam(model_path: str | None = None, camera: int = 0) -> None:
    pipeline = VisionPipeline(model_path=model_path)
    print("웹캠 실시간 테스트 시작 (Ctrl+C로 종료)\n")
    pipeline.run(camera_index=camera)


# Landmark·head pose·감정 시각화용 헬퍼 함수

def _draw_debug_overlay(frame: np.ndarray, output: dict, landmarks) -> np.ndarray:
    """
    프레임 위에 디버그 오버레이를 그려 반환한다.
    - 얼굴 메쉬 전체 (연두색 점)
    - Head pose 추정에 사용된 6개 핵심 랜드마크 (빨간 원)
    - Head pose 3D 축 (X=파랑, Y=초록, Z=빨강)
    - 감정 확률 막대그래프 (우측 상단) -> 임시로 추가한 기능
    - 현재 감정·confidence·head pose 수치 텍스트
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    if not output["face_detected"] or landmarks is None:
        cv2.putText(vis, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return vis

    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    # 1) 얼굴 메쉬 전체 (연두색 점)
    mp_drawing.draw_landmarks(
        image=vis,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 255, 120), thickness=1, circle_radius=1),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(0, 200, 80), thickness=1),
    )

    # 2) Head pose 핵심 6개 랜드마크 (빨간 원)
    for idx in _LANDMARK_INDICES:
        lm = landmarks.landmark[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(vis, str(idx), (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 255), 1)

    # 3) Head pose 3D 축 시각화 (코 끝을 원점으로)
    nose = landmarks.landmark[1]
    nose_pt = (int(nose.x * w), int(nose.y * h))
    fl = float(w)
    cam = np.array([[fl, 0, w / 2], [0, fl, h / 2], [0, 0, 1]], dtype=np.float64)

    image_pts = np.array([
        [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
        for i in _LANDMARK_INDICES
    ], dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        _FACE_3D_MODEL, image_pts, cam, np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if ok:
        axis_len = 80.0
        axes_3d = np.array([
            [axis_len, 0, 0],   # X (파랑)
            [0, axis_len, 0],   # Y (초록)
            [0, 0, axis_len],   # Z (빨강)
        ], dtype=np.float64)
        projected, _ = cv2.projectPoints(axes_3d, rvec, tvec, cam, np.zeros((4, 1)))
        for pt, color in zip(projected, [(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
            cv2.arrowedLine(vis, nose_pt,
                            (int(pt[0][0]), int(pt[0][1])), color, 2, tipLength=0.2)

    # 4) 감정 확률 막대그래프 (우측 상단)
    scores = output.get("emotion_scores") or {}
    bar_x, bar_y = w - 200, 20
    bar_w, bar_h = 160, 14
    for i, label in enumerate(EMOTION_LABELS):
        prob = scores.get(label, 0.0)
        y = bar_y + i * (bar_h + 4)
        cv2.rectangle(vis, (bar_x, y), (bar_x + bar_w, y + bar_h), (50, 50, 50), -1)
        fill = int(prob * bar_w)
        color = (0, 200, 0) if label == output.get("emotion") else (180, 180, 180)
        cv2.rectangle(vis, (bar_x, y), (bar_x + fill, y + bar_h), color, -1)
        cv2.putText(vis, f"{label:<9} {prob:.2f}", (bar_x - 5, y + bar_h - 3),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1)

    # 5) 수치 텍스트 (좌측 상단)
    hp = output.get("head_pose") or {}
    lines = [
        f"emotion : {output.get('emotion')} ({output.get('confidence', 0):.2f})",
        f"yaw     : {hp.get('yaw', 0):+.1f} deg",
        f"pitch   : {hp.get('pitch', 0):+.1f} deg",
        f"roll    : {hp.get('roll', 0):+.1f} deg",
        f"peak    : {output.get('peak_emotion')} ({output.get('peak_confidence', 0):.2f})",
    ]
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (12, 30 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    return vis


def test_debug_webcam(model_path: str | None = None, camera: int = 0) -> None:
    """랜드마크·head pose·감정을 화면에 그려 실시간 확인."""
    import mediapipe as mp_inner

    pipeline = VisionPipeline(model_path=model_path)
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print(f"[ERROR] 카메라를 열 수 없습니다 (index={camera})")
        return

    print("[DEBUG] 랜드마크 시각화 시작 — q 키로 종료")
    face_mesh_debug = mp_inner.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh_debug.process(rgb)
            landmarks = results.multi_face_landmarks[0] if results.multi_face_landmarks else None

            output = pipeline.process_frame(frame.copy())
            vis = _draw_debug_overlay(frame, output, landmarks)

            cv2.imshow("Vision Debug — q to quit", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            elapsed = time.time() - t0
            sleep = pipeline.FRAME_INTERVAL - elapsed
            if sleep > 0:
                time.sleep(sleep)
    finally:
        cap.release()
        face_mesh_debug.close()
        pipeline._face_mesh.close()
        cv2.destroyAllWindows()


def test_debug_image(image_path: str, model_path: str | None = None) -> None:
    """정적 이미지에 랜드마크 오버레이를 그려 창으로 표시."""
    import mediapipe as mp_inner

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] 이미지를 읽을 수 없습니다: {image_path}")
        return

    pipeline = VisionPipeline(model_path=model_path)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_mesh_debug = mp_inner.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    results = face_mesh_debug.process(rgb)
    landmarks = results.multi_face_landmarks[0] if results.multi_face_landmarks else None
    face_mesh_debug.close()

    output = pipeline.process_frame(frame.copy())
    print(json.dumps(output, ensure_ascii=False, indent=2))

    vis = _draw_debug_overlay(frame, output, landmarks)
    cv2.imshow(f"Debug: {image_path} — any key to close", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pipeline._face_mesh.close()


def test_image(image_path: str, model_path: str | None = None, device: str | None = None) -> None:
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] 이미지를 읽을 수 없습니다: {image_path}")
        return

    pipeline = VisionPipeline(model_path=model_path, device=device)
    output = pipeline.process_frame(frame)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    pipeline._face_mesh.close()


def test_mock() -> None:
    """카메라·모델 없이 출력 형식 및 스펙 일치 여부만 검증"""

    now = round(time.time(), 3)

    detected = {
        "timestamp": now,
        "face_detected": True,
        "emotion": "sad",
        "confidence": 0.72,
        "emotion_scores": {
            "sad": 0.72, "angry": 0.10, "happy": 0.05,
            "neutral": 0.08, "surprised": 0.03, "fearful": 0.02,
        },
        "head_pose": {"yaw": -12.3, "pitch": 5.1, "roll": 2.0},
        "peak_emotion": "fearful",
        "peak_confidence": 0.74,
        "peak_detected_at": round(now - 3.0, 3),
    }

    no_face = {
        "timestamp": now,
        "face_detected": False,
        "emotion": None,
        "confidence": None,
        "emotion_scores": None,
        "head_pose": None,
        "peak_emotion": None,
        "peak_confidence": 0.0,
        "peak_detected_at": None,
    }

    print("=== [MOCK] face_detected: true ===")
    print(json.dumps(detected, ensure_ascii=False, indent=2))
    print("\n=== [MOCK] face_detected: false ===")
    print(json.dumps(no_face, ensure_ascii=False, indent=2))

    _validate_spec(detected)
    print("\n스펙 검증 통과")


def test_peak_reset() -> None:
    """reset_peak 동작 확인 (단위 테스트)"""
    pipeline = VisionPipeline()

    # 가짜 peak 주입
    pipeline._peak_emotion = "angry"
    pipeline._peak_confidence = 0.9
    pipeline._peak_detected_at = time.time()

    pipeline.reset_peak()

    assert pipeline._peak_emotion is None
    assert pipeline._peak_confidence == 0.0
    assert pipeline._peak_detected_at is None
    print("reset_peak 테스트 통과")
    pipeline._face_mesh.close()


# 스펙 검증 헬퍼 함수

def _validate_spec(output: dict) -> None:
    required = {
        "timestamp", "face_detected", "emotion", "confidence",
        "emotion_scores", "head_pose",
        "peak_emotion", "peak_confidence", "peak_detected_at",
    }
    missing = required - set(output.keys())
    assert not missing, f"누락된 필드: {missing}"

    if output["face_detected"]:
        assert output["emotion"] in EMOTION_LABELS, \
            f"잘못된 emotion 레이블: {output['emotion']}"
        assert set(output["emotion_scores"].keys()) == set(EMOTION_LABELS), \
            "emotion_scores 레이블 불일치"
        score_sum = sum(output["emotion_scores"].values())
        assert abs(score_sum - 1.0) < 0.01, f"emotion_scores 합계 != 1 (got {score_sum:.4f})"
        assert isinstance(output["head_pose"], dict), "head_pose가 dict가 아님"
        assert set(output["head_pose"].keys()) == {"yaw", "pitch", "roll"}, \
            "head_pose 필드 불일치"
    else:
        for field in ("emotion", "confidence", "emotion_scores", "head_pose"):
            assert output[field] is None, f"face_detected=false 인데 {field}가 null이 아님"


# ResNet에 들어가는 얼굴 크롭 이미지와 텐서를 시각적으로 확인하는 테스트

def test_show_crop(source: str | None, model_path: str | None, camera: int, save: str | None) -> None:
    """
    ResNet에 실제로 들어가는 이미지를 확인한다.

    출력:
      - 얼굴 크롭 원본 (PIL, 가변 크기) — 창으로 표시
      - 정규화 해제된 텐서 이미지 (224×224) — 창으로 표시
      - 텐서 shape / min / max / mean 출력

    사용:
      python vision_test.py --crop                      # 웹캠 1프레임
      python vision_test.py --crop --image face.jpg     # 정적 이미지
      python vision_test.py --crop --save crop_out.png  # 파일로 저장
    """
    import torch
    from PIL import Image as PILImage
    import mediapipe as mp_inner

    if source:
        frame = cv2.imread(source)
        if frame is None:
            print(f"[ERROR] 이미지를 읽을 수 없습니다: {source}")
            return
    else:
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            print(f"[ERROR] 카메라를 열 수 없습니다 (index={camera})")
            return
        print("웹캠에서 프레임 캡처 중... (스페이스바로 캡처)")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imshow("Capture — SPACE to grab, q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return
        cap.release()
        cv2.destroyAllWindows()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    face_mesh = mp_inner.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    results = face_mesh.process(rgb)
    face_mesh.close()

    if not results.multi_face_landmarks:
        print("[ERROR] 얼굴을 감지하지 못했습니다.")
        return

    landmarks = results.multi_face_landmarks[0]

    xs = [lm.x * w for lm in landmarks.landmark]
    ys = [lm.y * h for lm in landmarks.landmark]
    pad = 20
    x1 = int(max(min(xs) - pad, 0))
    x2 = int(min(max(xs) + pad, w))
    y1 = int(max(min(ys) - pad, 0))
    y2 = int(min(max(ys) + pad, h))
    crop_pil = PILImage.fromarray(rgb[y1:y2, x1:x2])

    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(crop_pil)  # (3, 224, 224)

    print(f"\n[ResNet 입력 텐서]")
    print(f"  shape : {tuple(tensor.shape)}")
    print(f"  dtype : {tensor.dtype}")
    print(f"  min   : {tensor.min().item():.4f}")
    print(f"  max   : {tensor.max().item():.4f}")
    print(f"  mean  : {tensor.mean().item():.4f}")
    print(f"  std   : {tensor.std().item():.4f}")
    print(f"\n[얼굴 크롭 원본]")
    print(f"  크기  : {crop_pil.size}  (W×H, PIL)")
    print(f"  bbox  : x({x1}~{x2}), y({y1}~{y2}), padding={pad}px")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denorm = (tensor * std + mean).clamp(0, 1)
    denorm_np = (denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    denorm_bgr = cv2.cvtColor(denorm_np, cv2.COLOR_RGB2BGR)

    crop_np = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)

    if save:
        cv2.imwrite(save, denorm_bgr)
        print(f"\n저장 완료: {save}")

    cv2.imshow(f"1) 얼굴 크롭 원본  {crop_pil.size[0]}x{crop_pil.size[1]}", crop_np)
    cv2.imshow("2) ResNet 입력 (224x224, 정규화 해제)", denorm_bgr)
    print("\n아무 키나 누르면 창이 닫힙니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Pipeline 테스트")
    parser.add_argument("--image", type=str, default=None, help="테스트할 이미지 경로")
    parser.add_argument("--debug", action="store_true",
                        help="랜드마크·head pose·감정 막대그래프를 창으로 시각화")
    parser.add_argument("--mock", action="store_true", help="mock 출력 + 스펙 검증")
    parser.add_argument("--unit", action="store_true", help="단위 테스트 (reset_peak 등)")
    parser.add_argument("--crop", action="store_true",
                        help="ResNet에 들어가는 얼굴 크롭 이미지를 확인")
    parser.add_argument("--save", type=str, default=None, help="--crop 결과를 파일로 저장")
    parser.add_argument("--model", type=str, default=None, help="ResNet 가중치 .pth 경로")
    parser.add_argument("--device", type=str, default=None, help="실행 장치: cpu | cuda")
    parser.add_argument("--camera", type=int, default=0, help="카메라 인덱스 (기본 0)")
    args = parser.parse_args()

    if args.mock:
        test_mock()
    elif args.unit:
        test_peak_reset()
    elif args.crop:
        test_show_crop(source=args.image, model_path=args.model, camera=args.camera, save=args.save)
    elif args.image and args.debug:
        test_debug_image(args.image, model_path=args.model)
    elif args.image:
        test_image(args.image, model_path=args.model, device=args.device)
    elif args.debug:
        test_debug_webcam(model_path=args.model, camera=args.camera)
    else:
        test_webcam(model_path=args.model, camera=args.camera)
