# app.py
# People Capture Hybrid: YOLOv8 (+seg OK) + MediaPipe Pose (+Hands, wrist-gated) + Face+Iris (head-gated) + Hysteresis Tracker
# Run:
#   python app.py --camera 0 --width 1280 --height 720
#   python app.py --device cuda:0   # force GPU if available
#   python app.py --no-yolo         # pose-only

import argparse
import time
import warnings
from typing import List, Tuple, Optional

import cv2
import numpy as np

# ---- Optional YOLO import ----
HAS_YOLO = True
try:
    from ultralytics import YOLO
except Exception as e:
    HAS_YOLO = False
    YOLO_IMPORT_ERR = e

# ---- Optional torch for device check ----
HAS_TORCH = True
try:
    import torch
except Exception:
    HAS_TORCH = False

# ---- MediaPipe ----
import mediapipe as mp
mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

# ---- Gating constants ----
WRIST_VIS_TH = 0.6         # wrist visibility to trigger Hands
HANDS_HOLD_FRAMES = 10     # keep Hands on for N frames after trigger
FACE_HOLD_FRAMES = 10      # keep Face on for N frames after trigger
HEAD_MIN_PIX = 160         # min track box height to trigger Face (approx head close enough)

# ---- Tracker hysteresis ----
HIT_TH = 3      # frames to confirm a new track
MISS_TH = 6     # frames to delete a lost track
IOU_TH = 0.35   # association threshold
EMA_ALPHA = 0.7 # bbox/score smoothing


def resolve_device(arg: str = "auto") -> str:
    if arg != "auto":
        if arg.startswith("cuda") and not (HAS_TORCH and torch.cuda.is_available()):
            warnings.warn("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return arg
    if HAS_TORCH and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def xyxy_to_int(box: np.ndarray) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.astype(int).tolist()
    return x1, y1, x2, y2


def landmark_in_box(lm, box, img_w: int, img_h: int) -> bool:
    x1, y1, x2, y2 = box
    x = lm.x * img_w
    y = lm.y * img_h
    return x1 <= x <= x2 and y1 <= y <= y2


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = ua + ub - inter
    return float(inter / union) if union > 0 else 0.0


class Track:
    __slots__ = ("id", "box", "score", "hits", "misses", "confirmed")

    def __init__(self, tid: int, box: np.ndarray, score: float):
        self.id = tid
        self.box = box.astype(np.float32)
        self.score = float(score)
        self.hits = 1
        self.misses = 0
        self.confirmed = False

    def update(self, box: np.ndarray, score: float):
        self.box = EMA_ALPHA * box.astype(np.float32) + (1.0 - EMA_ALPHA) * self.box
        self.score = EMA_ALPHA * float(score) + (1.0 - EMA_ALPHA) * self.score
        self.hits += 1
        self.misses = 0
        if not self.confirmed and self.hits >= HIT_TH:
            self.confirmed = True

    def mark_missed(self):
        self.misses += 1


class Tracker:
    def __init__(self, iou_th: float = IOU_TH):
        self.iou_th = iou_th
        self.tracks: List[Track] = []
        self.next_id = 1

    def _match(self, dets: List[Tuple[np.ndarray, float]]):
        matches = []
        unmatched_t = list(range(len(self.tracks)))
        unmatched_d = list(range(len(dets)))
        if not self.tracks or not dets:
            return matches, unmatched_t, unmatched_d

        iou_matrix = np.zeros((len(self.tracks), len(dets)), dtype=np.float32)
        for ti, t in enumerate(self.tracks):
            for di, (dbox, _) in enumerate(dets):
                iou_matrix[ti, di] = box_iou(t.box, dbox)

        while True:
            ti, di = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[ti, di] < self.iou_th:
                break
            matches.append((ti, di))
            iou_matrix[ti, :] = -1.0
            iou_matrix[:, di] = -1.0
            unmatched_t.remove(ti)
            unmatched_d.remove(di)
            if (iou_matrix > self.iou_th).sum() == 0:
                break
        return matches, unmatched_t, unmatched_d

    def update(self, dets: List[Tuple[np.ndarray, float]]) -> List[Track]:
        matches, unmatched_t, unmatched_d = self._match(dets)

        for ti, di in matches:
            box, score = dets[di]
            self.tracks[ti].update(box, score)

        for ti in unmatched_t:
            self.tracks[ti].mark_missed()

        for di in unmatched_d:
            box, score = dets[di]
            self.tracks.append(Track(self.next_id, box, score))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if t.misses < MISS_TH]
        return [t for t in self.tracks if t.confirmed]


def fuse(yolo_dets: List[Tuple[np.ndarray, float]],
         pose_result,
         img_w: int,
         img_h: int,
         min_visible_inside: int = 8) -> List[Tuple[np.ndarray, float]]:
    lms = pose_result.pose_landmarks.landmark if pose_result and pose_result.pose_landmarks else []
    visible = [lm for lm in lms if getattr(lm, "visibility", 0.0) > 0.5]
    pose_score = (len(visible) / len(lms)) if lms else 0.0

    fused = []
    for box, score in yolo_dets:
        inside = sum(1 for lm in visible if landmark_in_box(lm, box, img_w, img_h))
        confirmed = inside >= min_visible_inside
        fused_score = (0.7 * score + 0.3 * pose_score) if confirmed else (0.5 * score)
        fused.append((box, float(fused_score)))
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


def draw_pose(frame, pose_result):
    if pose_result and pose_result.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_draw.DrawingSpec(color=(120, 180, 255), thickness=2)
        )


def draw_hands(frame, hands_result):
    if not hands_result or not hands_result.multi_hand_landmarks:
        return
    for hand_landmarks in hands_result.multi_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_draw.DrawingSpec(color=(255, 200, 120), thickness=2)
        )


def draw_face(frame, face_result):
    if not face_result or not face_result.multi_face_landmarks:
        return
    for fl in face_result.multi_face_landmarks:
        mp_draw.draw_landmarks(
            image=frame,
            landmark_list=fl,
            connections=mp_face.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_draw.DrawingSpec(color=(200, 200, 255), thickness=1)
        )


def draw_tracks(frame, tracks: List[Track]):
    for t in tracks:
        x1, y1, x2, y2 = xyxy_to_int(t.box)
        color = (0, 210, 255)  # cyan
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {t.id}  {t.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        yb = max(0, y1 - th - 4)
        cv2.rectangle(frame, (x1, yb), (x1 + tw + 6, yb + th + 6), (0, 0, 0), -1)
        cv2.putText(frame, label, (x1 + 3, yb + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)


def get_people_from_yolo(result, img_w, img_h, person_class=0, conf_th=0.35):
    dets = []
    try:
        boxes = result.boxes
        if boxes is None:
            return dets
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        cls  = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
        for b, s, c in zip(xyxy, conf, cls):
            if int(c) == person_class and float(s) >= conf_th:
                x1 = int(max(0, min(img_w - 1, b[0])))
                y1 = int(max(0, min(img_h - 1, b[1])))
                x2 = int(max(0, min(img_w - 1, b[2])))
                y2 = int(max(0, min(img_h - 1, b[3])))
                dets.append((np.array([x1, y1, x2, y2], dtype=np.float32), float(s)))
    except Exception:
        pass
    return dets


def wrist_visible(pose_res) -> bool:
    if not pose_res or not pose_res.pose_landmarks:
        return False
    lm = pose_res.pose_landmarks.landmark
    # Pose indices: 15 = left_wrist, 16 = right_wrist
    return (getattr(lm[15], "visibility", 0.0) > WRIST_VIS_TH) or \
           (getattr(lm[16], "visibility", 0.0) > WRIST_VIS_TH)


def main():
    p = argparse.ArgumentParser(description="People Capture Hybrid: YOLOv8 + Pose/Hands/Face + Hysteresis")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda:0", "cuda:1"])
    p.add_argument("--model", default="yolov8n-seg.pt", help="Ultralytics model path (seg or det).")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--flip", action="store_true", help="Flip webcam horizontally.")
    p.add_argument("--no-yolo", action="store_true", help="Disable YOLO. Pose-only.")
    p.add_argument("--no-hands", action="store_true", help="Disable MediaPipe Hands.")
    p.add_argument("--no-face", action="store_true", help="Disable MediaPipe FaceMesh.")
    args = p.parse_args()

    DEVICE = resolve_device(args.device)
    ULTRA_DEVICE = 0 if DEVICE.startswith("cuda") else "cpu"

    if args.no_yolo or not HAS_YOLO:
        if not HAS_YOLO:
            warnings.warn(f"Ultralytics not available: {YOLO_IMPORT_ERR}")
        yolo_model = None
    else:
        try:
            yolo_model = YOLO(args.model)
        except Exception as e:
            warnings.warn(f"Failed to load YOLO model '{args.model}': {e}\nContinuing without YOLO.")
            yolo_model = None

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
    )
    hands = None
    if not args.no_hands:
        hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    face = None
    if not args.no_face:
        face = mp_face.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,  # iris
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    hands_budget = 0
    face_budget = 0

    tracker = Tracker(IOU_TH)

    prev_t = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.flip:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        # YOLO (optional)
        yolo_dets = []
        if yolo_model is not None:
            try:
                r = yolo_model.predict(source=frame, device=ULTRA_DEVICE, conf=args.conf, verbose=False)[0]
                yolo_dets = get_people_from_yolo(r, w, h, person_class=0, conf_th=args.conf)
            except Exception as e:
                warnings.warn(f"YOLO inference failed: {e}")
                yolo_dets = []

        # Pose
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb)

        # Wrist-gated Hands
        if wrist_visible(pose_res):
            hands_budget = HANDS_HOLD_FRAMES

        hands_res = None
        if hands is not None and hands_budget > 0:
            hands_res = hands.process(rgb)
            hands_budget -= 1

        # Fusion → Tracker
        dets_for_tracking: List[Tuple[np.ndarray, float]] = []
        if yolo_dets:
            dets_for_tracking = fuse(yolo_dets, pose_res, w, h, min_visible_inside=8)
        confirmed_tracks = tracker.update(dets_for_tracking)

        # Head-size gate for Face: if any confirmed track tall enough
        if confirmed_tracks:
            if any((t.box[3] - t.box[1]) >= HEAD_MIN_PIX for t in confirmed_tracks):
                face_budget = FACE_HOLD_FRAMES

        face_res = None
        if face is not None and face_budget > 0:
            face_res = face.process(rgb)
            face_budget -= 1

        # Draw
        draw_pose(frame, pose_res)
        if hands_res:
            draw_hands(frame, hands_res)
        if face_res:
            draw_face(frame, face_res)
        draw_tracks(frame, confirmed_tracks)

        # HUD
        now = time.time()
        dt = now - prev_t
        prev_t = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
        hud = f"FPS: {fps:.1f}  people: {len(confirmed_tracks)}  hands:{hands_budget>0}  face:{face_budget>0}"
        cv2.putText(frame, hud, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)

        cv2.imshow("People Capture Hybrid", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    pose.close()
    if hands:
        hands.close()
    if face:
        face.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
