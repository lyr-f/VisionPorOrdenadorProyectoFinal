import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import cv2
import numpy as np
import mediapipe as mp
import os
import csv


CSV_PATH = "data/repetitions.csv"  # Número de repeticiones configuradas

# Model path
MODEL_PATH = "data/pose_landmarker_full.task"  # Obtenido con MediaPipe

# Camera
CAMERA_INDEX = 0

# Thresholds
SQUAT_DOWN_ANGLE = 100   # knee: smaller = deeper squat
SQUAT_UP_ANGLE = 160

PUSHUP_DOWN_ANGLE = 95   # elbow
PUSHUP_UP_ANGLE = 160

# Se mantiene por compatibilidad (antes era salto vertical). Ahora se usa SOLO el cooldown.
JUMP_MIN_FRAMES_BETWEEN_COUNTS = 8
JUMP_LIFT_DELTA = 0.08
JUMP_RETURN_DELTA = 0.03

# Jumping Jacks thresholds (normalizados por anchura de hombros/caderas)
JACK_LEG_OPEN_RATIO = 1.60     # distancia entre tobillos / ref para considerar piernas "abiertas"
JACK_LEG_CLOSE_RATIO = 1.20    # para volver a "cerrado" (histeresis)

JACK_ARM_OPEN_RATIO = 1.60     # distancia entre muñecas / ref para considerar brazos "abiertos"
JACK_ARM_CLOSE_RATIO = 1.20    # para volver a "cerrado" (histeresis)

JACK_WRIST_ABOVE_SHOULDER_MARGIN = 0.02  # muñeca por encima del hombro (en y normalizada; y menor = más arriba)

# Visual
WINDOW_NAME = "Workout - MediaPipe Pose (LIVE_STREAM)"
PINK_BGR = (196, 196, 255)  # BGR

# Finish behavior
AUTO_EXIT_ON_FINISH = False          # if True, exit automatically after FINISH_HOLD_SECONDS
FINISH_HOLD_SECONDS = 3.0            # time to keep the pink screen before auto exit
FINISH_HOLD_STARTS_ON_FIRST_FRAME = True

# Mirror mode (selfie view)
MIRROR_MODE = True


def load_workout_plan_from_csv(csv_path: str = CSV_PATH) -> Dict[str, int]:
    """
    Reads data/repetitions.csv format:
    exercise,reps
    pushups,0
    squats,5
    jumps,1

    Returns:
    {
      "pushups": 0,
      "squats": 5,
      "jumps": 1
    }
    """
    plan: Dict[str, int] = {
        "squats": 0,
        "pushups": 0,
        "jumps": 0,
    }

    if not os.path.exists(csv_path):
        return plan

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ex = (row.get("exercise") or "").strip().lower()
            reps_raw = (row.get("reps") or "").strip()

            if ex not in plan:
                continue

            try:
                reps = int(reps_raw)
            except ValueError:
                reps = 0

            reps = max(0, reps)
            plan[ex] = reps

    return plan


WORKOUT_PLAN: Dict[str, int] = load_workout_plan_from_csv()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees, with b as the vertex."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba) + 1e-9
    nbc = np.linalg.norm(bc) + 1e-9
    cosang = float(np.dot(ba, bc) / (nba * nbc))
    cosang = _clamp(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def lm_xy(landmarks, idx: int) -> np.ndarray:
    """Return normalized (x, y) in [0..1] as numpy array."""
    lm = landmarks[idx]
    return np.array([lm.x, lm.y], dtype=np.float32)


# Estados del ejercicio

@dataclass
class ExerciseState:
    count: int = 0
    phase: str = "start"  # "up"/"down"/"closed"/"open"/...
    last_count_frame: int = 0
    jump_baseline: Optional[float] = None
    jump_baseline_alpha: float = 0.05


class WorkoutCounter:
    def __init__(self, plan: Dict[str, int]):
        self.plan = plan
        self.exercises: List[str] = [k for k, v in plan.items() if v > 0]
        self.states: Dict[str, ExerciseState] = {k: ExerciseState() for k in plan.keys()}
        self.current_idx: int = 0
        self.finished: bool = (len(self.exercises) == 0)

    @property
    def current_exercise(self) -> Optional[str]:
        if self.finished or self.current_idx >= len(self.exercises):
            return None
        return self.exercises[self.current_idx]

    def target(self, ex: str) -> int:
        return int(self.plan.get(ex, 0))

    def is_done_ex(self, ex: str) -> bool:
        return self.states[ex].count >= self.target(ex) and self.target(ex) > 0

    def update_progress(self):
        """If current exercise reached the goal, advance."""
        while True:
            cur = self.current_exercise
            if cur is None:
                self.finished = True
                return
            if self.is_done_ex(cur):
                self.current_idx += 1
                if self.current_idx >= len(self.exercises):
                    self.finished = True
                    return
                continue
            break


# Lógica de repetición
def update_squat(counter: WorkoutCounter, landmarks, frame_id: int):
    ex = "squats"
    if counter.target(ex) <= 0 or counter.is_done_ex(ex):
        return

    # MediaPipe Pose landmark indices:
    # left hip, knee, ankle and right hip, knee, ankle
    L_HIP, L_KNEE, L_ANK = 23, 25, 27
    R_HIP, R_KNEE, R_ANK = 24, 26, 28

    left_angle = angle_degrees(lm_xy(landmarks, L_HIP), lm_xy(landmarks, L_KNEE), lm_xy(landmarks, L_ANK))
    right_angle = angle_degrees(lm_xy(landmarks, R_HIP), lm_xy(landmarks, R_KNEE), lm_xy(landmarks, R_ANK))
    knee_angle = (left_angle + right_angle) / 2.0

    st = counter.states[ex]

    if st.phase == "start":
        st.phase = "up" if knee_angle > SQUAT_UP_ANGLE else "down"

    # down -> up counts 1
    if st.phase == "up":
        if knee_angle < SQUAT_DOWN_ANGLE:
            st.phase = "down"
    elif st.phase == "down":
        if knee_angle > SQUAT_UP_ANGLE:
            st.count += 1
            st.phase = "up"


def update_pushup(counter: WorkoutCounter, landmarks, frame_id: int):
    ex = "pushups"
    if counter.target(ex) <= 0 or counter.is_done_ex(ex):
        return

    # shoulders, elbows, wrists
    L_SH, L_EL, L_WR = 11, 13, 15
    R_SH, R_EL, R_WR = 12, 14, 16

    left_angle = angle_degrees(lm_xy(landmarks, L_SH), lm_xy(landmarks, L_EL), lm_xy(landmarks, L_WR))
    right_angle = angle_degrees(lm_xy(landmarks, R_SH), lm_xy(landmarks, R_EL), lm_xy(landmarks, R_WR))
    elbow_angle = (left_angle + right_angle) / 2.0

    st = counter.states[ex]
    if st.phase == "start":
        st.phase = "up" if elbow_angle > PUSHUP_UP_ANGLE else "down"

    if st.phase == "up":
        if elbow_angle < PUSHUP_DOWN_ANGLE:
            st.phase = "down"
    elif st.phase == "down":
        if elbow_angle > PUSHUP_UP_ANGLE:
            st.count += 1
            st.phase = "up"


def update_jump(counter: WorkoutCounter, landmarks, frame_id: int):
    """
    Jumping jacks:
    - "open" cuando piernas abiertas + brazos abiertos (muñecas separadas y por encima de hombros)
    - cuenta 1 rep cuando hace: closed -> open -> closed
    """
    ex = "jumps"
    if counter.target(ex) <= 0 or counter.is_done_ex(ex):
        return

    # Indices MediaPipe Pose
    L_SH, R_SH = 11, 12
    L_HIP, R_HIP = 23, 24
    L_WR, R_WR = 15, 16
    L_ANK, R_ANK = 27, 28

    l_sh = lm_xy(landmarks, L_SH)
    r_sh = lm_xy(landmarks, R_SH)
    l_hip = lm_xy(landmarks, L_HIP)
    r_hip = lm_xy(landmarks, R_HIP)
    l_wr = lm_xy(landmarks, L_WR)
    r_wr = lm_xy(landmarks, R_WR)
    l_ank = lm_xy(landmarks, L_ANK)
    r_ank = lm_xy(landmarks, R_ANK)

    shoulder_w = float(np.linalg.norm(l_sh - r_sh))
    hip_w = float(np.linalg.norm(l_hip - r_hip))
    ref = max(shoulder_w, hip_w, 1e-6)

    wrist_dist_ratio = float(np.linalg.norm(l_wr - r_wr) / ref)
    ankle_dist_ratio = float(np.linalg.norm(l_ank - r_ank) / ref)

    shoulder_y = float((l_sh[1] + r_sh[1]) / 2.0)
    wrists_y = float((l_wr[1] + r_wr[1]) / 2.0)

    wrists_above_shoulders = wrists_y < (shoulder_y - JACK_WRIST_ABOVE_SHOULDER_MARGIN)

    arms_open = (wrist_dist_ratio >= JACK_ARM_OPEN_RATIO) and wrists_above_shoulders
    arms_closed = (wrist_dist_ratio <= JACK_ARM_CLOSE_RATIO) and (wrists_y > (shoulder_y - JACK_WRIST_ABOVE_SHOULDER_MARGIN * 0.5))

    legs_open = ankle_dist_ratio >= JACK_LEG_OPEN_RATIO
    legs_closed = ankle_dist_ratio <= JACK_LEG_CLOSE_RATIO

    is_open = arms_open and legs_open
    is_closed = arms_closed and legs_closed

    st = counter.states[ex]

    if st.phase == "start":
        # Si empieza ya abierto, lo ponemos en open. Si no, cerrado.
        st.phase = "open" if is_open else "closed"
        st.last_count_frame = -10**9

    if st.phase == "closed":
        # Espera a detectar abierto completo
        if is_open:
            st.phase = "open"

    elif st.phase == "open":
        # Cuenta cuando vuelve a cerrado (ciclo completo)
        if is_closed:
            if frame_id - st.last_count_frame >= JUMP_MIN_FRAMES_BETWEEN_COUNTS:
                st.count += 1
                st.last_count_frame = frame_id
            st.phase = "closed"


def update_reps_for_current_exercise(counter: WorkoutCounter, landmarks, frame_id: int):
    cur = counter.current_exercise
    if cur is None:
        return

    if cur == "squats":
        update_squat(counter, landmarks, frame_id)
    elif cur == "pushups":
        update_pushup(counter, landmarks, frame_id)
    elif cur == "jumps":
        update_jump(counter, landmarks, frame_id)

    counter.update_progress()


# Funciones para mostrar el texto en la ventana

def draw_hud(frame_bgr: np.ndarray, counter: WorkoutCounter):
    x0, y0 = 20, 30
    line_h = 28

    cur = counter.current_exercise
    title = f"Current exercise: {cur}" if cur else ("COMPLETED" if counter.finished else "No exercise")
    cv2.putText(frame_bgr, title, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame_bgr, title, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    y = y0 + 40
    for ex, target in counter.plan.items():
        done = counter.states[ex].count
        txt = f"{ex}: {done}/{target}"
        scale = 0.75 if ex != cur else 0.85

        cv2.putText(frame_bgr, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame_bgr, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
        y += line_h


def apply_segmentation_pink_background(frame_bgr: np.ndarray, segmentation_mask: np.ndarray) -> np.ndarray:
    """segmentation_mask: float32 [0..1], shape (H, W)"""
    h, w = frame_bgr.shape[:2]
    mask = cv2.resize(segmentation_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), 2.5)

    fg = frame_bgr.astype(np.float32)
    bg = np.full_like(fg, PINK_BGR, dtype=np.float32)

    out = fg * mask[..., None] + bg * (1.0 - mask[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_congrats(frame_bgr: np.ndarray):
    h, w = frame_bgr.shape[:2]
    msg = "Congrats! You finished your workout."
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 3
    (tw, th), _ = cv2.getTextSize(msg, font, scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2

    cv2.putText(frame_bgr, msg, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame_bgr, msg, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


# Código principal

_latest_result = None
_latest_segmask = None
_latest_timestamp_ms = 0


def _result_callback(result, output_image: mp.Image, timestamp_ms: int):
    global _latest_result, _latest_segmask, _latest_timestamp_ms
    _latest_result = result
    _latest_timestamp_ms = timestamp_ms

    seg = None
    if result is not None and getattr(result, "segmentation_masks", None):
        seg_img = result.segmentation_masks[0]
        seg = np.array(seg_img.numpy_view(), dtype=np.float32)
    _latest_segmask = seg


def _window_was_closed(window_name: str) -> bool:
    """
    If the user clicks the X, some OpenCV backends keep the loop running unless you check this.
    Returns True if the window is gone/closed.
    """
    try:
        prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        return prop < 1
    except cv2.error:
        return True


def detect_exercise():
    global _latest_result, _latest_segmask

    counter = WorkoutCounter(WORKOUT_PLAN)

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=True,
        result_callback=_result_callback,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open the camera (index={CAMERA_INDEX}).")

    frame_id = 0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    finished_at = None  # for auto-exit timer

    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            # Allow closing by clicking the X
            if _window_was_closed(WINDOW_NAME):
                break

            ok, frame_bgr = cap.read()
            if not ok:
                break

            # Mirror mode (selfie view)
            if MIRROR_MODE:
                frame_bgr = cv2.flip(frame_bgr, 1)

            frame_id += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int(time.time() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            out = frame_bgr.copy()

            if _latest_result is not None and getattr(_latest_result, "pose_landmarks", None):
                if len(_latest_result.pose_landmarks) > 0:
                    landmarks = _latest_result.pose_landmarks[0]
                    if not counter.finished:
                        update_reps_for_current_exercise(counter, landmarks, frame_id)

            if counter.finished:
                # start timer the first time we hit finished
                if finished_at is None and FINISH_HOLD_STARTS_ON_FIRST_FRAME:
                    finished_at = time.time()

                if _latest_segmask is not None:
                    out = apply_segmentation_pink_background(out, _latest_segmask)
                else:
                    out[:] = PINK_BGR

                draw_congrats(out)

                if AUTO_EXIT_ON_FINISH and finished_at is not None:
                    if (time.time() - finished_at) >= FINISH_HOLD_SECONDS:
                        break
            else:
                draw_hud(out, counter)

            cv2.imshow(WINDOW_NAME, out)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_exercise()
