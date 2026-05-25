import cv2
import os
import re
import json
import argparse
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from typing import Optional
import numpy as np

# Poti do videov, modelov in logov
DATA_DIR = "data"
OUTPUT_DIR = "output/videos"
LOG_DIR = "output/logs"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
CALIBRATION_CONF_DIR = "calibration/conf"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Povezave med značilkami roke
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

# Mapiranje oznak kamer na ime konfiguracije
CAMERA_MAP = {
    "camP_0": "left",
    "camP_1": "mid",
    "camP_2": "right",
}

# ROI (Region of Interest) za vsako kamero kot delež slike (x1, y1, x2, y2)
CAMERA_ROI = {
    "left":  (0.25, 0.05, 0.75, 0.85),
    "mid":   (0.25, 0.10, 0.75, 0.80),
    "right": (0.20, 0.15, 0.75, 0.80),
    None:    (1.0, 1.0, 1.0, 1.0),
}


def _dist2d(a, b):
    """Evklidska razdalja med dvema 2D točkama."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class ActiveHandSelector:
    """
    Izbere aktivno roko med več zaznanimi rokami glede na gibanje.
    Po lock_after okvirjih zaklene izbiro – motnje so ignorirane.
    """

    def __init__(self, history_len=20, lock_after=30):
        self.history_len  = history_len
        self.lock_after   = lock_after
        self.slot_positions = [
            deque(maxlen=history_len),
            deque(maxlen=history_len),
        ]
        self.active_slot = None  # type: Optional[int]
        self.locked      = False
        self.frames_seen = 0

    def _activity_score(self, slot_idx):
        hist = list(self.slot_positions[slot_idx])
        if len(hist) < 2:
            return 0.0
        return sum(_dist2d(hist[j], hist[j - 1]) for j in range(1, len(hist)))

    def _assign_to_slots(self, positions):
        n = len(positions)
        if n == 0:
            return []
        if n == 1:
            slot = self.active_slot if self.active_slot is not None else 0
            return [(0, slot)]

        last = [
            self.slot_positions[s][-1] if self.slot_positions[s] else None
            for s in range(2)
        ]

        if last[0] is None and last[1] is None:
            return [(0, 0), (1, 1)]
        if last[0] is None:
            d0 = _dist2d(positions[0], last[1])
            d1 = _dist2d(positions[1], last[1])
            return [(0, 1), (1, 0)] if d0 < d1 else [(0, 0), (1, 1)]
        if last[1] is None:
            d0 = _dist2d(positions[0], last[0])
            d1 = _dist2d(positions[1], last[0])
            return [(0, 0), (1, 1)] if d0 <= d1 else [(0, 1), (1, 0)]

        cost_straight = _dist2d(positions[0], last[0]) + _dist2d(positions[1], last[1])
        cost_cross    = _dist2d(positions[0], last[1]) + _dist2d(positions[1], last[0])
        return [(0, 0), (1, 1)] if cost_straight <= cost_cross else [(0, 1), (1, 0)]

    def select(self, positions):
        # type: (list) -> Optional[int]
        if not positions:
            return None

        self.frames_seen += 1
        assignments = self._assign_to_slots(positions)
        for pos_idx, slot_idx in assignments:
            self.slot_positions[slot_idx].append(positions[pos_idx])

        if not self.locked:
            scores = [self._activity_score(0), self._activity_score(1)]
            if self.active_slot is None:
                self.active_slot = 0 if scores[0] >= scores[1] else 1
            else:
                inactive = 1 - self.active_slot
                if scores[inactive] > scores[self.active_slot] * 1.5:
                    self.active_slot = inactive

            if self.frames_seen >= self.lock_after:
                self.locked = True
                print("  [ActiveHandSelector] Aktivna roka zaklenjena na reži {} "
                      "po {} okvirjih.".format(self.active_slot, self.frames_seen))

        for pos_idx, slot_idx in assignments:
            if slot_idx == self.active_slot:
                return pos_idx

        last_known = self.slot_positions[self.active_slot]
        if last_known and positions:
            ref = last_known[-1]
            return min(range(len(positions)), key=lambda i: _dist2d(positions[i], ref))

        return 0


def download_model():
    """Prenese MediaPipe model, če še ni v mapi."""
    if not os.path.exists(MODEL_PATH):
        print("Prenasam MediaPipe model...")
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Prenos uspesen.")


def detect_camera_from_filename(filename):
    """
    Iz imena datoteke prepozna katera kamera je bila uporabljena.
    Format: patient_XXXcamP_Y_timestamp
    camP_0 = left, camP_1 = mid, camP_2 = right
    Vrne ime kamere ('left', 'mid', 'right') ali None.
    """
    basename = os.path.basename(filename)
    match = re.search(r'camP_(\d)', basename)
    if match:
        cam_key = "camP_{}".format(match.group(1))
        return CAMERA_MAP.get(cam_key, None)
    return None


def load_calibration_config(camera_name):
    """
    Naloži kalibracijsko konfiguracijo za dano kamero.
    Najprej poizkusi lastno kalibracijo, nato fallback na validacijsko datoteko.
    Vrne slovar s kalibracijskimi parametri ali None.
    """
    # Poskusi naložiti lastno kalibracijo
    config_path = os.path.join(CALIBRATION_CONF_DIR, f"{camera_name}_calibration.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print("  Naložena kalibracija iz: {}".format(config_path))
        return config

    fallback_paths = [
        os.path.join("calibration", "calibration.json"),
        "calibration.json",
    ]
    for fb_path in fallback_paths:
        if os.path.exists(fb_path):
            with open(fb_path, "r", encoding="utf-8") as f:
                all_config = json.load(f)
            if camera_name in all_config:
                ref = all_config[camera_name]
                config = {
                    "camera_name": camera_name,
                    "camera_matrix": ref["cameraMatrix"],
                    "dist_coeffs": (
                        [ref["distortionCoeffs"][0]]
                        if isinstance(ref["distortionCoeffs"][0], list)
                        else [ref["distortionCoeffs"]]
                    ),
                    "homography": ref.get("homography", None),
                    "pixels_per_mm": None,
                    "source": "validation_fallback",
                }
                if config["homography"] is not None:
                    H = np.array(config["homography"])
                    try:
                        H_inv = np.linalg.inv(H)
                        scale_x = np.linalg.norm(H_inv[:2, 0])
                        scale_y = np.linalg.norm(H_inv[:2, 1])
                        config["pixels_per_mm"] = (scale_x + scale_y) / 2.0
                    except np.linalg.LinAlgError:
                        config["pixels_per_mm"] = None

                print("  Naložena kalibracija (fallback) iz: {}".format(fb_path))
                return config

    return None


class CameraCalibration:
    """Razred za upravljanje kalibracije kamere in pretvorbo enot."""

    def __init__(self, config):
        """
        Inicializira kalibracijo iz konfiguracijske datoteke.
        config: slovar s ključi camera_matrix, dist_coeffs, homography, pixels_per_mm
        """
        self.config = config
        self.camera_matrix = np.array(config["camera_matrix"], dtype=np.float64)
        self.dist_coeffs   = np.array(config["dist_coeffs"], dtype=np.float64)
        self.homography    = (
            np.array(config["homography"], dtype=np.float64)
            if config.get("homography") else None
        )
        self.pixels_per_mm = config.get("pixels_per_mm", None)
        self.new_camera_matrix = None
        if "new_camera_matrix" in config and config["new_camera_matrix"] is not None:
            self.new_camera_matrix = np.array(config["new_camera_matrix"], dtype=np.float64)
        self.is_calibrated = True
        self.camera_name   = config.get("camera_name", "unknown")

    def undistort_frame(self, frame):
        """Odstrani distorzijo iz slike."""
        if self.new_camera_matrix is not None:
            return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def undistort_point(self, point):
        """Odstrani distorzijo iz posamezne točke (x, y) v pikslih."""
        pts = np.array([[[point[0], point[1]]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        return (undistorted[0][0][0], undistorted[0][0][1])

    def pixel_to_mm(self, distance_px):
        """Pretvori razdaljo iz pikslov v milimetre."""
        if self.pixels_per_mm is not None and self.pixels_per_mm > 0:
            return distance_px / self.pixels_per_mm
        fx = self.camera_matrix[0, 0]
        estimated_ppm = fx / 500.0
        return distance_px / estimated_ppm

    def pixel_velocity_to_mm_s(self, vel_px_s):
        """Pretvori hitrost iz px/s v mm/s."""
        return self.pixel_to_mm(vel_px_s)

    def pixel_acc_to_mm_s2(self, acc_px_s2):
        """Pretvori pospešek iz px/s² v mm/s²."""
        return self.pixel_to_mm(acc_px_s2)

    def point_to_mm_coords(self, point_px):
        """Pretvori točko v pikslih v koordinate v mm z uporabo homografije."""
        if self.homography is not None:
            pt     = np.array([point_px[0], point_px[1], 1.0])
            result = self.homography @ pt
            if abs(result[2]) > 1e-10:
                return (result[0] / result[2], result[1] / result[2])
        return (self.pixel_to_mm(point_px[0]), self.pixel_to_mm(point_px[1]))


# ==============================================================================
# BIRD'S-EYE VIEW
# ==============================================================================

def build_birds_eye_transform(homography, out_w=600, out_h=600, scale_px_per_mm=2.0):
    """
    Zgradi transformacijsko matriko za bird's-eye view iz homografije.

    Homografija H preslika točke iz slikovnega prostora (px) v realni prostor (mm).
    Zato je transformacija za warpPerspective:
        M = S @ H
    kjer S skalira mm → px v izhodnem okvirju.

    Args:
        homography:     3×3 homografija iz kalibracije (px → mm)
        out_w, out_h:   velikost izhodne slike v pikslih
        scale_px_per_mm: koliko pikslov na mm v izhodnem bird's-eye pogledu

    Returns:
        M (3×3 np.ndarray): transformacijska matrika za cv2.warpPerspective
        (out_w, out_h): velikost izhodne slike
    """
    S = np.array([
        [scale_px_per_mm, 0,               out_w / 2.0],
        [0,               scale_px_per_mm, out_h / 2.0],
        [0,               0,               1.0         ],
    ], dtype=np.float64)

    M = S @ homography
    return M, (out_w, out_h)


def warp_to_birds_eye(frame, M, out_size):
    """
    Aplicira bird's-eye transformacijo na sliko.

    Args:
        frame:    vhodna slika (po undistortion)
        M:        transformacijska matrika (iz build_birds_eye_transform)
        out_size: (širina, višina) izhodne slike

    Returns:
        Transformirana slika bird's-eye pogleda.
    """
    return cv2.warpPerspective(frame, M, out_size,
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))


def draw_birds_eye_overlay(bev_frame, hand_landmarks_list, active_idx,
                           M, orig_w, orig_h, scale_px_per_mm,
                           wrist_trail, out_w, out_h):
    """
    Izriše skelet roke in sled zapestja na bird's-eye sliki.

    Args:
        bev_frame:          bird's-eye slika
        hand_landmarks_list: seznam zaznanih rok
        active_idx:         indeks aktivne roke
        M:                  transformacijska matrika (px → bev px)
        orig_w, orig_h:     dimenzije originalne slike
        scale_px_per_mm:    merilo za prikaz osi
        wrist_trail:        deque s preteklimi zapestnimi pozicijami (v orig px)
        out_w, out_h:       dimenzije bird's-eye slike
    """

    def transform_point(px, py):
        """Pretvori točko iz originalne slike v bird's-eye koordinate."""
        pt  = np.array([px, py, 1.0], dtype=np.float64)
        res = M @ pt
        if abs(res[2]) > 1e-10:
            return (int(res[0] / res[2]), int(res[1] / res[2]))
        return (0, 0)

    # Izriši sled zapestja (zelene pike)
    trail_list = list(wrist_trail)
    for k in range(1, len(trail_list)):
        p1 = transform_point(trail_list[k - 1][0], trail_list[k - 1][1])
        p2 = transform_point(trail_list[k][0],     trail_list[k][1])
        alpha = k / max(len(trail_list), 1)
        color = (0, int(180 * alpha), 0)
        cv2.line(bev_frame, p1, p2, color, 1)

    # Izriši roke
    for i, hand_landmarks in enumerate(hand_landmarks_list):
        is_active = (i == active_idx)
        line_color  = (255, 255, 255) if is_active else (100, 100, 100)
        point_color = (0, 0, 0)       if is_active else (60, 60, 60)
        thick       = 2               if is_active else 1

        for connection in HAND_CONNECTIONS:
            p1_lm = hand_landmarks[connection[0]]
            p2_lm = hand_landmarks[connection[1]]
            tp1 = transform_point(p1_lm.x * orig_w, p1_lm.y * orig_h)
            tp2 = transform_point(p2_lm.x * orig_w, p2_lm.y * orig_h)
            cv2.line(bev_frame, tp1, tp2, line_color, thick)

        for lm in hand_landmarks:
            tp = transform_point(lm.x * orig_w, lm.y * orig_h)
            cv2.circle(bev_frame, tp, 3 if is_active else 2, point_color, -1)

        # Oznaka A/M pri zapestju
        wrist_tp = transform_point(
            hand_landmarks[0].x * orig_w,
            hand_landmarks[0].y * orig_h
        )
        label       = "A" if is_active else "M"
        label_color = (0, 220, 0) if is_active else (100, 100, 100)
        cv2.putText(bev_frame, label,
                    (wrist_tp[0] + 5, wrist_tp[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2, cv2.LINE_AA)

    # Koordinatna os v sredini (za orientacijo)
    cx, cy   = out_w // 2, out_h // 2
    axis_len = int(30 * scale_px_per_mm)
    cv2.arrowedLine(bev_frame, (cx, cy), (cx + axis_len, cy), (0, 0, 200), 1, tipLength=0.2)
    cv2.arrowedLine(bev_frame, (cx, cy), (cx, cy - axis_len), (200, 0, 0), 1, tipLength=0.2)
    cv2.putText(bev_frame, "X", (cx + axis_len + 3, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)
    cv2.putText(bev_frame, "Y", (cx - 4, cy - axis_len - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 0), 1)

    # Legenda merila
    bar_mm  = 50
    bar_px  = int(bar_mm * scale_px_per_mm)
    bx, by  = 10, out_h - 20
    cv2.line(bev_frame, (bx, by), (bx + bar_px, by), (200, 200, 200), 2)
    cv2.putText(bev_frame, "{} mm".format(bar_mm),
                (bx, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Napis
    cv2.putText(bev_frame, "BIRD'S-EYE VIEW",
                (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 60), 1, cv2.LINE_AA)

    return bev_frame


def create_combined_frame(main_frame, bev_frame, bev_scale=0.35):
    """
    Vstavi pomanjšan bird's-eye view v spodnji desni kot glavnega okvirja.

    Args:
        main_frame: glavni obdelani okvir
        bev_frame:  bird's-eye okvir
        bev_scale:  faktor pomanjšanja bird's-eye vstavka (0.0–1.0)

    Returns:
        Kombinirani okvir z vstavljenim bird's-eye pogledom.
    """
    h_main, w_main = main_frame.shape[:2]
    bev_w = int(bev_frame.shape[1] * bev_scale)
    bev_h = int(bev_frame.shape[0] * bev_scale)
    bev_small = cv2.resize(bev_frame, (bev_w, bev_h))

    # Obroba vstavka
    cv2.rectangle(bev_small, (0, 0), (bev_w - 1, bev_h - 1), (180, 180, 60), 2)

    margin  = 8
    x_off   = w_main - bev_w - margin
    y_off   = h_main - bev_h - margin

    combined = main_frame.copy()
    # Prosojno ozadje pod vstavkom
    roi_bg = combined[y_off:y_off + bev_h, x_off:x_off + bev_w]
    combined[y_off:y_off + bev_h, x_off:x_off + bev_w] = cv2.addWeighted(
        roi_bg, 0.15, bev_small, 0.85, 0)

    return combined


# ==============================================================================
# OSTALE POMOŽNE FUNKCIJE
# ==============================================================================

def draw_roi(frame, roi_x1, roi_y1, roi_x2, roi_y2):
    overlay = frame.copy()
    cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 180), 1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    cv2.putText(frame, "ROI", (roi_x1 + 4, roi_y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 180), 1, cv2.LINE_AA)
    return frame


def draw_kinematic_hud(frame, path_mm, vel_mm, acc_mm, path_hist, vel_hist, acc_hist,
                       calibrated=False, locked=False):
    overlay_w = 170
    overlay_h = 155
    margin    = 12
    corner_x, corner_y = 12, 320
    graph_h   = 30
    num_points = len(path_hist)
    alpha     = 0.82

    unit_dist = "mm"   if calibrated else "px"
    unit_vel  = "mm/s" if calibrated else "px/s"
    unit_acc  = "mm/s2" if calibrated else "px/s2"

    overlay = frame.copy()
    cv2.rectangle(overlay, (corner_x, corner_y),
                  (corner_x + overlay_w, corner_y + overlay_h),
                  (245, 245, 245), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    font      = cv2.FONT_HERSHEY_TRIPLEX
    color_txt = (30, 30, 30)

    status_parts = []
    if calibrated:
        status_parts.append("[KAL]")
    if locked:
        status_parts.append("[ZAK]")
    if status_parts:
        status_color = (0, 130, 0) if calibrated else (120, 80, 0)
        cv2.putText(frame, " ".join(status_parts),
                    (corner_x + margin, corner_y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1, cv2.LINE_AA)

    cv2.putText(frame, "x: {:.1f} {}".format(path_mm, unit_dist),
                (corner_x + margin, corner_y + 109), font, 0.45, color_txt, 1, cv2.LINE_AA)
    cv2.putText(frame, "v: {:.1f} {}".format(vel_mm, unit_vel),
                (corner_x + margin, corner_y + 129), font, 0.45, color_txt, 1, cv2.LINE_AA)
    cv2.putText(frame, "a: {:.1f} {}".format(acc_mm, unit_acc),
                (corner_x + margin, corner_y + 149), font, 0.45, color_txt, 1, cv2.LINE_AA)

    graph_x0 = corner_x + margin
    graph_y0  = corner_y + 45

    def normalize(vals):
        arr = list(vals)
        mn, mx = np.min(arr), np.max(arr)
        return [(v - mn) / (mx - mn + 1e-5) for v in arr]

    if num_points > 1:
        path_norm = normalize(path_hist)
        vel_norm  = normalize(vel_hist)
        acc_norm  = normalize(acc_hist)
        for i in range(1, num_points):
            cv2.line(frame,
                     (graph_x0 + i - 1, graph_y0 - int(path_norm[i - 1] * graph_h)),
                     (graph_x0 + i,     graph_y0 - int(path_norm[i]     * graph_h)),
                     (30, 30, 30), 1)
            cv2.line(frame,
                     (graph_x0 + i - 1, graph_y0 + 18 - int(vel_norm[i - 1] * graph_h)),
                     (graph_x0 + i,     graph_y0 + 18 - int(vel_norm[i]     * graph_h)),
                     (80, 70, 200), 1)
            cv2.line(frame,
                     (graph_x0 + i - 1, graph_y0 + 45 - int(acc_norm[i - 1] * graph_h)),
                     (graph_x0 + i,     graph_y0 + 45 - int(acc_norm[i]     * graph_h)),
                     (150, 40, 60), 1)

    cv2.putText(frame, "x", (graph_x0 - 5, graph_y0 - 2),  font, 0.37, (30, 30, 30),  1, cv2.LINE_AA)
    cv2.putText(frame, "v", (graph_x0 - 5, graph_y0 + 18), font, 0.37, (80, 70, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "a", (graph_x0 - 5, graph_y0 + 36), font, 0.37, (150, 40, 60), 1, cv2.LINE_AA)
    return frame


def draw_hand_skeleton(frame, hand_landmarks, width, height, active=True, in_roi=True):
    if not in_roi:
        line_color   = (0, 140, 255); point_color = (0, 100, 200)
        line_thick   = 1;             point_radius = 2
        label        = "?";           label_color  = (0, 140, 255)
    elif active:
        line_color   = (255, 255, 255); point_color = (0, 0, 0)
        line_thick   = 2;               point_radius = 4
        label        = "A";             label_color  = (0, 200, 0)
    else:
        line_color   = (130, 130, 130); point_color = (80, 80, 80)
        line_thick   = 1;               point_radius = 3
        label        = "M";             label_color  = (100, 100, 100)

    for connection in HAND_CONNECTIONS:
        p1 = hand_landmarks[connection[0]]
        p2 = hand_landmarks[connection[1]]
        cv2.line(frame,
                 (int(p1.x * width), int(p1.y * height)),
                 (int(p2.x * width), int(p2.y * height)),
                 line_color, line_thick)
    for landmark in hand_landmarks:
        cv2.circle(frame,
                   (int(landmark.x * width), int(landmark.y * height)),
                   point_radius, point_color, -1)

    wrist = hand_landmarks[0]
    cv2.putText(frame, label,
                (int(wrist.x * width) + 8, int(wrist.y * height) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2, cv2.LINE_AA)


# ==============================================================================
# GLAVNA FUNKCIJA
# ==============================================================================

def process_video(input_path, output_path, show_roi=True, birds_eye=False,
                  bev_out_w=600, bev_out_h=600, bev_scale_px_per_mm=2.0):
    """
    Obdelava videa z detekcijo roke in kinematičnimi izračuni.

    Args:
        input_path:          pot do vhodnega videa
        output_path:         pot do izhodnega videa
        show_roi:            prikaži ROI pravokotnik
        birds_eye:           vstavi bird's-eye view vstavek v izhodni video
        bev_out_w/h:         velikost bird's-eye platna v pikslih
        bev_scale_px_per_mm: merilo bird's-eye pogleda (px/mm)
    """
    download_model()

    # Zaznaj kamero iz imena datoteke in naloži kalibracijo
    camera_name = detect_camera_from_filename(input_path)
    calibration = None
    calibrated  = False

    if camera_name:
        print("  Zaznana kamera: {} (iz imena datoteke)".format(camera_name))
        config = load_calibration_config(camera_name)
        if config is not None:
            calibration = CameraCalibration(config)
            calibrated  = True
            if calibration.pixels_per_mm:
                print("  Kalibracija aktivna: pixels_per_mm = {:.4f}".format(calibration.pixels_per_mm))
            else:
                print("  Kalibracija aktivna (homografija)")
        else:
            print("  [OPOZORILO] Kalibracijska konfiguracija za '{}' ni najdena.".format(camera_name))
            print("  Zaženite kalibracijo: python calibration/scripts/calibrate.py")
    else:
        print("  [INFO] Kamera ni prepoznana iz imena datoteke. Uporaba pikselskih enot.")

    # Birds eye view, opcijsko
    '''
    bev_M    = None
    bev_size = None
    if birds_eye:
        if calibrated and calibration.homography is not None:
            bev_M, bev_size = build_birds_eye_transform(
                calibration.homography,
                out_w=bev_out_w, out_h=bev_out_h,
                scale_px_per_mm=bev_scale_px_per_mm
            )
            print("  Bird's-eye view: VKLOPLJEN ({}×{} px, {:.1f} px/mm)".format(
                bev_out_w, bev_out_h, bev_scale_px_per_mm))
        else:
            print("  [OPOZORILO] Bird's-eye view zahteva kalibracijo s homografijo – preskočen.")
            birds_eye = False
    '''

    # MediaPipe detektor
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    hand_selector = ActiveHandSelector(history_len=20, lock_after=30)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Napaka: Ni mogoce odpreti videa {}".format(input_path))
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0
    dt = 1.0 / fps

    roi_fracs = CAMERA_ROI.get(camera_name, CAMERA_ROI[None])
    roi_x1 = int(width  * roi_fracs[0])
    roi_y1 = int(height * roi_fracs[1])
    roi_x2 = int(width  * roi_fracs[2])
    roi_y2 = int(height * roi_fracs[3])
    print("  ROI: ({}, {}) – ({}, {}) px".format(roi_x1, roi_y1, roi_x2, roi_y2))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Zacetek obdelave videa: {}".format(input_path))
    print("  Resolucija: {}x{}, FPS: {:.1f}".format(width, height, fps))
    print("  Enote: {}".format("metrične (mm, mm/s, mm/s²)" if calibrated else "pikselske (px, px/s, px/s²)"))

    frame_count = 0
    prev_pos    = None
    prev_vel_px = 0.0
    path_sum_px = 0.0

    times    = []
    paths    = []
    vels     = []
    accs     = []
    paths_px = []
    hist_len  = 100
    path_hist = deque(maxlen=hist_len)
    vel_hist  = deque(maxlen=hist_len)
    acc_hist  = deque(maxlen=hist_len)

    # Sled zapestja za bird's-eye (v originalnih px koordinatah)
    wrist_trail = deque(maxlen=80)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if calibrated:
            frame_undistorted = calibration.undistort_frame(frame)
        else:
            frame_undistorted = frame

        image_rgb = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = detector.detect(mp_image)

        display_frame     = frame_undistorted.copy()
        active_hand_found = False
        active_global_idx = 0

        if detection_result.hand_landmarks:
            all_wrist_positions = []
            all_in_roi          = []

            for hand_landmarks in detection_result.hand_landmarks:
                wrist  = hand_landmarks[0]
                pos_px = (wrist.x * width, wrist.y * height)
                if calibrated:
                    pos_px = calibration.undistort_point(pos_px)
                in_roi = (roi_x1 <= pos_px[0] <= roi_x2 and
                          roi_y1 <= pos_px[1] <= roi_y2)
                all_wrist_positions.append(pos_px)
                all_in_roi.append(in_roi)

            roi_positions = []
            roi_hand_idxs = []
            for i, (pos, ir) in enumerate(zip(all_wrist_positions, all_in_roi)):
                if ir:
                    roi_positions.append(pos)
                    roi_hand_idxs.append(i)

            if not roi_positions:
                roi_positions = all_wrist_positions
                roi_hand_idxs = list(range(len(detection_result.hand_landmarks)))

            selected_in_roi   = hand_selector.select(roi_positions)
            if selected_in_roi is None:
                selected_in_roi = 0
            active_global_idx = roi_hand_idxs[selected_in_roi]
            active_hand_found = True

            curr_pos_px = all_wrist_positions[active_global_idx]
            wrist_trail.append(curr_pos_px)  # za bird's-eye sled

            if prev_pos is not None:
                dx   = curr_pos_px[0] - prev_pos[0]
                dy   = curr_pos_px[1] - prev_pos[1]
                d_px = (dx ** 2 + dy ** 2) ** 0.5
            else:
                d_px = 0.0

            path_sum_px += d_px
            curr_vel_px  = d_px / dt if prev_pos is not None else 0.0
            acc_px       = (curr_vel_px - prev_vel_px) / dt if prev_pos is not None else 0.0

            if calibrated:
                path_metric = calibration.pixel_to_mm(path_sum_px)
                vel_metric  = calibration.pixel_velocity_to_mm_s(curr_vel_px)
                acc_metric  = calibration.pixel_acc_to_mm_s2(acc_px)
            else:
                path_metric = path_sum_px
                vel_metric  = curr_vel_px
                acc_metric  = acc_px

            t = frame_count * dt
            times.append(t)
            paths.append(path_metric)
            vels.append(vel_metric)
            accs.append(acc_metric)
            paths_px.append(path_sum_px)
            path_hist.append(path_metric)
            vel_hist.append(vel_metric)
            acc_hist.append(acc_metric)

            prev_pos    = curr_pos_px
            prev_vel_px = curr_vel_px

            for i, hand_landmarks in enumerate(detection_result.hand_landmarks):
                draw_hand_skeleton(display_frame, hand_landmarks, width, height,
                                   active=(i == active_global_idx),
                                   in_roi=all_in_roi[i])

        if not active_hand_found:
            t = frame_count * dt
            times.append(t)
            paths.append(paths[-1] if paths else 0.0)
            vels.append((vels[-1] * 0.85) if vels else 0.0)
            accs.append((accs[-1] * 0.85) if accs else 0.0)
            paths_px.append(path_sum_px)
            path_hist.append(path_hist[-1] if path_hist else 0.0)
            vel_hist.append((vel_hist[-1] * 0.85) if vel_hist else 0.0)
            acc_hist.append(0.0)

        if show_roi:
            display_frame = draw_roi(display_frame, roi_x1, roi_y1, roi_x2, roi_y2)

        display_frame = draw_kinematic_hud(
            display_frame,
            paths[-1], vels[-1], accs[-1],
            path_hist, vel_hist, acc_hist,
            calibrated=calibrated,
            locked=hand_selector.locked,
        )

        # ---- BIRD'S-EYE VIEW vstavek ----
        if birds_eye and bev_M is not None:
            bev_frame = warp_to_birds_eye(frame_undistorted, bev_M, bev_size)
            bev_frame = draw_birds_eye_overlay(
                bev_frame,
                detection_result.hand_landmarks if detection_result.hand_landmarks else [],
                active_global_idx,
                bev_M, width, height,
                bev_scale_px_per_mm,
                wrist_trail,
                bev_out_w, bev_out_h,
            )
            display_frame = create_combined_frame(display_frame, bev_frame, bev_scale=0.35)
        # ---- konec bird's-eye ----

        out.write(display_frame)
        frame_count += 1

    # Zapis dnevnika
    log_path = os.path.join(LOG_DIR, os.path.splitext(os.path.basename(output_path))[0] + ".log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Obdelan video: {}\n".format(input_path))
        f.write("Izhodna datoteka: {}\n".format(output_path))
        f.write("Število okvirjev: {}\n".format(frame_count))
        f.write("Kamera: {}\n".format(camera_name if camera_name else "neznana"))
        f.write("Kalibracija: {}\n".format("DA" if calibrated else "NE"))
        if calibrated and calibration.pixels_per_mm:
            f.write("Pixels per mm: {:.4f}\n".format(calibration.pixels_per_mm))
        f.write("Selekcija aktivne roke: DA (ROI filter + zaklep)\n")
        f.write("Roka zaklenjena: {}\n".format("DA" if hand_selector.locked else "NE"))
        f.write("ROI: ({},{})-({},{}) px\n".format(roi_x1, roi_y1, roi_x2, roi_y2))
        f.write("Confidence pragovi: detection=0.3, presence=0.3, tracking=0.3\n")
        f.write("Bird's-eye view: {}\n".format("DA" if birds_eye else "NE"))

    # Zapis kinematike
    kin_path = os.path.join(LOG_DIR, os.path.splitext(os.path.basename(output_path))[0] + "_kinematika.csv")
    with open(kin_path, "w", encoding="utf-8") as f:
        if calibrated:
            f.write("cas[s];pot[mm];hitrost[mm/s];pospesek[mm/s2];pot_px[px]\n")
            for t, s, v, a, s_px in zip(times, paths, vels, accs, paths_px):
                f.write("{:.3f};{:.2f};{:.2f};{:.2f};{:.2f}\n".format(t, s, v, a, s_px))
        else:
            f.write("cas[s];pot[px];hitrost[px/s];pospesek[px/s2]\n")
            for t, s, v, a in zip(times, paths, vels, accs):
                f.write("{:.3f};{:.2f};{:.2f};{:.2f}\n".format(t, s, v, a))

    cap.release()
    out.release()
    print("Konec obdelave. Prebranih okvirjev: {}".format(frame_count))
    print("Video shranjen v: {}".format(output_path))
    print("Kinematika shranjena v: {}".format(kin_path))


def get_mp4_files_recursively(data_dir):
    mp4_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith('.mp4'):
                mp4_files.append(os.path.join(root, f))
    return mp4_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Obdelava .mp4 video datotek za 9HPT z MediaPipe hand tracking in kalibracijo kamer."
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Ime ali pot do vhodne .mp4 datoteke ali 'all' za obdelavo vseh *.mp4 v data/")
    parser.add_argument("--output", "-o", required=False,
                        help="Ime ali pot za izhodno .mp4 datoteko (samo pri obdelavi ene datoteke)")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Onemogoči uporabo kalibracije (vsi izračuni v pikslih)")
    parser.add_argument("--no-roi", action="store_true",
                        help="Onemogoči prikaz ROI pravokotnika na videu")
    parser.add_argument("--roi", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Ročno nastavi ROI kot deleži slike npr.: --roi 0.1 0.2 0.75 0.95")
    parser.add_argument("--lock-after", type=int, default=30,
                        help="Po koliko okvirjih zakleni aktivno roko (privzeto: 30)")
    parser.add_argument("--birds-eye", action="store_true",
                        help="Vstavi bird's-eye view vstavek v izhodni video (zahteva homografijo)")
    parser.add_argument("--bev-size", nargs=2, type=int, default=[600, 600],
                        metavar=("SIRINA", "VISINA"),
                        help="Velikost bird's-eye platna v pikslih (privzeto: 600 600)")
    parser.add_argument("--bev-scale", type=float, default=2.0,
                        help="Merilo bird's-eye pogleda v px/mm (privzeto: 2.0)")

    args = parser.parse_args()

    if args.roi:
        for cam in list(CAMERA_ROI.keys()):
            CAMERA_ROI[cam] = tuple(args.roi)
        print("ROI ročno nastavljen: {}".format(args.roi))

    print("=" * 60)
    print("9HPT Analiza - Robotski vid")
    print("=" * 60)

    available_calibrations = []
    for cam_name in ["left", "mid", "right"]:
        config_path   = os.path.join(CALIBRATION_CONF_DIR, "{}_calibration.json".format(cam_name))
        fallback_path = os.path.join("calibration", "calibration.json")
        if os.path.exists(config_path):
            available_calibrations.append("{} (lastna)".format(cam_name))
        elif os.path.exists(fallback_path):
            available_calibrations.append("{} (fallback)".format(cam_name))

    if available_calibrations and not args.no_calibration:
        print("Razpoložljive kalibracije: {}".format(", ".join(available_calibrations)))
    elif args.no_calibration:
        print("Kalibracija onemogočena (--no-calibration)")
    else:
        print("[INFO] Nobena kalibracija ni na voljo. Uporaba pikselskih enot.")
        print("  Za kalibracijo zaženite: python calibration/scripts/calibrate.py")

    print()

    show_roi   = not args.no_roi
    birds_eye  = args.birds_eye
    bev_out_w, bev_out_h = args.bev_size
    bev_scale  = args.bev_scale

    if args.input.lower() == "all":
        files = get_mp4_files_recursively(DATA_DIR)
        if not files:
            print("V mapi '{}' ni nobene mp4 datoteke za obdelavo.".format(DATA_DIR))
        for in_path in files:
            in_basename = os.path.splitext(os.path.basename(in_path))[0]
            out_path    = os.path.join(OUTPUT_DIR, in_basename + "_obdelan.mp4")
            process_video(in_path, out_path, show_roi=show_roi,
                          birds_eye=birds_eye,
                          bev_out_w=bev_out_w, bev_out_h=bev_out_h,
                          bev_scale_px_per_mm=bev_scale)
    else:
        if os.path.isfile(args.input):
            in_path = args.input
        else:
            in_path = os.path.join(DATA_DIR, args.input)
        if not os.path.isfile(in_path):
            print("Napaka: Ne najdem datoteke '{}'!".format(args.input))
            exit(1)
        if args.output:
            out_path = os.path.join(OUTPUT_DIR, args.output)
        else:
            in_basename = os.path.splitext(os.path.basename(in_path))[0]
            out_path    = os.path.join(OUTPUT_DIR, in_basename + "_obdelan.mp4")
        process_video(in_path, out_path, show_roi=show_roi,
                      birds_eye=birds_eye,
                      bev_out_w=bev_out_w, bev_out_h=bev_out_h,
                      bev_scale_px_per_mm=bev_scale)