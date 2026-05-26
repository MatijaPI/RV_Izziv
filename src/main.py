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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    print("[OPOZORILO] matplotlib ni nameščen – grafi ne bodo shranjeni.")
    print("  Namestite z: pip install matplotlib")

try:
    from scipy.signal import savgol_filter
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

# ==============================================================================
# KONSTANTE IN KONFIGURACIJA
# ==============================================================================

DATA_DIR             = "data"
OUTPUT_DIR           = "output/videos"
GRAPH_DIR            = "output/graphs"
LOG_DIR              = "output/logs"
MODEL_DIR            = "models"
MODEL_PATH           = os.path.join(MODEL_DIR, "hand_landmarker.task")
CALIBRATION_CONF_DIR = "calibration/conf"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR,  exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

IDX_WRIST     = 0
IDX_THUMB_TIP = 4
IDX_INDEX_TIP = 8

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

CAMERA_MAP = {"camP_0":"left","camP_1":"mid","camP_2":"right"}

CAMERA_ROI = {
    "left":  (0.25, 0.05, 0.70, 0.85),
    "mid":   (0.25, 0.10, 0.70, 0.80),
    "right": (0.20, 0.15, 0.70, 0.80),
    None:    (1.0,  1.0,  1.0,  1.0),
}

PINCH_THRESHOLD_PX = 40.0
PINCH_THRESHOLD_MM = 20.0


# ==============================================================================
# POMOŽNE FUNKCIJE
# ==============================================================================

def _dist2d(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5


# ==============================================================================
# GAUSSOV GLAJILNIK POLOŽAJEV (kavzalni, brez zakasnitve)
# ==============================================================================

class KinematicSmoother:
    """
    Kavzalno Gaussovo glajenje položajev v realnem času.

    Vzdržuje okno zadnjih window_size položajev in vrne tehtano povprečje
    z Gaussovimi utežmi (center okna = najnovejši vzorec).

    Parametri:
        sigma       – standardna deviacija Gaussove porazdelitve v okvirjih
                      (večja vrednost = bolj gladko, a manj odzivno)
        window_size – dolžina okna (privzeto: int(4*sigma)+1, vsaj 3)
        method      – 'gauss' (privzeto) ali 'savgol' (zahteva scipy)
    """

    def __init__(self, sigma=1.5, method="gauss"):
        self.sigma  = max(sigma, 0.1)
        self.method = method if (method == "savgol" and SCIPY_OK) else "gauss"

        # Dolžina okna – liho število, vsaj 3
        ws = max(3, int(4 * self.sigma) + 1)
        if ws % 2 == 0:
            ws += 1
        self.window_size = ws

        # Gaussove uteži (indeks 0 = najstarejši, -1 = najnovejši)
        idxs      = np.arange(self.window_size)
        center    = self.window_size - 1          # uteži naraščajo proti koncu
        raw_w     = np.exp(-0.5 * ((idxs - center) / self.sigma) ** 2)
        self.weights = raw_w / raw_w.sum()        # normalizacija

        # Krožni medpomnilnik za x in y ločeno
        self._buf_x = deque(maxlen=self.window_size)
        self._buf_y = deque(maxlen=self.window_size)

    def update(self, pos):
        """
        Posodobi okno z novo točko in vrne zglajen položaj.
        pos: (x, y) v pikslih
        """
        self._buf_x.append(pos[0])
        self._buf_y.append(pos[1])

        n = len(self._buf_x)
        if n < 2:
            return pos  # premalo vzorcev – vrni nespremenjeno

        # Prilagodi uteži na dejansko dolžino okna (na začetku videa)
        if n < self.window_size:
            w = self.weights[-n:]
            w = w / w.sum()
        else:
            w = self.weights

        sx = float(np.dot(w, list(self._buf_x)))
        sy = float(np.dot(w, list(self._buf_y)))
        return (sx, sy)

    def smooth_series_postprocess(self, series):
        """
        Naknadna (post-process) obdelava celotne serije z Gaussian ali Savitzky-Golay.
        Uporablja se za grafe po koncu videa – ne vpliva na vrednosti v videu.
        Vrne numpy array enake dolžine.
        """
        arr = np.array(series, dtype=np.float64)
        if len(arr) < 5:
            return arr

        if self.method == "savgol" and SCIPY_OK:
            wl = min(self.window_size, len(arr))
            if wl % 2 == 0:
                wl -= 1
            wl = max(wl, 5)
            return savgol_filter(arr, window_length=wl, polyorder=3)
        else:
            # Gaussov filter (scipy.ndimage.gaussian_filter1d ni potreben –
            # implementiramo z scipy-neodvisno konvolucijo)
            k  = int(3 * self.sigma)
            xs = np.arange(-k, k + 1)
            kernel = np.exp(-0.5 * (xs / self.sigma) ** 2)
            kernel /= kernel.sum()
            return np.convolve(arr, kernel, mode="same")


# ==============================================================================
# KINEMATIČNI SLEDILNIK
# ==============================================================================

class KinematicTracker:
    """
    Sledilnik kinematičnih parametrov za eno točko.
    Vgrajeno Gaussovo glajenje položajev (kavzalno, brez zakasnitve).
    """

    def __init__(self, name, decay=0.85, smoother=None):
        self.name     = name
        self.decay    = decay
        self.smoother = smoother   # KinematicSmoother ali None

        self.prev_pos  = None
        self.prev_vel  = 0.0
        self.path_sum  = 0.0

        self.times = []; self.paths = []; self.vels = []; self.accs = []
        self.path_hist = deque(maxlen=100)
        self.vel_hist  = deque(maxlen=100)
        self.acc_hist  = deque(maxlen=100)

        # Surovi (neglajeni) položaji – za post-process grafe
        self._raw_positions = []

    def update(self, pos_px, dt, calibration=None):
        """
        pos_px: surova pozicija v pikslih (po undistortion)
        Vrne (path_metric, vel_metric, acc_metric) – vrednosti na podlagi
        glajene pozicije (če je smoother aktiven).
        """
        self._raw_positions.append(pos_px)

        # Glajenje položaja
        if self.smoother is not None:
            pos = self.smoother.update(pos_px)
        else:
            pos = pos_px

        d_px = _dist2d(pos, self.prev_pos) if self.prev_pos is not None else 0.0
        self.path_sum += d_px

        curr_vel_px = d_px / dt if self.prev_pos is not None else 0.0
        acc_px      = (curr_vel_px - self.prev_vel) / dt if self.prev_pos is not None else 0.0

        if calibration is not None:
            path_m = calibration.pixel_to_mm(self.path_sum)
            vel_m  = calibration.pixel_velocity_to_mm_s(curr_vel_px)
            acc_m  = calibration.pixel_acc_to_mm_s2(acc_px)
        else:
            path_m, vel_m, acc_m = self.path_sum, curr_vel_px, acc_px

        self.prev_pos = pos
        self.prev_vel = curr_vel_px

        self.paths.append(path_m); self.vels.append(vel_m); self.accs.append(acc_m)
        self.path_hist.append(path_m); self.vel_hist.append(vel_m); self.acc_hist.append(acc_m)
        return path_m, vel_m, acc_m

    def missing(self):
        lp = self.paths[-1] if self.paths else 0.0
        lv = self.vels[-1]  if self.vels  else 0.0
        la = self.accs[-1]  if self.accs  else 0.0
        # Ohrani položaj v smootherju (ne dodaj nove točke – smoother ne sme "pozabiti")
        self.paths.append(lp); self.vels.append(lv*self.decay); self.accs.append(la*self.decay)
        self.path_hist.append(lp); self.vel_hist.append(lv*self.decay); self.acc_hist.append(0.0)

    def set_time(self, t):
        self.times.append(t)

    def smoothed_vels(self):
        """Post-process glajene hitrosti za grafe (ne vpliva na vrednosti v videu)."""
        if self.smoother is not None:
            return self.smoother.smooth_series_postprocess(self.vels)
        return np.array(self.vels)

    def smoothed_accs(self):
        """Post-process glajeni pospeški za grafe."""
        if self.smoother is not None:
            return self.smoother.smooth_series_postprocess(self.accs)
        return np.array(self.accs)


# ==============================================================================
# DETEKTOR PRIJEMA
# ==============================================================================

class PinchDetector:
    def __init__(self, threshold_px=PINCH_THRESHOLD_PX,
                 threshold_mm=PINCH_THRESHOLD_MM, min_frames=3):
        self.threshold_px  = threshold_px
        self.threshold_mm  = threshold_mm
        self.min_frames    = min_frames
        self.state         = "OPEN"
        self.candidate_cnt = 0
        self.events        = []
        self.distances     = []
        self.is_grasping   = False

    def update(self, thumb_px, index_px, t, frame_idx, calibration=None):
        dist_px = _dist2d(thumb_px, index_px)
        if calibration is not None:
            dist_m = calibration.pixel_to_mm(dist_px); thr = self.threshold_mm
        else:
            dist_m = dist_px; thr = self.threshold_px
        self.distances.append(dist_m)
        new_state = "CLOSED" if dist_m < thr else "OPEN"
        if new_state != self.state:
            self.candidate_cnt += 1
            if self.candidate_cnt >= self.min_frames:
                self.state = new_state; self.candidate_cnt = 0
                self.is_grasping = (self.state == "CLOSED")
                self.events.append({"t":t,"type":"grasp" if self.is_grasping else "release",
                                    "frame":frame_idx,"dist":dist_m})
                return True
        else:
            self.candidate_cnt = 0
        return False

    def missing(self):
        self.distances.append(self.distances[-1] if self.distances else 0.0)


# ==============================================================================
# SELEKTOR AKTIVNE ROKE
# ==============================================================================

class ActiveHandSelector:
    def __init__(self, history_len=20, lock_after=30):
        self.history_len   = history_len
        self.lock_after    = lock_after
        self.slot_positions = [deque(maxlen=history_len), deque(maxlen=history_len)]
        self.active_slot   = None  # type: Optional[int]
        self.locked        = False
        self.frames_seen   = 0

    def _activity_score(self, si):
        hist = list(self.slot_positions[si])
        return sum(_dist2d(hist[j],hist[j-1]) for j in range(1,len(hist))) if len(hist)>=2 else 0.0

    def _assign_to_slots(self, positions):
        n = len(positions)
        if n == 0: return []
        if n == 1:
            return [(0, self.active_slot if self.active_slot is not None else 0)]
        last = [self.slot_positions[s][-1] if self.slot_positions[s] else None for s in range(2)]
        if last[0] is None and last[1] is None: return [(0,0),(1,1)]
        if last[0] is None:
            return [(0,1),(1,0)] if _dist2d(positions[0],last[1])<_dist2d(positions[1],last[1]) else [(0,0),(1,1)]
        if last[1] is None:
            return [(0,0),(1,1)] if _dist2d(positions[0],last[0])<=_dist2d(positions[1],last[0]) else [(0,1),(1,0)]
        cs = _dist2d(positions[0],last[0])+_dist2d(positions[1],last[1])
        cc = _dist2d(positions[0],last[1])+_dist2d(positions[1],last[0])
        return [(0,0),(1,1)] if cs<=cc else [(0,1),(1,0)]

    def select(self, positions):
        # type: (list) -> Optional[int]
        if not positions: return None
        self.frames_seen += 1
        assignments = self._assign_to_slots(positions)
        for pi, si in assignments:
            self.slot_positions[si].append(positions[pi])
        if not self.locked:
            scores = [self._activity_score(0), self._activity_score(1)]
            if self.active_slot is None:
                self.active_slot = 0 if scores[0]>=scores[1] else 1
            else:
                inactive = 1-self.active_slot
                if scores[inactive] > scores[self.active_slot]*1.5:
                    self.active_slot = inactive
            if self.frames_seen >= self.lock_after:
                self.locked = True
                print("  [ActiveHandSelector] Aktivna roka {} določena po {} okvirjih.".format(
                    self.active_slot, self.frames_seen))
        for pi, si in assignments:
            if si == self.active_slot: return pi
        lk = self.slot_positions[self.active_slot]
        if lk and positions:
            ref = lk[-1]
            return min(range(len(positions)), key=lambda i: _dist2d(positions[i],ref))
        return 0


# ==============================================================================
# KALIBRACIJA
# ==============================================================================

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Prenasam MediaPipe model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", MODEL_PATH)
        print("Prenos uspesen.")


def detect_camera_from_filename(filename):
    m = re.search(r'camP_(\d)', os.path.basename(filename))
    return CAMERA_MAP.get("camP_{}".format(m.group(1)), None) if m else None


def load_calibration_config(camera_name):
    cp = os.path.join(CALIBRATION_CONF_DIR, "{}_calibration.json".format(camera_name))
    if os.path.exists(cp):
        with open(cp,"r",encoding="utf-8") as f: config = json.load(f)
        print("  Naložena kalibracija iz: {}".format(cp)); return config
    for fb in [os.path.join("calibration","calibration.json"),"calibration.json"]:
        if os.path.exists(fb):
            with open(fb,"r",encoding="utf-8") as f: ac = json.load(f)
            if camera_name in ac:
                ref = ac[camera_name]
                config = {"camera_name":camera_name,"camera_matrix":ref["cameraMatrix"],
                          "dist_coeffs":([ref["distortionCoeffs"][0]]
                                         if isinstance(ref["distortionCoeffs"][0],list)
                                         else [ref["distortionCoeffs"]]),
                          "homography":ref.get("homography",None),"pixels_per_mm":None,
                          "source":"validation_fallback"}
                if config["homography"] is not None:
                    H = np.array(config["homography"])
                    try:
                        Hi = np.linalg.inv(H)
                        config["pixels_per_mm"] = (np.linalg.norm(Hi[:2,0])+np.linalg.norm(Hi[:2,1]))/2.0
                    except np.linalg.LinAlgError: pass
                print("  Naložena kalibracija (fallback) iz: {}".format(fb)); return config
    return None


class CameraCalibration:
    def __init__(self, config):
        self.camera_matrix     = np.array(config["camera_matrix"],  dtype=np.float64)
        self.dist_coeffs       = np.array(config["dist_coeffs"],    dtype=np.float64)
        self.homography        = (np.array(config["homography"],dtype=np.float64)
                                  if config.get("homography") else None)
        self.pixels_per_mm     = config.get("pixels_per_mm",None)
        self.new_camera_matrix = (np.array(config["new_camera_matrix"],dtype=np.float64)
                                  if config.get("new_camera_matrix") else None)
        self.camera_name = config.get("camera_name","unknown")

    def undistort_frame(self, frame):
        if self.new_camera_matrix is not None:
            return cv2.undistort(frame,self.camera_matrix,self.dist_coeffs,None,self.new_camera_matrix)
        return cv2.undistort(frame,self.camera_matrix,self.dist_coeffs)

    def undistort_point(self, point):
        u = cv2.undistortPoints(np.array([[[point[0],point[1]]]],dtype=np.float64),
                                self.camera_matrix,self.dist_coeffs,P=self.camera_matrix)
        return (u[0][0][0],u[0][0][1])

    def pixel_to_mm(self, d):
        if self.pixels_per_mm and self.pixels_per_mm>0: return d/self.pixels_per_mm
        return d/(self.camera_matrix[0,0]/500.0)
    def pixel_velocity_to_mm_s(self,v): return self.pixel_to_mm(v)
    def pixel_acc_to_mm_s2(self,a):     return self.pixel_to_mm(a)


# ==============================================================================
# BIRD'S-EYE VIEW (zakomentirano)
# ==============================================================================

def build_birds_eye_transform(homography, out_w=600, out_h=600, scale_px_per_mm=2.0):
    S = np.array([[scale_px_per_mm,0,out_w/2.0],[0,scale_px_per_mm,out_h/2.0],[0,0,1.0]],dtype=np.float64)
    return S@homography,(out_w,out_h)

def warp_to_birds_eye(frame,M,out_size):
    return cv2.warpPerspective(frame,M,out_size,flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))

def draw_birds_eye_overlay(bev,hll,ai,M,ow,oh,spm,wt,bw,bh):
    def tp(px,py):
        pt=np.array([px,py,1.0],dtype=np.float64); r=M@pt
        return (int(r[0]/r[2]),int(r[1]/r[2])) if abs(r[2])>1e-10 else (0,0)
    tl=list(wt)
    for k in range(1,len(tl)):
        cv2.line(bev,tp(*tl[k-1]),tp(*tl[k]),(0,int(180*k/max(len(tl),1)),0),1)
    for i,hl in enumerate(hll):
        lc=(255,255,255) if i==ai else (100,100,100)
        for c in HAND_CONNECTIONS:
            cv2.line(bev,tp(hl[c[0]].x*ow,hl[c[0]].y*oh),tp(hl[c[1]].x*ow,hl[c[1]].y*oh),lc,2 if i==ai else 1)
        for lm in hl: cv2.circle(bev,tp(lm.x*ow,lm.y*oh),3 if i==ai else 2,(0,0,0),-1)
    cx,cy=bw//2,bh//2; al=int(30*spm)
    cv2.arrowedLine(bev,(cx,cy),(cx+al,cy),(0,0,200),1,tipLength=0.2)
    cv2.arrowedLine(bev,(cx,cy),(cx,cy-al),(200,0,0),1,tipLength=0.2)
    cv2.putText(bev,"BIRD'S-EYE",(5,14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(180,180,60),1,cv2.LINE_AA)
    return bev

def create_combined_frame(mf,bf,bev_scale=0.35):
    h,w=mf.shape[:2]; bw=int(bf.shape[1]*bev_scale); bh=int(bf.shape[0]*bev_scale)
    bs=cv2.resize(bf,(bw,bh)); cv2.rectangle(bs,(0,0),(bw-1,bh-1),(180,180,60),2)
    x0,y0=w-bw-8,h-bh-8; comb=mf.copy()
    comb[y0:y0+bh,x0:x0+bw]=cv2.addWeighted(comb[y0:y0+bh,x0:x0+bw],0.15,bs,0.85,0)
    return comb


# ==============================================================================
# RISANJE – ROI
# ==============================================================================

def draw_roi(frame, x1, y1, x2, y2):
    ov = frame.copy()
    cv2.rectangle(ov,(x1,y1),(x2,y2),(0,255,180),1)
    frame = cv2.addWeighted(ov,0.5,frame,0.5,0)
    cv2.putText(frame,"ROI",(x1+4,y1+14),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,180),1,cv2.LINE_AA)
    return frame


# ==============================================================================
# HUD – spodnji pas
# ==============================================================================

def draw_kinematic_hud(frame,
                       hand_tracker, thumb_tracker, index_tracker,
                       calibrated=False,
                       is_grasping=False,
                       pinch_dist=None,
                       smooth_sigma=None):
    """
    Horizontalni HUD pas na dnu slike.
    Trije bloki (Zapestje | Palec | Kazalec), vsak z x / v / a vrednostmi.
    """
    h_frame, w_frame = frame.shape[:2]
    pad_h   = 80
    pad_top = 6
    bar_y   = h_frame - pad_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w_frame, h_frame), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0)
    cv2.line(frame, (0, bar_y), (w_frame, bar_y), (80, 80, 80), 1)

    ud = "mm"    if calibrated else "px"
    uv = "mm/s"  if calibrated else "px/s"
    ua = "mm/s²" if calibrated else "px/s²"

    hx = hand_tracker.paths[-1]  if hand_tracker.paths  else 0.0
    hv = hand_tracker.vels[-1]   if hand_tracker.vels   else 0.0
    ha = hand_tracker.accs[-1]   if hand_tracker.accs   else 0.0
    tx = thumb_tracker.paths[-1] if thumb_tracker.paths else 0.0
    tv = thumb_tracker.vels[-1]  if thumb_tracker.vels  else 0.0
    ta = thumb_tracker.accs[-1]  if thumb_tracker.accs  else 0.0
    ix = index_tracker.paths[-1] if index_tracker.paths else 0.0
    iv = index_tracker.vels[-1]  if index_tracker.vels  else 0.0
    ia = index_tracker.accs[-1]  if index_tracker.accs  else 0.0

    block_w = w_frame // 3
    blocks = [
        ("ZAPESTJE", hx, hv, ha, (220, 220, 220)),
        ("PALEC",    tx, tv, ta, (60,  160, 255)),
        ("KAZALEC",  ix, iv, ia, (220, 120,  30)),
    ]

    font_label = cv2.FONT_HERSHEY_SIMPLEX
    font_val   = cv2.FONT_HERSHEY_DUPLEX

    for col, (label, xv, vv, av, color) in enumerate(blocks):
        bx = col * block_w
        if col > 0:
            cv2.line(frame, (bx, bar_y+4), (bx, h_frame-4), (70, 70, 70), 1)
        cx = bx + block_w // 2

        label_size = cv2.getTextSize(label, font_label, 0.42, 1)[0]
        cv2.putText(frame, label,
                    (cx - label_size[0]//2, bar_y + pad_top + 14),
                    font_label, 0.42, color, 1, cv2.LINE_AA)

        line1 = "x:{:.0f} {}".format(xv, ud)
        line2 = "v:{:.0f} {}   a:{:.0f} {}".format(vv, uv, av, ua)

        l1_size = cv2.getTextSize(line1, font_val, 0.48, 1)[0]
        l2_size = cv2.getTextSize(line2, font_label, 0.37, 1)[0]

        cv2.putText(frame, line1,
                    (cx - l1_size[0]//2, bar_y + pad_top + 38),
                    font_val, 0.48, color, 1, cv2.LINE_AA)
        cv2.putText(frame, line2,
                    (cx - l2_size[0]//2, bar_y + pad_top + 58),
                    font_label, 0.37, (180, 180, 180), 1, cv2.LINE_AA)

    # Status vrstice – levo
    status_y = bar_y + pad_top + 14
    status_x = 8
    if calibrated:
        cv2.putText(frame, "KALIB.",
                    (status_x, status_y),
                    font_label, 0.35, (0, 200, 80), 1, cv2.LINE_AA)
        status_y += 14
        
    '''    
    if smooth_sigma is not None:
        cv2.putText(frame, "GLAJEN(s={:.1f})".format(smooth_sigma),
                    (status_x, status_y),
                    font_label, 0.30, (180, 180, 60), 1, cv2.LINE_AA)
        status_y += 12
    '''

    # Pinch indikator – desno
    if pinch_dist is not None:
        p_color = (0, 210, 80) if is_grasping else (200, 0, 0)
        p_label = "PRIJEM" if is_grasping else "ODPRTO"
        cv2.putText(frame, p_label,
                    (8, bar_y + pad_top - 20),
                    font_label, 0.35, p_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "pinch: {:.0f}{}".format(pinch_dist, "mm" if calibrated else "px"),
                    (8, bar_y + pad_top - 10),
                    font_label, 0.33, (140, 140, 140), 1, cv2.LINE_AA)

    return frame


# ==============================================================================
# RISANJE – skelet roke
# ==============================================================================

def draw_hand_skeleton(frame, hand_landmarks, width, height,
                       active=True, in_roi=True, is_grasping=False):
    if not in_roi:
        lc,pc,lt,pr = (0,140,255),(0,100,200),1,2; lbl,lc2 = "?",(0,140,255)
    elif active:
        lc,pc,lt,pr = (255,255,255),(0,0,0),2,4;   lbl,lc2 = "A",(0,200,0)
    else:
        lc,pc,lt,pr = (130,130,130),(80,80,80),1,3; lbl,lc2 = "M",(100,100,100)

    for conn in HAND_CONNECTIONS:
        p1=hand_landmarks[conn[0]]; p2=hand_landmarks[conn[1]]
        cv2.line(frame,(int(p1.x*width),int(p1.y*height)),
                       (int(p2.x*width),int(p2.y*height)),lc,lt)
    for lm in hand_landmarks:
        cv2.circle(frame,(int(lm.x*width),int(lm.y*height)),pr,pc,-1)

    wrist = hand_landmarks[0]
    cv2.putText(frame,lbl,(int(wrist.x*width)+8,int(wrist.y*height)-8),
                cv2.FONT_HERSHEY_SIMPLEX,0.55,lc2,2,cv2.LINE_AA)

    if active and in_roi:
        tp  = hand_landmarks[IDX_THUMB_TIP]
        tpx = (int(tp.x*width), int(tp.y*height))
        cv2.circle(frame, tpx, 7, (60,160,255), 2)
        cv2.putText(frame,"P",(tpx[0]+6,tpx[1]-6),cv2.FONT_HERSHEY_SIMPLEX,0.4,(60,160,255),1,cv2.LINE_AA)

        ip  = hand_landmarks[IDX_INDEX_TIP]
        ipx = (int(ip.x*width), int(ip.y*height))
        cv2.circle(frame, ipx, 7, (220,120,30), 2)
        cv2.putText(frame,"K",(ipx[0]+6,ipx[1]-6),cv2.FONT_HERSHEY_SIMPLEX,0.4,(220,120,30),1,cv2.LINE_AA)

        pinch_color = (0,210,80) if is_grasping else (120,120,120)
        cv2.line(frame, tpx, ipx, pinch_color, 1)


# ==============================================================================
# GRAFI
# ==============================================================================

def save_kinematic_graphs(output_graph_path,
                          times,
                          hand_tracker, thumb_tracker, index_tracker,
                          pinch_events, pinch_distances,
                          calibrated=False):
    """
    Shrani PNG graf s 4 subplot-i.
    Hitrosti in pospeški so prikazani v post-process glajeni obliki
    (če je smoother aktiven), surova krivulja je dodana kot tanka polprosojnica.
    """
    if not MATPLOTLIB_OK:
        return

    ud = "mm"    if calibrated else "px"
    uv = "mm/s"  if calibrated else "px/s"
    ua = "mm/s²" if calibrated else "px/s²"

    t = np.array(times)
    n = len(t)

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    fig.suptitle("Kinematični parametri – 9HPT", fontsize=13, fontweight="bold")

    # --- 1. Pot ---
    ax = axes[0]
    ax.plot(t, hand_tracker.paths[:n],  color="black",   lw=1.5, label="Roka (zapestje)")
    ax.plot(t, thumb_tracker.paths[:n], color="#E07000", lw=1.2, label="Palec (konica)",   alpha=0.85)
    ax.plot(t, index_tracker.paths[:n], color="#0060C0", lw=1.2, label="Kazalec (konica)", alpha=0.85)
    ax.set_ylabel("Pot [{}]".format(ud)); ax.legend(fontsize=8,loc="upper left"); ax.grid(True,alpha=0.3)

    # --- 2. Hitrost (glajeno + surovo) ---
    ax = axes[1]
    for tracker, clr_raw, clr_sm, lbl in [
        (hand_tracker,  "#909090", "black",   "Roka"),
        (thumb_tracker, "#F0A050", "#E07000", "Palec"),
        (index_tracker, "#5090D0", "#0060C0", "Kazalec"),
    ]:
        raw = np.array(tracker.vels[:n])
        sm  = tracker.smoothed_vels()[:n]
        #ax.plot(t[:len(raw)], raw, color=clr_raw, lw=0.6, alpha=0.35)
        ax.plot(t[:len(sm)],  sm,  color=clr_sm,  lw=1.3, label=lbl)
    ax.set_ylabel("Hitrost [{}]".format(uv)); ax.legend(fontsize=8,loc="upper left"); ax.grid(True,alpha=0.3)

    # --- 3. Pospešek (glajeno + surovo) ---
    ax = axes[2]
    for tracker, clr_raw, clr_sm, lbl in [
        (hand_tracker,  "#909090", "black",   "Roka"),
        (thumb_tracker, "#F0A050", "#E07000", "Palec"),
        (index_tracker, "#5090D0", "#0060C0", "Kazalec"),
    ]:
        raw = np.array(tracker.accs[:n])
        sm  = tracker.smoothed_accs()[:n]
        #ax.plot(t[:len(raw)], raw, color=clr_raw, lw=0.6, alpha=0.35)
        ax.plot(t[:len(sm)],  sm,  color=clr_sm,  lw=1.3, label=lbl)
    ax.set_ylabel("Pospešek [{}]".format(ua)); ax.legend(fontsize=8,loc="upper left"); ax.grid(True,alpha=0.3)

    # --- 4. Pinch ---
    ax4 = axes[3]
    if pinch_distances:
        pd  = np.array(pinch_distances[:n])
        ax4.plot(t[:len(pd)], pd, color="#008060", lw=1.2, label="Pinch razdalja")
        thr = PINCH_THRESHOLD_MM if calibrated else PINCH_THRESHOLD_PX
        ax4.axhline(thr, color="gray", lw=0.8, ls="--", label="Prag ({} {})".format(thr, ud))
        for ev in pinch_events:
            if ev["t"] <= t[-1]:
                clr = "green" if ev["type"]=="grasp" else "red"
                ax4.axvline(ev["t"], color=clr, lw=1.0, ls=":", alpha=0.8)
                ax4.text(ev["t"], pd.max()*0.9 if len(pd)>0 else thr*2,
                         "Prijem" if ev["type"]=="grasp" else "Odlaganje",
                         fontsize=7, color=clr, rotation=90, va="top")
    ax4.set_ylabel("Pinch [{}]".format(ud)); ax4.set_xlabel("Čas [s]")
    ax4.legend(fontsize=8,loc="upper right"); ax4.grid(True,alpha=0.3)

    # Opomba o glajenju
    sm_note = ""
    if hand_tracker.smoother is not None:
        sm_note = "  [Glajeno: {} σ={:.1f}]".format(
            hand_tracker.smoother.method.upper(), hand_tracker.smoother.sigma)
    fig.text(0.99, 0.01, sm_note, ha="right", va="bottom", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(output_graph_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Graf shranjen: {}".format(output_graph_path))


# ==============================================================================
# GLAVNA FUNKCIJA
# ==============================================================================

def process_video(input_path, output_path, show_roi=True, birds_eye=False,
                  bev_out_w=600, bev_out_h=600, bev_scale_px_per_mm=2.0,
                  smooth_sigma=1.5, smooth_method="gauss"):
    download_model()

    camera_name = detect_camera_from_filename(input_path)
    calibration = None; calibrated = False

    if camera_name:
        print("  Zaznana kamera: {}".format(camera_name))
        config = load_calibration_config(camera_name)
        if config:
            calibration = CameraCalibration(config); calibrated = True
            print("  Kalibracija aktivna: ppm={:.4f}".format(calibration.pixels_per_mm)
                  if calibration.pixels_per_mm else "  Kalibracija aktivna (homografija)")
        else:
            print("  [OPOZORILO] Kalibracija za '{}' ni najdena.".format(camera_name))
    else:
        print("  [INFO] Kamera ni prepoznana – pikselske enote.")

    # Smoother
    use_smooth = smooth_sigma > 0.0
    if use_smooth:
        print("  Glajenje: {} σ={:.1f} (okno={})".format(
            smooth_method.upper(), smooth_sigma,
            max(3, int(4*smooth_sigma)+1)))
        smoother_h = KinematicSmoother(sigma=smooth_sigma, method=smooth_method)
        smoother_t = KinematicSmoother(sigma=smooth_sigma, method=smooth_method)
        smoother_i = KinematicSmoother(sigma=smooth_sigma, method=smooth_method)
    else:
        print("  Glajenje: IZKLOPLJENO")
        smoother_h = smoother_t = smoother_i = None

    # Bird's-eye (zakomentirano)
    '''
    bev_M=bev_size=None
    if birds_eye and calibrated and calibration.homography is not None:
        bev_M,bev_size=build_birds_eye_transform(calibration.homography,
            out_w=bev_out_w,out_h=bev_out_h,scale_px_per_mm=bev_scale_px_per_mm)
        print("  Bird's-eye: VKLOPLJEN")
    else: birds_eye=False
    '''

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=2,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.3,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    hand_selector  = ActiveHandSelector(history_len=20, lock_after=30)
    hand_tracker   = KinematicTracker("roka",    smoother=smoother_h)
    thumb_tracker  = KinematicTracker("palec",   smoother=smoother_t)
    index_tracker  = KinematicTracker("kazalec", smoother=smoother_i)
    pinch_detector = PinchDetector()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Napaka: Ne morem odpreti {}".format(input_path)); return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps==0 or fps!=fps: fps=30.0
    dt = 1.0/fps

    roi_fracs = CAMERA_ROI.get(camera_name, CAMERA_ROI[None])
    rx1=int(width*roi_fracs[0]); ry1=int(height*roi_fracs[1])
    rx2=int(width*roi_fracs[2]); ry2=int(height*roi_fracs[3])
    print("  ROI: ({},{})–({},{}) px".format(rx1,ry1,rx2,ry2))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_w  = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Obdelava: {}  [{}x{} @ {:.1f}fps]".format(input_path,width,height,fps))

    frame_count = 0; times = []; wrist_trail = deque(maxlen=80)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_ud  = calibration.undistort_frame(frame) if calibrated else frame
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(frame_ud,cv2.COLOR_BGR2RGB))
        det       = detector.detect(mp_image)

        display      = frame_ud.copy()
        hand_found   = False
        active_g_idx = 0
        t            = frame_count*dt
        times.append(t)

        if det.hand_landmarks:
            all_wrist=[]; all_in_roi=[]
            for hl in det.hand_landmarks:
                pos=(hl[IDX_WRIST].x*width, hl[IDX_WRIST].y*height)
                if calibrated: pos=calibration.undistort_point(pos)
                all_wrist.append(pos)
                all_in_roi.append(rx1<=pos[0]<=rx2 and ry1<=pos[1]<=ry2)

            rpos=[p for p,r in zip(all_wrist,all_in_roi) if r]
            ridx=[i for i,r in enumerate(all_in_roi) if r]
            if not rpos: rpos=all_wrist; ridx=list(range(len(det.hand_landmarks)))

            sel=hand_selector.select(rpos)
            if sel is None: sel=0
            active_g_idx=ridx[sel]; hand_found=True
            ahl=det.hand_landmarks[active_g_idx]

            wrist_pos=(all_wrist[active_g_idx]); wrist_trail.append(wrist_pos)
            thlm=ahl[IDX_THUMB_TIP]; ilm=ahl[IDX_INDEX_TIP]
            thumb_pos=(thlm.x*width, thlm.y*height)
            index_pos=(ilm.x*width,  ilm.y*height)
            if calibrated:
                thumb_pos=calibration.undistort_point(thumb_pos)
                index_pos=calibration.undistort_point(index_pos)

            hand_tracker.set_time(t);  hand_tracker.update(wrist_pos, dt, calibration)
            thumb_tracker.set_time(t); thumb_tracker.update(thumb_pos, dt, calibration)
            index_tracker.set_time(t); index_tracker.update(index_pos, dt, calibration)
            pinch_detector.update(thumb_pos, index_pos, t, frame_count, calibration)

            for i,hl in enumerate(det.hand_landmarks):
                draw_hand_skeleton(display, hl, width, height,
                                   active=(i==active_g_idx),
                                   in_roi=all_in_roi[i],
                                   is_grasping=pinch_detector.is_grasping if i==active_g_idx else False)

        if not hand_found:
            hand_tracker.set_time(t);  hand_tracker.missing()
            thumb_tracker.set_time(t); thumb_tracker.missing()
            index_tracker.set_time(t); index_tracker.missing()
            pinch_detector.missing()

        if show_roi:
            display = draw_roi(display, rx1, ry1, rx2, ry2)

        display = draw_kinematic_hud(
            display,
            hand_tracker, thumb_tracker, index_tracker,
            calibrated=calibrated,
            is_grasping=pinch_detector.is_grasping,
            pinch_dist=pinch_detector.distances[-1] if pinch_detector.distances else None,
            smooth_sigma=smooth_sigma if use_smooth else None,
        )

        # Bird's-eye (zakomentirano)
        '''
        if birds_eye and bev_M is not None:
            bf=warp_to_birds_eye(frame_ud,bev_M,bev_size)
            bf=draw_birds_eye_overlay(bf,det.hand_landmarks or [],active_g_idx,
                                      bev_M,width,height,bev_scale_px_per_mm,
                                      wrist_trail,bev_out_w,bev_out_h)
            display=create_combined_frame(display,bf,bev_scale=0.35)
        '''

        out_w.write(display)
        frame_count += 1

    # --- Shranjevanje ---
    bn = os.path.splitext(os.path.basename(output_path))[0]

    log_path = os.path.join(LOG_DIR, bn+".log")
    with open(log_path,"w",encoding="utf-8") as f:
        f.write("Obdelan video: {}\n".format(input_path))
        f.write("Izhodna datoteka: {}\n".format(output_path))
        f.write("Okvirji: {}\n".format(frame_count))
        f.write("Kamera: {}\n".format(camera_name or "neznana"))
        f.write("Kalibracija: {}\n".format("DA" if calibrated else "NE"))
        if calibrated and calibration.pixels_per_mm:
            f.write("Pixels/mm: {:.4f}\n".format(calibration.pixels_per_mm))
        f.write("ROI: ({},{})-({},{}) px\n".format(rx1,ry1,rx2,ry2))
        f.write("Zaklep roke: {}\n".format("DA" if hand_selector.locked else "NE"))
        f.write("Glajenje: {} sigma={:.2f}\n".format(
            smooth_method.upper() if use_smooth else "NE", smooth_sigma))
        f.write("Dogodki prijema/odlaganja: {}\n".format(len(pinch_detector.events)))
        for ev in pinch_detector.events:
            f.write("  {}: t={:.3f}s  frame={}  dist={:.1f}\n".format(
                ev["type"],ev["t"],ev["frame"],ev["dist"]))

    ud=("mm" if calibrated else "px"); uv=("mm/s" if calibrated else "px/s"); ua=("mm/s2" if calibrated else "px/s2")
    kin_path = os.path.join(LOG_DIR, bn+"_kinematika.csv")
    with open(kin_path,"w",encoding="utf-8") as f:
        f.write("cas[s];pot_roka[{u}];v_roka[{v}];a_roka[{a}];"
                "pot_palec[{u}];v_palec[{v}];a_palec[{a}];"
                "pot_kazalec[{u}];v_kazalec[{v}];a_kazalec[{a}];pinch[{u}]\n".format(u=ud,v=uv,a=ua))
        n_rows=min(len(times),len(hand_tracker.paths),len(thumb_tracker.paths),len(index_tracker.paths))
        pd=pinch_detector.distances
        # Post-process glajene serije za CSV
        hv_sm = hand_tracker.smoothed_vels();  ha_sm = hand_tracker.smoothed_accs()
        tv_sm = thumb_tracker.smoothed_vels(); ta_sm = thumb_tracker.smoothed_accs()
        iv_sm = index_tracker.smoothed_vels(); ia_sm = index_tracker.smoothed_accs()
        for i in range(n_rows):
            f.write("{:.3f};{:.2f};{:.2f};{:.2f};{:.2f};{:.2f};{:.2f};{:.2f};{:.2f};{:.2f};{:.2f}\n".format(
                times[i],
                hand_tracker.paths[i],
                float(hv_sm[i]) if i<len(hv_sm) else hand_tracker.vels[i],
                float(ha_sm[i]) if i<len(ha_sm) else hand_tracker.accs[i],
                thumb_tracker.paths[i],
                float(tv_sm[i]) if i<len(tv_sm) else thumb_tracker.vels[i],
                float(ta_sm[i]) if i<len(ta_sm) else thumb_tracker.accs[i],
                index_tracker.paths[i],
                float(iv_sm[i]) if i<len(iv_sm) else index_tracker.vels[i],
                float(ia_sm[i]) if i<len(ia_sm) else index_tracker.accs[i],
                pd[i] if i<len(pd) else 0.0))

    graph_path = os.path.join(GRAPH_DIR, bn+"_graf.png")
    save_kinematic_graphs(graph_path, times,
                          hand_tracker, thumb_tracker, index_tracker,
                          pinch_detector.events, pinch_detector.distances,
                          calibrated=calibrated)

    cap.release(); out_w.release()
    print("Konec. Okvirjev: {}".format(frame_count))
    print("Video:      {}".format(output_path))
    print("Kinematika: {}".format(kin_path))
    print("Log:        {}".format(log_path))
    if MATPLOTLIB_OK: print("Graf:       {}".format(graph_path))


def get_mp4_files_recursively(data_dir):
    mp4s=[]
    for root,_,files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith('.mp4'): mp4s.append(os.path.join(root,f))
    return mp4s


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="9HPT analiza – kinematika roke, palca in kazalca z Gaussovim glajenjem.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Pot do vhodne .mp4 datoteke ali 'all' za vse videe v data/")
    parser.add_argument("--output", "-o", required=False,
                        help="Pot do izhodne .mp4 datoteke (samo pri obdelavi enega videa)")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Onemogoči kalibracijo – vsi izračuni v pikslih")
    parser.add_argument("--no-roi", action="store_true",
                        help="Skrij ROI pravokotnik na izhodnem videu")
    parser.add_argument("--roi", nargs=4, type=float, metavar=("X1","Y1","X2","Y2"),
                        help="Ročni ROI kot deleži slike (0.0–1.0), npr.: --roi 0.1 0.2 0.75 0.95")
    parser.add_argument("--lock-after", type=int, default=30,
                        help="Zaklep aktivne roke po N okvirjih (privzeto: 30)")
    parser.add_argument("--birds-eye", action="store_true",
                        help="Vstavi bird's-eye view vstavek (zahteva homografijo iz kalibracije)")
    parser.add_argument("--bev-size", nargs=2, type=int, default=[600,600], metavar=("W","H"),
                        help="Velikost bird's-eye platna v pikslih (privzeto: 600 600)")
    parser.add_argument("--bev-scale", type=float, default=2.0,
                        help="Merilo bird's-eye pogleda v px/mm (privzeto: 2.0)")
    parser.add_argument("--pinch-thr-mm", type=float, default=PINCH_THRESHOLD_MM,
                        help="Prag pinch razdalje v mm pri kalibriranem načinu (privzeto: {})".format(PINCH_THRESHOLD_MM))
    parser.add_argument("--pinch-thr-px", type=float, default=PINCH_THRESHOLD_PX,
                        help="Prag pinch razdalje v pikslih brez kalibracije (privzeto: {})".format(PINCH_THRESHOLD_PX))
    parser.add_argument("--smooth-sigma", type=float, default=1.5,
                        help=("Standardna deviacija Gaussovega glajenja položajev v okvirjih.\n"
                              "  0.0 = glajenje izklopljeno\n"
                              "  1.5 = privzeto (rahlo glajenje)\n"
                              "  3.0 = močnejše glajenje\n"
                              "Večja vrednost → bolj gladke krivulje, a manjša odzivnost."))
    parser.add_argument("--smooth-method", type=str, default="gauss",
                        choices=["gauss","savgol"],
                        help=("Metoda post-process glajenja za grafe/CSV:\n"
                              "  gauss  – Gaussova konvolucija (privzeto, brez scipy)\n"
                              "  savgol – Savitzky-Golay filter (zahteva scipy)"))

    args = parser.parse_args()

    if args.roi:
        for cam in list(CAMERA_ROI.keys()): CAMERA_ROI[cam]=tuple(args.roi)
        print("ROI ročno nastavljen: {}".format(args.roi))

    PINCH_THRESHOLD_MM = args.pinch_thr_mm
    PINCH_THRESHOLD_PX = args.pinch_thr_px

    print("="*60); print("9HPT Analiza – Robotski vid"); print("="*60)

    acal=[]
    for cn in ["left","mid","right"]:
        if os.path.exists(os.path.join(CALIBRATION_CONF_DIR,"{}_calibration.json".format(cn))):
            acal.append("{} (lastna)".format(cn))
        elif os.path.exists(os.path.join("calibration","calibration.json")):
            acal.append("{} (fallback)".format(cn))
    if acal and not args.no_calibration: print("Kalibracije: {}".format(", ".join(acal)))
    elif args.no_calibration: print("Kalibracija onemogočena.")
    else: print("[INFO] Kalibracija ni na voljo – pikselske enote.")

    if args.smooth_sigma > 0:
        print("Glajenje: {} σ={:.1f}".format(args.smooth_method.upper(), args.smooth_sigma))
        if args.smooth_method == "savgol" and not SCIPY_OK:
            print("  [OPOZORILO] scipy ni nameščen – fallback na GAUSS.")
    else:
        print("Glajenje: IZKLOPLJENO")
    print()

    show_roi=not args.no_roi; bw,bh=args.bev_size
    def run(ip,op):
        process_video(ip, op,
                      show_roi=show_roi,
                      birds_eye=args.birds_eye,
                      bev_out_w=bw, bev_out_h=bh,
                      bev_scale_px_per_mm=args.bev_scale,
                      smooth_sigma=args.smooth_sigma,
                      smooth_method=args.smooth_method)

    if args.input.lower()=="all":
        files=get_mp4_files_recursively(DATA_DIR)
        if not files: print("V '{}' ni mp4 datotek.".format(DATA_DIR))
        for ip in files:
            run(ip, os.path.join(OUTPUT_DIR,
                os.path.splitext(os.path.basename(ip))[0]+"_obdelan.mp4"))
    else:
        ip=args.input if os.path.isfile(args.input) else os.path.join(DATA_DIR,args.input)
        if not os.path.isfile(ip): print("Napaka: '{}' ni najden!".format(args.input)); exit(1)
        op=(os.path.join(OUTPUT_DIR,args.output) if args.output
            else os.path.join(OUTPUT_DIR,os.path.splitext(os.path.basename(ip))[0]+"_obdelan.mp4"))
        run(ip,op)