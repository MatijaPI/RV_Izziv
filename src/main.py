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


def download_model():
    """Prenese MediaPipe model, če še ni v mapi."""
    if not os.path.exists(MODEL_PATH):
        print("Prenasam MediaPipe model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
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
        cam_key = f"camP_{match.group(1)}"
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
        print(f"  Naložena kalibracija iz: {config_path}")
        return config

    # Fallback: uporabi validacijsko datoteko (calibration.json)
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
                # Pretvori v interni format
                config = {
                    "camera_name": camera_name,
                    "camera_matrix": ref["cameraMatrix"],
                    "dist_coeffs": [ref["distortionCoeffs"][0]] if isinstance(ref["distortionCoeffs"][0], list) else [ref["distortionCoeffs"]],
                    "homography": ref.get("homography", None),
                    "pixels_per_mm": None,  # Izračunamo iz focal length
                    "source": "validation_fallback",
                }
                # Ocenimo pixels_per_mm iz homografije, če je na voljo
                if config["homography"] is not None:
                    H = np.array(config["homography"])
                    # pixels_per_mm iz homografije: inverz skalirnega faktorja
                    # Homografija preslika piksle v mm, torej je njena inverza mm->px
                    try:
                        H_inv = np.linalg.inv(H)
                        # Skalirni faktor = norma prvih dveh stolpcev H_inv
                        scale_x = np.linalg.norm(H_inv[:2, 0])
                        scale_y = np.linalg.norm(H_inv[:2, 1])
                        config["pixels_per_mm"] = (scale_x + scale_y) / 2.0
                    except np.linalg.LinAlgError:
                        config["pixels_per_mm"] = None

                print(f"  Naložena kalibracija (fallback) iz: {fb_path}")
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
        self.dist_coeffs = np.array(config["dist_coeffs"], dtype=np.float64)
        self.homography = np.array(config["homography"], dtype=np.float64) if config.get("homography") else None
        self.pixels_per_mm = config.get("pixels_per_mm", None)

        # Nova kamera matrika za undistortion
        self.new_camera_matrix = None
        if "new_camera_matrix" in config and config["new_camera_matrix"] is not None:
            self.new_camera_matrix = np.array(config["new_camera_matrix"], dtype=np.float64)

        self.is_calibrated = True
        self.camera_name = config.get("camera_name", "unknown")

    def undistort_frame(self, frame):
        """Odstrani distorzijo iz slike."""
        if self.new_camera_matrix is not None:
            return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)
        else:
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
        # Fallback: uporabi focal length za grobo oceno (predpostavka: povprečna delovna razdalja ~300mm)
        fx = self.camera_matrix[0, 0]
        # Brez dodatnih informacij uporabimo pixels_per_mm ≈ fx / working_distance
        # Privzeta delovna razdalja ~500mm za kalibracijo šahovnice
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
            pt = np.array([point_px[0], point_px[1], 1.0])
            result = self.homography @ pt
            if abs(result[2]) > 1e-10:
                return (result[0] / result[2], result[1] / result[2])
        # Fallback brez homografije
        return (self.pixel_to_mm(point_px[0]), self.pixel_to_mm(point_px[1]))


def draw_kinematic_hud(frame, path_mm, vel_mm, acc_mm, path_hist, vel_hist, acc_hist, calibrated=False):
    """Risanje elegantnega overlay-a in mini grafov za kinematične parametre."""
    overlay_w = 200
    overlay_h = 150
    margin = 12
    corner_x, corner_y = 12, 320
    graph_h = 30
    num_points = len(path_hist)
    alpha = 0.8

    # Enota glede na kalibracijo
    unit_dist = "mm" if calibrated else "px"
    unit_vel = "mm/s" if calibrated else "px/s"
    unit_acc = "mm/s2" if calibrated else "px/s2"
    
    # Ustvari prosojno ozadje
    overlay = frame.copy()
    cv2.rectangle(overlay, (corner_x, corner_y), 
                  (corner_x + overlay_w, corner_y + overlay_h), 
                  (245, 245, 245), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    
    # Izpis kinematike
    font = cv2.FONT_HERSHEY_TRIPLEX
    color_txt = (30, 30, 30)
    cv2.putText(frame, f"x: {path_mm:.1f} {unit_dist}", (corner_x + margin, corner_y + 104), font, 0.45, color_txt, 1, cv2.LINE_AA)
    cv2.putText(frame, f"v: {vel_mm:.1f} {unit_vel}", (corner_x + margin, corner_y + 124), font, 0.45, color_txt, 1, cv2.LINE_AA)
    cv2.putText(frame, f"a: {acc_mm:.1f} {unit_acc}", (corner_x + margin, corner_y + 144), font, 0.45, color_txt, 1, cv2.LINE_AA)

    # Oznaka kalibracije
    if calibrated:
        cv2.putText(frame, "[KALIBRIRANO]", (corner_x + margin, corner_y + 16), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 130, 0), 1, cv2.LINE_AA)
    
    # Real-time mini grafi (pot - črna, hitrost - modra, pospešek - rdeča)
    graph_x0 = corner_x + margin
    graph_y0 = corner_y + 40
    # Normaliziraj
    def normalize(vals): 
        return [(v - np.min(vals))/(np.max(vals) - np.min(vals) + 1e-5) if len(vals) > 1 else 0.5 for v in vals]
    path_norm = normalize(path_hist)
    vel_norm = normalize(vel_hist)
    acc_norm = normalize(acc_hist)
    for i in range(1, num_points):
        cv2.line(frame, (graph_x0 + i-1, graph_y0 - int(path_norm[i-1]*graph_h)), 
                        (graph_x0 + i,   graph_y0 - int(path_norm[i]*graph_h)), (30,30,30), 1)
        cv2.line(frame, (graph_x0 + i-1, graph_y0+18 - int(vel_norm[i-1]*graph_h)), 
                        (graph_x0 + i,   graph_y0+18 - int(vel_norm[i]*graph_h)), (80,70,200), 1)
        cv2.line(frame, (graph_x0 + i-1, graph_y0+45 - int(acc_norm[i-1]*graph_h)), 
                        (graph_x0 + i,   graph_y0+45 - int(acc_norm[i]*graph_h)), (150,40,60), 1)
    # Oznake osi
    cv2.putText(frame, "x", (graph_x0-5, graph_y0-2), font, 0.37, (30,30,30), 1, cv2.LINE_AA)
    cv2.putText(frame, "v", (graph_x0-5, graph_y0+18), font, 0.37, (80,70,200), 1, cv2.LINE_AA)
    cv2.putText(frame, "a", (graph_x0-5, graph_y0+36), font, 0.37, (150,40,60), 1, cv2.LINE_AA)
    return frame


def process_video(input_path, output_path):
    """Obdelava videa z detekcijo roke in kinematičnimi izračuni."""
    download_model()

    # Zaznaj kamero iz imena datoteke in naloži kalibracijo
    camera_name = detect_camera_from_filename(input_path)
    calibration = None
    calibrated = False

    if camera_name:
        print(f"  Zaznana kamera: {camera_name} (iz imena datoteke)")
        config = load_calibration_config(camera_name)
        if config is not None:
            calibration = CameraCalibration(config)
            calibrated = True
            print(f"  Kalibracija aktivna: pixels_per_mm = {calibration.pixels_per_mm:.4f}" 
                  if calibration.pixels_per_mm else "  Kalibracija aktivna (homografija)")
        else:
            print(f"  [OPOZORILO] Kalibracijska konfiguracija za '{camera_name}' ni najdena.")
            print(f"  Zaženite kalibracijo: python calibration/scripts/calibrate.py")
    else:
        print(f"  [INFO] Kamera ni prepoznana iz imena datoteke. Uporaba pikselskih enot.")

    # MediaPipe detektor
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Odpre video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Napaka: Ni mogoce odpreti videa {input_path}")
        return

    # Podatki o videu
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0
    dt = 1.0 / fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Zacetek obdelave videa: {input_path}")
    print(f"  Resolucija: {width}x{height}, FPS: {fps:.1f}")
    if calibrated:
        print(f"  Enote: metrične (mm, mm/s, mm/s²)")
    else:
        print(f"  Enote: pikselske (px, px/s, px/s²)")
    frame_count = 0

    # Spremenljivke za kinematiko (v pikslih, pretvorba na koncu)
    prev_pos = None
    prev_vel_px = 0.0
    path_sum_px = 0.0

    # Seznami za časovno vrsto
    times = []
    paths = []       # v metričnih enotah (mm ali px)
    vels = []        # v metričnih enotah (mm/s ali px/s)
    accs = []        # v metričnih enotah (mm/s² ali px/s²)
    paths_px = []    # vedno v pikslih (za referenco)
    hist_len = 100
    path_hist = deque(maxlen=hist_len)
    vel_hist = deque(maxlen=hist_len)
    acc_hist = deque(maxlen=hist_len)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Odstranitev distorzije, če je kalibracija na voljo
        if calibrated:
            frame_undistorted = calibration.undistort_frame(frame)
        else:
            frame_undistorted = frame

        # Pretvorba v RGB
        image_rgb = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detekcija
        detection_result = detector.detect(mp_image)

        # Izris detekcij in kinematika
        wrist_found = False
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                wrist = hand_landmarks[0]
                curr_pos_px = (wrist.x * width, wrist.y * height)

                # Odstranitev distorzije točke
                if calibrated:
                    curr_pos_px = calibration.undistort_point(curr_pos_px)

                wrist_found = True

                # Izračun kinematičnih podatkov v pikslih
                if prev_pos is not None:
                    dx = curr_pos_px[0] - prev_pos[0]
                    dy = curr_pos_px[1] - prev_pos[1]
                    d_px = (dx**2 + dy**2)**0.5
                else:
                    d_px = 0.0

                path_sum_px += d_px
                curr_vel_px = d_px / dt if prev_pos is not None else 0.0
                acc_px = (curr_vel_px - prev_vel_px) / dt if prev_pos is not None else 0.0

                # Pretvorba v metrične enote
                if calibrated:
                    path_metric = calibration.pixel_to_mm(path_sum_px)
                    vel_metric = calibration.pixel_velocity_to_mm_s(curr_vel_px)
                    acc_metric = calibration.pixel_acc_to_mm_s2(acc_px)
                else:
                    path_metric = path_sum_px
                    vel_metric = curr_vel_px
                    acc_metric = acc_px

                # Shranjevanje rezultatov
                t = frame_count * dt
                times.append(t)
                paths.append(path_metric)
                vels.append(vel_metric)
                accs.append(acc_metric)
                paths_px.append(path_sum_px)
                path_hist.append(path_metric)
                vel_hist.append(vel_metric)
                acc_hist.append(acc_metric)

                prev_pos = curr_pos_px
                prev_vel_px = curr_vel_px

                break  # Upoštevamo samo prvo zaznano roko

        if not wrist_found:
            t = frame_count * dt
            times.append(t)
            paths.append(paths[-1] if paths else 0.0)
            vels.append(0.0)
            accs.append(0.0)
            paths_px.append(path_sum_px)
            path_hist.append(path_hist[-1] if path_hist else 0.0)
            vel_hist.append(0.0)
            acc_hist.append(0.0)

        # Izris detekcij na sliko (uporabimo undistortirano sliko)
        display_frame = frame_undistorted.copy()
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                for connection in HAND_CONNECTIONS:
                    p1 = hand_landmarks[connection[0]]
                    p2 = hand_landmarks[connection[1]]
                    cv2.line(display_frame, (int(p1.x * width), int(p1.y * height)), 
                                    (int(p2.x * width), int(p2.y * height)), (255, 255, 255), 2)
                for landmark in hand_landmarks:
                    cv2.circle(display_frame, (int(landmark.x * width), int(landmark.y * height)), 4, (0, 0, 0), -1)

        # Izris overlay za kinematiko
        display_frame = draw_kinematic_hud(
            display_frame, paths[-1], vels[-1], accs[-1], 
            path_hist, vel_hist, acc_hist, calibrated=calibrated
        )

        out.write(display_frame)
        frame_count += 1

    # Zapis dnevnika obdelave
    log_path = os.path.join(LOG_DIR, os.path.splitext(os.path.basename(output_path))[0] + ".log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Obdelan video: {input_path}\n")
        f.write(f"Izhodna datoteka: {output_path}\n")
        f.write(f"Število okvirjev: {frame_count}\n")
        f.write(f"Kamera: {camera_name if camera_name else 'neznana'}\n")
        f.write(f"Kalibracija: {'DA' if calibrated else 'NE'}\n")
        if calibrated and calibration.pixels_per_mm:
            f.write(f"Pixels per mm: {calibration.pixels_per_mm:.4f}\n")

    # Zapis kinematičnih podatkov
    kin_path = os.path.join(LOG_DIR, os.path.splitext(os.path.basename(output_path))[0] + "_kinematika.csv")
    with open(kin_path, "w", encoding="utf-8") as f:
        if calibrated:
            f.write("cas[s];pot[mm];hitrost[mm/s];pospesek[mm/s2];pot_px[px]\n")
            for t, s, v, a, s_px in zip(times, paths, vels, accs, paths_px):
                f.write(f"{t:.3f};{s:.2f};{v:.2f};{a:.2f};{s_px:.2f}\n")
        else:
            f.write("cas[s];pot[px];hitrost[px/s];pospesek[px/s2]\n")
            for t, s, v, a in zip(times, paths, vels, accs):
                f.write(f"{t:.3f};{s:.2f};{v:.2f};{a:.2f}\n")

    # Zapre video
    cap.release()
    out.release()
    print(f"Konec obdelave. Prebranih okvirjev: {frame_count}")
    print(f"Video shranjen v: {output_path}")
    print(f"Kinematika shranjena v: {kin_path}")


def get_mp4_files_recursively(data_dir):
    """Poišče vse .mp4 v vseh podmapah data_dir."""
    mp4_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith('.mp4'):
                mp4_files.append(os.path.join(root, f))
    return mp4_files


# Glavna funkcija in CLI argumenti
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Obdelava .mp4 video datotek za 9HPT z MediaPipe hand tracking in kalibracijo kamer."
    )
    parser.add_argument("--input", "-i", required=True, 
                        help="Ime ali pot do vhodne .mp4 datoteke ali 'all' za obdelavo vseh *.mp4 v data/")
    parser.add_argument("--output", "-o", required=False, 
                        help="Ime ali pot za izhodno .mp4 datoteko (uporabi le pri obdelavi ene datoteke)")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Onemogoči uporabo kalibracije (vsi izračuni v pikslih)")

    args = parser.parse_args()

    print("=" * 60)
    print("9HPT Analiza - Robotski vid")
    print("=" * 60)

    # Preveri razpoložljivost kalibracij
    available_calibrations = []
    for cam_name in ["left", "mid", "right"]:
        config_path = os.path.join(CALIBRATION_CONF_DIR, f"{cam_name}_calibration.json")
        fallback_path = os.path.join("calibration", "calibration.json")
        if os.path.exists(config_path):
            available_calibrations.append(f"{cam_name} (lastna)")
        elif os.path.exists(fallback_path):
            available_calibrations.append(f"{cam_name} (fallback)")

    if available_calibrations and not args.no_calibration:
        print(f"Razpoložljive kalibracije: {', '.join(available_calibrations)}")
    elif args.no_calibration:
        print("Kalibracija onemogočena (--no-calibration)")
    else:
        print("[INFO] Nobena kalibracija ni na voljo. Uporaba pikselskih enot.")
        print("  Za kalibracijo zaženite: python calibration/scripts/calibrate.py")

    print()

    if args.input.lower() == "all":
        files = get_mp4_files_recursively(DATA_DIR)
        if not files:
            print(f"V mapi '{DATA_DIR}' ni nobene mp4 datoteke za obdelavo.")
        for in_path in files:
            in_basename = os.path.splitext(os.path.basename(in_path))[0]
            out_name = in_basename + "_obdelan.mp4"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            process_video(in_path, out_path)
    else:
        # Poišči absolutno ali relativno pot, ali pa v data/
        if os.path.isfile(args.input):
            in_path = args.input
        else:
            in_path = os.path.join(DATA_DIR, args.input)
        if not os.path.isfile(in_path):
            print(f"Napaka: Ne najdem datoteke '{args.input}'!")
            exit(1)
        # Če je output podan, uporabimo tistega, sicer generiramo ime
        if args.output:
            out_path = os.path.join(OUTPUT_DIR, args.output)
        else:
            in_basename = os.path.splitext(os.path.basename(in_path))[0]
            out_name = in_basename + "_obdelan.mp4"
            out_path = os.path.join(OUTPUT_DIR, out_name)
        process_video(in_path, out_path)