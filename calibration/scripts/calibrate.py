"""
Kalibracijska skripta za vse tri kamere (left, mid, right).

Uporaba:
    python calibration/scripts/calibrate.py

Skripta poišče kalibracijske slike v:
    calibration/left/   (leva kamera - camP_0)
    calibration/mid/    (sredinska kamera - camP_1)
    calibration/right/  (desna kamera - camP_2)

Izhodne kalibracijske konfiguracije se shranijo v:
    calibration/conf/left_calibration.json
    calibration/conf/mid_calibration.json
    calibration/conf/right_calibration.json

Šahovnica: privzeto 8x8 kvadratkov (7x7 notranjih kotov), velikost stranice 20mm.
POZOR: Nastavi CHECKERBOARD_ROWS in CHECKERBOARD_COLS glede na svojo šahovnico!
"""

import cv2
import numpy as np
import json
import os
import sys
import glob

# ============================================================
# PARAMETRI ŠAHOVNICE - PRILAGODI PO POTREBI!
# Notranji koti = število kvadratkov - 1 (v vsaki smeri)
# Primer: 8x8 šahovnica → 7x7 notranjih kotov
#         9x6 šahovnica → 8x5 notranjih kotov
# ============================================================
CHECKERBOARD_ROWS = 6  # notranji koti po višini
CHECKERBOARD_COLS = 9  # notranji koti po širini
SQUARE_SIZE_MM = 20.0  # velikost stranice kvadratka v milimetrih

# Direktoriji
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMERAS = {
    "left": os.path.join(BASE_DIR, "left"),
    "mid": os.path.join(BASE_DIR, "mid"),
    "right": os.path.join(BASE_DIR, "right"),
}
CONF_DIR = os.path.join(BASE_DIR, "conf")
DEBUG_DIR = os.path.join(BASE_DIR, "debug")


def find_calibration_images(camera_dir):
    """Poišče vse slike (jpg, png, bmp) v direktoriju."""
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]
    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(camera_dir, ext)))
    images.sort()
    return images


def preprocess_image(gray):
    """
    Predprocesiranje slike za boljšo detekcijo kotov.
    Uporabi CLAHE za izboljšanje kontrasta (pomaga pri neenakomernem osvetljevanju).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced


def find_corners_with_rotations(gray, pattern_size):
    """
    Poskusi najti kote šahovnice pri vseh 4 orientacijah slike (0°, 90°, 180°, 270°).
    Vrne (ret, corners, rotated_gray) za prvo uspešno orientacijo.
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    rotations = [
        (0, gray),
        (1, cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)),
        (2, cv2.rotate(gray, cv2.ROTATE_180)),
        (3, cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    for rot_code, rotated in rotations:
        # Poskusi z originalnim kontrastom
        ret, corners = cv2.findChessboardCorners(rotated, pattern_size, flags)
        if ret:
            return ret, corners, rotated, rot_code

        # Poskusi z CLAHE predprocesiranjem
        enhanced = preprocess_image(rotated)
        ret, corners = cv2.findChessboardCorners(enhanced, pattern_size, flags)
        if ret:
            return ret, corners, enhanced, rot_code

    return False, None, gray, 0


def calibrate_camera(image_paths, camera_name):
    """
    Izvede kalibracijo kamere z uporabo šahovničnih slik.
    Vrne kalibracijsko konfiguracijo ali None v primeru napake.
    """
    pattern_size = (CHECKERBOARD_COLS, CHECKERBOARD_ROWS)

    # Priprava objekt in slikovnih točk
    objp = np.zeros((CHECKERBOARD_ROWS * CHECKERBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_COLS, 0:CHECKERBOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM  # v milimetrih

    obj_points = []  # 3D točke v realnem prostoru
    img_points = []  # 2D točke v slikovni ravnini
    image_size = None

    print(f"\n{'='*60}")
    print(f"Kalibracija kamere: {camera_name}")
    print(f"{'='*60}")
    print(f"Iščem vzorec: {CHECKERBOARD_COLS}x{CHECKERBOARD_ROWS} notranjih kotov")
    print(f"Število najdenih slik: {len(image_paths)}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    # Mapa za debug slike
    debug_camera_dir = os.path.join(DEBUG_DIR, camera_name)
    os.makedirs(debug_camera_dir, exist_ok=True)

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"  [OPOZORILO] Ni mogoce prebrati slike: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        # Poišči šahovnične kote (s poskusi rotacij)
        ret, corners, processed_gray, rot_code = find_corners_with_rotations(gray, pattern_size)

        if ret:
            # Natančnejša lokalizacija kotov
            corners_refined = cv2.cornerSubPix(processed_gray, corners, (11, 11), (-1, -1), criteria)

            # Za obj_points uporabimo originalne dimenzije (brez rotacije)
            obj_points.append(objp)
            img_points.append(corners_refined)

            rot_msg = f" (rotacija: {rot_code * 90}°)" if rot_code != 0 else ""
            print(f"  [OK] Koti najdeni: {os.path.basename(img_path)}{rot_msg}")

            # Shrani debug sliko z označenimi koti
            debug_img = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(debug_img, pattern_size, corners_refined, ret)
            debug_path = os.path.join(debug_camera_dir, f"detected_{os.path.basename(img_path)}")
            cv2.imwrite(debug_path, debug_img)

        else:
            print(f"  [NEUSPEH] Koti niso najdeni: {os.path.basename(img_path)}")
            # Shrani tudi neuspešno sliko za diagnostiko
            debug_path = os.path.join(debug_camera_dir, f"failed_{os.path.basename(img_path)}")
            cv2.imwrite(debug_path, gray)

    if len(obj_points) < 1:
        print(f"  [NAPAKA] Premalo uspesnih detekcij za kalibracijo kamere {camera_name}!")
        print(f"  [HINT] Preveri debug slike v: {debug_camera_dir}")
        print(f"  [HINT] Trenutna konfiguracija: {CHECKERBOARD_COLS}x{CHECKERBOARD_ROWS} notranjih kotov")
        print(f"  [HINT] Preštej notranje kote na svoji šahovnici in popravi CHECKERBOARD_ROWS/COLS!")
        return None

    print(f"\n  Uspesnih detekcij: {len(obj_points)}/{len(image_paths)}")
    print(f"  Izvajam kalibracijo...")

    # Kalibracija kamere
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    if not ret:
        print(f"  [NAPAKA] Kalibracija neuspesna za kamero {camera_name}!")
        return None

    # Izračun reprojekcijske napake
    total_error = 0
    for i in range(len(obj_points)):
        img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], img_points_proj, cv2.NORM_L2) / len(img_points_proj)
        total_error += error
    mean_error = total_error / len(obj_points)

    print(f"  Reprojekcijska napaka (RMS): {mean_error:.4f} px")
    print(f"  Kamera matrika:\n{camera_matrix}")
    print(f"  Distorzijski koeficienti: {dist_coeffs.ravel()}")

    # Izračun homografije (perspektivna transformacija za pretvorbo px -> mm)
    if len(img_points) > 0:
        last_img_pts = img_points[-1].reshape(-1, 2)
        last_obj_pts = objp[:, :2].astype(np.float32)
        homography, mask = cv2.findHomography(last_img_pts, last_obj_pts, cv2.RANSAC, 5.0)
    else:
        homography = None

    # Izračun pixels_per_mm
    pixels_per_mm = compute_pixels_per_mm(img_points, SQUARE_SIZE_MM)

    # Optimalna nova kamera matrika
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, image_size, 1, image_size
    )

    config = {
        "camera_name": camera_name,
        "image_size": list(image_size),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "new_camera_matrix": new_camera_matrix.tolist(),
        "roi": list(roi),
        "reprojection_error": mean_error,
        "square_size_mm": SQUARE_SIZE_MM,
        "checkerboard_size": [CHECKERBOARD_COLS, CHECKERBOARD_ROWS],
        "num_images_used": len(obj_points),
        "pixels_per_mm": pixels_per_mm,
        "homography": homography.tolist() if homography is not None else None,
    }

    print(f"  Pixels per mm: {pixels_per_mm:.4f}")
    print(f"  Kalibracija uspesna!")
    return config


def compute_pixels_per_mm(img_points_list, square_size_mm):
    """
    Izračuna povprečno razmerje pikslov na milimeter iz vseh detekcij.
    """
    all_distances = []
    for img_pts in img_points_list:
        pts = img_pts.reshape(-1, 2)
        for row in range(CHECKERBOARD_ROWS):
            for col in range(CHECKERBOARD_COLS - 1):
                idx1 = row * CHECKERBOARD_COLS + col
                idx2 = row * CHECKERBOARD_COLS + col + 1
                d = np.linalg.norm(pts[idx1] - pts[idx2])
                all_distances.append(d)
        for col in range(CHECKERBOARD_COLS):
            for row in range(CHECKERBOARD_ROWS - 1):
                idx1 = row * CHECKERBOARD_COLS + col
                idx2 = (row + 1) * CHECKERBOARD_COLS + col
                d = np.linalg.norm(pts[idx1] - pts[idx2])
                all_distances.append(d)

    if len(all_distances) == 0:
        return 1.0

    avg_pixel_dist = np.mean(all_distances)
    pixels_per_mm = avg_pixel_dist / square_size_mm
    return pixels_per_mm


def save_config(config, camera_name):
    """Shrani kalibracijsko konfiguracijo v JSON."""
    os.makedirs(CONF_DIR, exist_ok=True)
    output_path = os.path.join(CONF_DIR, f"{camera_name}_calibration.json")

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    config_serializable = convert(config)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_serializable, f, indent=2, ensure_ascii=False)
    print(f"  Konfiguracija shranjena: {output_path}")
    return output_path


def validate_calibration(config, camera_name):
    """Preveri kalibracijo z validacijsko datoteko, če obstaja."""
    validation_path = os.path.join(BASE_DIR, "calibration.json")
    if not os.path.exists(validation_path):
        validation_path = os.path.join(os.path.dirname(BASE_DIR), "calibration.json")
    if not os.path.exists(validation_path):
        print(f"  [INFO] Validacijska datoteka ne obstaja, preskakujem validacijo.")
        return

    with open(validation_path, "r") as f:
        validation_data = json.load(f)

    if camera_name not in validation_data:
        print(f"  [INFO] Kamera '{camera_name}' ni v validacijski datoteki.")
        return

    ref = validation_data[camera_name]
    ref_matrix = np.array(ref["cameraMatrix"])
    our_matrix = np.array(config["camera_matrix"])

    ref_fx, ref_fy = ref_matrix[0, 0], ref_matrix[1, 1]
    our_fx, our_fy = our_matrix[0, 0], our_matrix[1, 1]

    fx_diff = abs(ref_fx - our_fx) / ref_fx * 100
    fy_diff = abs(ref_fy - our_fy) / ref_fy * 100

    print(f"\n  --- Validacija ({camera_name}) ---")
    print(f"  Ref fx={ref_fx:.2f}, naš fx={our_fx:.2f} (razlika: {fx_diff:.2f}%)")
    print(f"  Ref fy={ref_fy:.2f}, naš fy={our_fy:.2f} (razlika: {fy_diff:.2f}%)")

    if fx_diff < 5 and fy_diff < 5:
        print(f"  [OK] Kalibracija je v mejah natančnosti (<5% razlika).")
    else:
        print(f"  [OPOZORILO] Večja razlika v kalibracijskih parametrih (>{5}%).")
        print(f"  To je lahko posledica uporabe različnih kalibracijskih slik.")


def main():
    """Glavna funkcija za kalibracijo vseh treh kamer."""
    print("=" * 60)
    print("KALIBRACIJA KAMER - Robotski vid izziv")
    print("=" * 60)
    print(f"Šahovnica: {CHECKERBOARD_COLS+1}x{CHECKERBOARD_ROWS+1} kvadratkov")
    print(f"Velikost kvadratka: {SQUARE_SIZE_MM} mm")
    print(f"Notranji koti: {CHECKERBOARD_COLS}x{CHECKERBOARD_ROWS}")

    results = {}
    success_count = 0

    for camera_name, camera_dir in CAMERAS.items():
        images = find_calibration_images(camera_dir)

        if len(images) == 0:
            print(f"\n[OPOZORILO] Ni kalibracijskih slik za kamero '{camera_name}' v {camera_dir}")
            print(f"  Dodajte slike (jpg/png) v mapo: {camera_dir}")
            continue

        config = calibrate_camera(images, camera_name)

        if config is not None:
            save_config(config, camera_name)
            validate_calibration(config, camera_name)
            results[camera_name] = config
            success_count += 1

    print(f"\n{'='*60}")
    print(f"REZULTAT: Uspesno kalibriranih kamer: {success_count}/3")
    print(f"{'='*60}")

    if success_count == 0:
        print("\n[INFO] Nobena kamera ni bila kalibrirana.")
        print("Mozni vzroki:")
        print("  1. Napacna konfiguracija šahovnice (CHECKERBOARD_ROWS/COLS)")
        print("     → Prestej notranje kote na svoji sahovnici in popravi vrednosti!")
        print("  2. Premalo slik ali slaba kakovost slik")
        print("     → Dodaj vec slik iz razlicnih kotov (priporoceno 10-20 slik)")
        print("  3. Poglej debug slike v: calibration/debug/")
        print("\nDodajte kalibracijske slike v ustrezne mape:")
        print(f"  - Leva kamera (camP_0):     calibration/left/")
        print(f"  - Sredinska kamera (camP_1): calibration/mid/")
        print(f"  - Desna kamera (camP_2):     calibration/right/")
        print("\nZa izvedbo kalibracije znova zaženite:")
        print("  python calibration/scripts/calibrate.py")
        sys.exit(1)

    return results


if __name__ == "__main__":
    main()