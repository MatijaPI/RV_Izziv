"""
Pomocna skripta za ugotavljanje pravilnih dimenzij sahovnice.
Poskusi razlicne konfiguracije kotov in poisce, katera deluje.

Uporaba:
    python calibration/scripts/detect_board_size.py pot/do/slike.jpg
"""

import cv2
import sys
import numpy as np

def try_all_sizes(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Napaka: ne morem prebrati {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    print(f"Testiram sliko: {img_path}")
    print(f"Velikost slike: {gray.shape[1]}x{gray.shape[0]}")
    print("=" * 50)

    found = False
    # Testiraj kombinacije od 3x3 do 12x12 notranjih kotov
    for cols in range(3, 13):
        for rows in range(3, cols + 1):  # rows <= cols da se izognemo duplikatom
            for rot_code in [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                if rot_code is not None:
                    test_img = cv2.rotate(enhanced, rot_code)
                else:
                    test_img = enhanced

                ret, corners = cv2.findChessboardCorners(test_img, (cols, rows), flags)
                if ret:
                    rot_deg = {None: 0, cv2.ROTATE_90_CLOCKWISE: 90,
                               cv2.ROTATE_180: 180, cv2.ROTATE_90_COUNTERCLOCKWISE: 270}[rot_code]
                    print(f"  ✓ NAJDENO: notranji koti = {cols}x{rows}  (kvadratki = {cols+1}x{rows+1})  rotacija={rot_deg}°")
                    print(f"    → Nastavi: CHECKERBOARD_COLS = {cols}, CHECKERBOARD_ROWS = {rows}")

                    # Shrani sliko z označenimi koti
                    out = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)
                    cv2.drawChessboardCorners(out, (cols, rows), corners, ret)
                    out_path = f"detected_{cols}x{rows}_rot{rot_deg}.jpg"
                    cv2.imwrite(out_path, out)
                    print(f"    → Shranjena debug slika: {out_path}")
                    found = True
                    break
            if found:
                break
        if found:
            break

    if not found:
        print("  ✗ Noben vzorec ni bil najden.")
        print("  Preveri ali je slika dovolj kakovostna in šahovnica vidna.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uporaba: python detect_board_size.py pot/do/slike.jpg")
        sys.exit(1)
    try_all_sizes(sys.argv[1])