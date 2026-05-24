import cv2
import os
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

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Povezave med značilkami roke
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), 
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
                    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]

def download_model():
    # Prenese MediaPipe model, če še ni v mapi
    if not os.path.exists(MODEL_PATH):
        print("Prenasam MediaPipe model...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("Prenos uspesen.")

def draw_kinematic_hud(frame, path, vel, acc, path_hist, vel_hist, acc_hist):
    # Risanje elegantnega overlay-a in mini grafov za kinematične parametre
    overlay_w = 180
    overlay_h = 150
    margin = 12
    corner_x, corner_y = 12, 320
    graph_h = 30
    num_points = len(path_hist)
    alpha = 0.8 # prosojnost ozadja
    
    # Ustvari prosojno ozadje
    overlay = frame.copy()
    cv2.rectangle(overlay, (corner_x, corner_y), 
                  (corner_x + overlay_w, corner_y + overlay_h), 
                  (245, 245, 245), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    
    # Izpis kinematike
    font = cv2.FONT_HERSHEY_TRIPLEX
    color_txt = (30, 30, 30)
    cv2.putText(frame, f"x: {path:.1f} px", (corner_x + margin, corner_y + 104), font, 0.55, color_txt, 1, cv2.LINE_AA)
    cv2.putText(frame, f"v: {vel:.1f} px/s", (corner_x + margin, corner_y + 124), font, 0.55, color_txt, 1, cv2.LINE_AA)
    cv2.putText(frame, f"a: {acc:.1f} px/s2", (corner_x + margin, corner_y + 144), font, 0.55, color_txt, 1, cv2.LINE_AA)
    
    # Real-time mini grafi (pot - črna, hitrost - modra, pospešek - rdeča)
    graph_x0 = corner_x + margin
    graph_y0 = corner_y + 40
    graph_w = overlay_w - 24
    # Normaliziraj
    def normalize(vals): return [(v - np.min(vals))/(np.max(vals) - np.min(vals) + 1e-5) if len(vals) > 1 else 0.5 for v in vals]
    path_norm = normalize(path_hist)
    vel_norm = normalize(vel_hist)
    acc_norm = normalize(acc_hist)
    for i in range(1, num_points):
        # Pot
        cv2.line(frame, (graph_x0 + i-1, graph_y0 - int(path_norm[i-1]*graph_h)), 
                        (graph_x0 + i,   graph_y0 - int(path_norm[i]*graph_h)), (30,30,30), 1)
        # Hitrost
        cv2.line(frame, (graph_x0 + i-1, graph_y0+18 - int(vel_norm[i-1]*graph_h)), 
                        (graph_x0 + i,   graph_y0+18 - int(vel_norm[i]*graph_h)), (80,70,200), 1)
        # Pospešek
        cv2.line(frame, (graph_x0 + i-1, graph_y0+45 - int(acc_norm[i-1]*graph_h)), 
                        (graph_x0 + i,   graph_y0+45 - int(acc_norm[i]*graph_h)), (150,40,60), 1)
    # Oznake osi
    cv2.putText(frame, "x", (graph_x0-5, graph_y0-2), font, 0.37, (30,30,30), 1, cv2.LINE_AA)
    cv2.putText(frame, "v",   (graph_x0-5, graph_y0+18), font, 0.37, (80,70,200), 1, cv2.LINE_AA)
    cv2.putText(frame, "a",   (graph_x0-5, graph_y0+36), font, 0.37, (150,40,60), 1, cv2.LINE_AA)
    return frame

def process_video(input_path, output_path):
    download_model()

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
    frame_count = 0

    # Spremenljivke za kinematiko
    prev_pos = None
    prev_vel = 0.0
    path_sum = 0.0

    # Seznami za časovno vrsto (čas, pot, hitrost, pospešek), za izpis in graf
    times = []
    paths = []
    vels = []
    accs = []
    hist_len = 100
    path_hist = deque(maxlen=hist_len)
    vel_hist = deque(maxlen=hist_len)
    acc_hist = deque(maxlen=hist_len)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Pretvorba v RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detekcija
        detection_result = detector.detect(mp_image)

        # Izris detekcij in kinematika
        wrist_found = False
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Začnemo z levo roko (oz. prvo detektirano)
                wrist = hand_landmarks[0]
                curr_pos = (wrist.x * width, wrist.y * height)
                wrist_found = True

                # Izračun kin. podatkov
                if prev_pos is not None:
                    dx = curr_pos[0] - prev_pos[0]
                    dy = curr_pos[1] - prev_pos[1]
                    d = (dx**2 + dy**2)**0.5
                else:
                    d = 0.0
                path_sum += d
                curr_vel = d / dt if prev_pos is not None else 0.0
                acc = (curr_vel - prev_vel) / dt if prev_pos is not None else 0.0

                # Shranjevanje rezultatov
                t = frame_count * dt
                times.append(t)
                paths.append(path_sum)
                vels.append(curr_vel)
                accs.append(acc)
                path_hist.append(path_sum)
                vel_hist.append(curr_vel)
                acc_hist.append(acc)

                prev_pos = curr_pos
                prev_vel = curr_vel

                break # Upoštevamo samo prvo zaznano roko

        if not wrist_found:
            # Če ni roke, shranimo prejšnje podatke (ostanejo enaki)
            t = frame_count * dt
            times.append(t)
            paths.append(path_sum)
            vels.append(0.0)
            accs.append(0.0)
            path_hist.append(path_sum)
            vel_hist.append(0.0)
            acc_hist.append(0.0)

        # Izris detekcij na sliko
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Izris povezav
                for connection in HAND_CONNECTIONS:
                    p1 = hand_landmarks[connection[0]]
                    p2 = hand_landmarks[connection[1]]
                    cv2.line(frame, (int(p1.x * width), int(p1.y * height)), 
                                    (int(p2.x * width), int(p2.y * height)), (255, 255, 255), 2)
                # Izris ključnih točk
                for landmark in hand_landmarks:
                    cv2.circle(frame, (int(landmark.x * width), int(landmark.y * height)), 4, (0, 0, 0), -1)

        # Izris overlay za kinematiko (HUD z napisi in real-time mini grafi)
        frame = draw_kinematic_hud(frame, paths[-1], vels[-1], accs[-1], path_hist, vel_hist, acc_hist)

        out.write(frame)
        frame_count += 1

    # Zapis dnevnika obdelave
    log_path = os.path.join(LOG_DIR, os.path.splitext(os.path.basename(output_path))[0] + ".log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Obdelan video: {input_path}\n")
        f.write(f"Izhodna datoteka: {output_path}\n")
        f.write(f"Število okvirjev: {frame_count}\n")

    # Zapis kinematičnih podatkov
    kin_path = os.path.join(LOG_DIR, os.path.splitext(os.path.basename(output_path))[0] + "_kinematika.csv")
    with open(kin_path, "w", encoding="utf-8") as f:
        f.write("cas[s];pot[px];hitrost[px/s];pospesek[px/s2]\n")
        for t, s, v, a in zip(times, paths, vels, accs):
            f.write(f"{t:.2f};{s:.2f};{v:.2f};{a:.2f}\n")

    # Zapre video
    cap.release()
    out.release()
    print(f"Konec obdelave. Prebranih okvirjev: {frame_count}")
    print(f"Video shranjen v: {output_path}")
    print(f"Kinematika shranjena v: {kin_path}")

def get_mp4_files_recursively(data_dir):
    # Poišče vse .mp4 v vseh podmapah data_dir
    mp4_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith('.mp4'):
                mp4_files.append(os.path.join(root, f))
    return mp4_files

# Glavna funkcija in CLI argumenti
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Obdelava .mp4 video datotek za 9HPT z MediaPipe hand tracking."
    )
    parser.add_argument("--input", "-i", required=True, help="Ime ali pot do vhodne .mp4 datoteke ali 'all' za obdelavo vseh *.mp4 v data/")
    parser.add_argument("--output", "-o", required=False, help="Ime ali pot za izhodno .mp4 datoteko (uporabi le pri obdelavi ene datoteke)")

    args = parser.parse_args()

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