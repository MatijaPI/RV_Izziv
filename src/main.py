import cv2
import os
import argparse
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Zacetek obdelave videa: {input_path}")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Pretvorba v RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detekcija
        detection_result = detector.detect(mp_image)

        # Izris detekcij
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

        out.write(frame)
        frame_count += 1

    # Zapis dnevnika obdelave
    log_path = os.path.join(LOG_DIR, os.path.splitext(os.path.basename(output_path))[0] + ".log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Obdelan video: {input_path}\n")
        f.write(f"Izhodna datoteka: {output_path}\n")
        f.write(f"Število okvirjev: {frame_count}\n")

    # Zapre video
    cap.release()
    out.release()
    print(f"Konec obdelave. Prebranih okvirjev: {frame_count}")
    print(f"Video shranjen v: {output_path}")

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