import cv2
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Poti do videov in modelov
DATA_DIR = "data"
OUTPUT_DIR = "output/videos"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

os.makedirs(OUTPUT_DIR, exist_ok=True)
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
                for connection in HAND_CONNECTIONS:
                    p1 = hand_landmarks[connection[0]]
                    p2 = hand_landmarks[connection[1]]
                    cv2.line(frame, (int(p1.x * width), int(p1.y * height)), 
                                    (int(p2.x * width), int(p2.y * height)), (255, 255, 255), 2)
                
                for landmark in hand_landmarks:
                    cv2.circle(frame, (int(landmark.x * width), int(landmark.y * height)), 4, (0, 0, 0), -1)

        out.write(frame)
        frame_count += 1

    # Zapre video
    cap.release()
    out.release()
    print(f"Konec obdelave. Prebranih okvirjev: {frame_count}")
    print(f"Video shranjen v: {output_path}")

# Main
if __name__ == "__main__":
    test_input = os.path.join(DATA_DIR, "test.mp4")
    test_output = os.path.join(OUTPUT_DIR, "test_izhod.mp4")

    if os.path.exists(test_input):
        process_video(test_input, test_output)
    else:
        print(f"Za testiranje dodajte video z imenom 'test.mp4' v mapo '{DATA_DIR}'.")