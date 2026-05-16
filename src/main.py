import cv2
import os

# Definicija poti za lokalno strukturo
DATA_DIR = "data"
OUTPUT_DIR = "output/videos"

# Zagotovimo obstoj izhodne mape
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_video(input_path, output_path):
    """Odpre vhodni video in prekopira okvirje v izhodni video."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Napaka: Ni mogoce odpreti videa {input_path}")
        return

    # Pridobivanje lastnosti videa
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # V primeru poskodovanih metapodatkov
    if fps == 0 or fps != fps:
        fps = 30.0

    # Inicializacija zapisovalca (mp4 kodek)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Zacetek obdelave videa: {input_path}")
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detekcija roke
        
        # Zapis okvirja v novo datoteko
        out.write(frame)
        frame_count += 1

    # Zapiranje
    cap.release()
    out.release()
    print(f"Konec obdelave. Prebranih okvirjev: {frame_count}")
    print(f"Video shranjen v: {output_path}")

if __name__ == "__main__":
    # Testni zagon z lokalnim videom
    test_input = os.path.join(DATA_DIR, "test.mp4")
    test_output = os.path.join(OUTPUT_DIR, "test_izhod.mp4")

    if os.path.exists(test_input):
        process_video(test_input, test_output)
    else:
        print(f"Za testiranje dodajte video z imenom 'test.mp4' v mapo '{DATA_DIR}'.")