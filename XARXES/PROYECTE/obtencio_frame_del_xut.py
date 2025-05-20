import os
import json
import cv2

# Funció per convertir el temps del partit a segons
def convert_game_time_to_seconds(game_time):
    try:
        part, time = game_time.split(" - ")
        minutes, seconds = map(int, time.split(":"))
        return minutes * 60 + seconds
    except Exception as e:
        print(f"Error convertint el temps {game_time}: {e}")
        return None

# Funció per extreure els frames d’un vídeo
def extract_shot_frames(json_path, video_path, output_folder, part_number_expected):
    print(f"Processant {video_path} per {json_path}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error al carregar el fitxer JSON {json_path}: {e}")
        return

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"No es pot obrir el vídeo {video_path}")
        return

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    for event in data.get('annotations', []):
        if isinstance(event, dict) and (event.get('label') == 'Shots off target' or event.get('label') == 'Shots on target' or event.get('label') == 'Goal'):
            game_time = event.get('gameTime')
            if game_time is not None:
                part_str = game_time.split(" - ")[0].strip()
                if part_str != str(part_number_expected):
                    continue

                time_in_seconds = convert_game_time_to_seconds(game_time)
                if time_in_seconds is not None:
                    frame_number = int((time_in_seconds + 0.5) * fps)
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = video_capture.read()
                    if ret:
                   
                        match_id = os.path.basename(os.path.dirname(json_path)).replace(" ", "_")
                        video_part = os.path.basename(video_path).split('.')[0]
                        frame_filename = f"{match_id}_{video_part}_shot_{int(time_in_seconds)}.jpg"
                        output_path = os.path.join(output_folder, frame_filename)
                        cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        print(f"Fotograma desat a: {output_path}")
                    else:
                        print(f"No s'ha pogut capturar el fotograma a {frame_number}")
    video_capture.release()

# Funció per processar tots els partits recursivament
def process_all_matches(matches_folder, output_folder):
    print(f"Iniciant recorregut de: {matches_folder}")
    for root, dirs, files in os.walk(matches_folder):
        if 'Labels-v2.json' in files:
            json_path = os.path.join(root, 'Labels-v2.json')
            video_part_1 = os.path.join(root, '1_224p.mkv')
            video_part_2 = os.path.join(root, '2_224p.mkv')

            if not os.path.exists(video_part_1):
                print(f"Vídeo part 1 no trobat a {root}")
                continue
            if not os.path.exists(video_part_2):
                print(f"Vídeo part 2 no trobat a {root}")
                continue

            match_output_folder = output_folder
            os.makedirs(match_output_folder, exist_ok=True)

            print(f"\nprocessant partit a: {root}")
            extract_shot_frames(json_path, video_part_1, match_output_folder, part_number_expected=1)
            extract_shot_frames(json_path, video_part_2, match_output_folder, part_number_expected=2)
        else:
            print(f" No s'ha trobat Labels-v2.json a: {root}")

matches_folder = "C:/Users/Joan/AppData/Local/Programs/Python/Python311/Lib/site-packages/SoccerNet/lligues"
output_folder = "C:/Users/Joan/Desktop/MatCAD/3r/2n Semestre/Xarxes/Projecte/Frames_finals"

process_all_matches(matches_folder, output_folder)
