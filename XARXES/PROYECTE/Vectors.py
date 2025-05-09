from ultralytics import YOLO
import cv2
import numpy as np
import os
import csv

# CONFIGURACIÓN
model = YOLO("C:/Users/david/UNI/XARXES/PROYECTE/yolov8x.pt")
input_folder = "C:/Users/david/UNI/XARXES/PROYECTE/Frames_finals (2)/Frames_finals/1"
output_csv = "C:/Users/david/UNI/XARXES/PROYECTE/features_kmeans_sin_porteria.csv"
DIST_THRESHOLD = 0.05  # ahora expresado como distancia relativa (5% del ancho/alto)

# FUNCIONES
def get_normalized_center(box, img_width, img_height):
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    return (x_center, y_center)

def calculate_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

# CREACIÓN DEL CSV
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "filename",
        "has_ball",
        "min_dist_to_ball",
        "num_players_near_ball"
    ])

    for filename in os.listdir(input_folder):
        path = os.path.join(input_folder, filename)
        frame = cv2.imread(path)
        if frame is None:
            continue

        height, width = frame.shape[:2]

        results = model(frame)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        players = [box for box, cls in zip(boxes, classes) if cls == 0]   # persona
        balls = [box for box, cls in zip(boxes, classes) if cls == 32]   # pelota

        has_ball = int(len(balls) > 0)
        min_dist = -1
        num_players_near_ball = 0

        if balls and players:
            ball_center = get_normalized_center(balls[0], width, height)

            dists = []
            for player in players:
                player_center = get_normalized_center(player, width, height)
                dist = calculate_distance(ball_center, player_center)
                dists.append(dist)
                if dist < DIST_THRESHOLD:
                    num_players_near_ball += 1

            min_dist = min(dists) if dists else -1

        # Escribir características al CSV
        writer.writerow([
            filename,
            has_ball,
            round(min_dist, 4),  # ahora en rango 0–1
            num_players_near_ball
        ])

print(f"\nCSV generado en {output_csv}")
