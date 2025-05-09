from ultralytics import YOLO
import cv2
import numpy as np
import os
import csv

# CONFIGURACIÓN
model = YOLO("C:/Users/david/UNI/XARXES/PROYECTE/yolov8x.pt")
input_folder = "C:/Users/david/UNI/XARXES/PROYECTE/Frames_finals (2)/Frames_finals/1"
output_csv = "C:/Users/david/UNI/XARXES/PROYECTE/features_kmeans_mejorado_sin_near.csv"
DIST_THRESHOLD = 0.05  # en proporción

# FUNCIONES
def get_normalized_center(box, img_width, img_height):
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    return (x_center, y_center)

def calculate_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def normalize_box(box, img_width, img_height):
    x1, y1, x2, y2 = box
    x = x1 / img_width
    y = y1 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return x, y, w, h

# CREACIÓN DEL CSV
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "filename",
        "ball_confidence",
        "ball_x", "ball_y", "ball_w", "ball_h",
        "min_dist_to_ball"
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
        confs = results.boxes.conf.cpu().numpy()

        players = [box for box, cls in zip(boxes, classes) if cls == 0]
        balls_info = [(box, conf) for box, cls, conf in zip(boxes, classes, confs) if cls == 32]

        # Inicializar valores por defecto
        ball_conf = 0.0
        ball_x = ball_y = ball_w = ball_h = -1
        min_dist = -1

        if balls_info:
            ball_box, ball_conf = balls_info[0]  # usar la más confiable
            ball_center = get_normalized_center(ball_box, width, height)
            ball_x, ball_y, ball_w, ball_h = normalize_box(ball_box, width, height)

            if players:
                dists = []
                for player in players:
                    player_center = get_normalized_center(player, width, height)
                    dist = calculate_distance(ball_center, player_center)
                    dists.append(dist)
                min_dist = min(dists) if dists else -1

        # Escribir vector
        writer.writerow([
            filename,
            round(ball_conf, 4),
            round(ball_x, 4), round(ball_y, 4),
            round(ball_w, 4), round(ball_h, 4),
            round(min_dist, 4)
        ])

print(f"\nCSV generado en {output_csv}")
