from ultralytics import YOLO
import cv2
import shutil
import numpy as np
import os

model = YOLO("C:/Users/david/UNI/XARXES/PROYECTE/yolov8x.pt")

input_folder = "C:/Users/david/UNI/XARXES/PROYECTE/Frames_finals (2)/Frames_finals/1"
output_good = "C:/Users/david/UNI/XARXES/PROYECTE/buenos"
output_bad = "C:/Users/david/UNI/XARXES/PROYECTE/malos"
os.makedirs(output_good, exist_ok=True)
os.makedirs(output_bad, exist_ok=True)

MAX_DIST = 40  # umbral para resolución baja

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def calculate_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

for filename in os.listdir(input_folder):
    path = os.path.join(input_folder, filename)
    frame = cv2.imread(path)

    if frame is None:
        continue

    results = model(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    players = [box for box, cls in zip(boxes, classes) if cls == 0]
    balls = [box for box, cls in zip(boxes, classes) if cls == 32]

    if not balls or not players:
        shutil.copy(path, os.path.join(output_bad, filename))
        continue

    pelota = balls[0]
    ball_center = get_center(pelota)

    min_dist = float('inf')
    for player_box in players:
        player_center = get_center(player_box)
        dist = calculate_distance(ball_center, player_center)
        if dist < min_dist:
            min_dist = dist

    if min_dist < MAX_DIST:
        shutil.copy(path, os.path.join(output_good, filename))
    else:
        shutil.copy(path, os.path.join(output_bad, filename))
