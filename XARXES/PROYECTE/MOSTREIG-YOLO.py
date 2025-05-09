from ultralytics import YOLO
import cv2
import os

model = YOLO("C:/Users/david/UNI/XARXES/PROYECTE/yolov8x.pt")
input_folder = "C:/Users/david/UNI/XARXES/PROYECTE/buenos"

for filename in os.listdir(input_folder):
    path = os.path.join(input_folder, filename)
    frame = cv2.imread(path)

    if frame is None:
        continue

    results = model(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = map(int, box)
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, model.names[cls], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Detecciones", frame)
    if cv2.waitKey(500) == 27:  # Mostrar cada 0.5 segundos, presiona ESC para salir
        break

cv2.destroyAllWindows()
