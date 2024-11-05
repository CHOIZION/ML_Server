import torch
import cv2
import pathlib

# Adjust pathlib for compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load the 'light' detection model
model_light = torch.hub.load(
    'ultralytics/yolov5', 'custom', path='ML/Models/light_best_yolo5.pt', force_reload=True
)
model_light.conf = 0.85  # Confidence threshold for 'light'
model_light.iou = 0.85   # IoU threshold for 'light'

# Detection settings for 'light'
min_width_light = 50
min_height_light = 50
max_width_light = 500
max_height_light = 500
min_aspect_ratio_light = 1.1

# Load the 'fan' detection model
model_fan = torch.hub.load(
    'ultralytics/yolov5', 'custom', path='ML/Models/best_fan_yolo5t.pt', force_reload=True
)
model_fan.conf = 0.88  # Confidence threshold for 'fan'
model_fan.iou = 0.88   # IoU threshold for 'fan'

# Detection settings for 'fan'
min_width_fan = 50
min_height_fan = 50
max_width_fan = 500
max_height_fan = 500
min_aspect_ratio_fan = 1.2

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect the highest confidence object for 'light'
    results_light = model_light(frame)
    max_conf_light = None
    best_det_light = None
    for det in results_light.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det[:6]
        label = model_light.names[int(cls)]
        if label == 'light':
            width, height = x2 - x1, y2 - y1
            aspect_ratio = height / width
            if (
                min_width_light <= width <= max_width_light and
                min_height_light <= height <= max_height_light and
                aspect_ratio >= min_aspect_ratio_light
            ):
                if max_conf_light is None or conf > max_conf_light:
                    max_conf_light = conf
                    best_det_light = (x1, y1, x2, y2, conf, label)

    # Draw the best detection for 'light'
    if best_det_light:
        x1, y1, x2, y2, conf, label = best_det_light
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2)
        cv2.putText(
            frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )

    # Detect the highest confidence object for 'fan'
    results_fan = model_fan(frame)
    max_conf_fan = None
    best_det_fan = None
    for det in results_fan.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det[:6]
        label = model_fan.names[int(cls)]
        if label == 'fan':
            width, height = x2 - x1, y2 - y1
            aspect_ratio = height / width
            if (
                min_width_fan <= width <= max_width_fan and
                min_height_fan <= height <= max_height_fan and
                aspect_ratio >= min_aspect_ratio_fan
            ):
                if max_conf_fan is None or conf > max_conf_fan:
                    max_conf_fan = conf
                    best_det_fan = (x1, y1, x2, y2, conf, label)

    # Draw the best detection for 'fan'
    if best_det_fan:
        x1, y1, x2, y2, conf, label = best_det_fan
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    # Display the resulting frame
    cv2.imshow('Webcam Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
