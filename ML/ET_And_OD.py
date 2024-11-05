import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import os

# 모델 파일의 절대 경로를 설정하여 경로 문제 해결
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'Models', 'itracing.h5')

# 모델 로드
model = tf.keras.models.load_model(model_path)

def predict_mouse_coordinates(cropped_frame):
    # 아이트래킹을 위한 모델 예측 함수
    cropped_image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
    cropped_image = cropped_image.resize((240, 60))
    input_data = img_to_array(cropped_image) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    predicted_coords = model.predict(input_data)
    predicted_coords = predicted_coords[0] * [1920, 1080]
    return min(max(int(predicted_coords[0]), 0), 1920), min(max(int(predicted_coords[1]), 0), 1080)

def eye_tracking(box_coordinates):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    x, y, w, h = 875, 330, 240, 60  # 아이트래킹 영역 설정을 위한 임시 좌표

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break
        frame = cv2.flip(frame, 1)

        # Face_Recognition.py에서 전달된 얼굴 바운딩 박스 좌표를 사용
        if box_coordinates[0] != -1:
            left, top, right, bottom = box_coordinates
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # 얼굴 바운딩 박스 유지
            cropped_frame = frame[y:y + h, x:x + w]
            predicted_x, predicted_y = predict_mouse_coordinates(cropped_frame)
            cv2.circle(frame, (predicted_x, predicted_y), 5, (0, 0, 255), -1)  # 아이트래킹 예측 좌표 표시
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)       # 아이트래킹 영역 표시

        # TODO: 객체 탐지 코드 (추후 구현 예정)

        cv2.imshow("Eye Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
