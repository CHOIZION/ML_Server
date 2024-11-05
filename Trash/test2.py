import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import face_recognition
import mediapipe as mp
import time
from glob import glob

# 모델 로드 (itracing 모델)
model = tf.keras.models.load_model('itracing.h5')

# 미디어파이프 FaceMesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# itracing 모델 예측 함수
def predict_mouse_coordinates(cropped_frame):
    if cropped_frame is None or cropped_frame.size == 0:
        return None, None

    cropped_image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
    cropped_image = cropped_image.resize((240, 60))
    input_data = img_to_array(cropped_image) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    predicted_coords = model.predict(input_data)
    predicted_coords = predicted_coords[0] * [640, 480]
    predicted_x, predicted_y = predicted_coords
    predicted_x = min(max(int(predicted_x), 0), 640)
    predicted_y = min(max(int(predicted_y), 0), 480)

    return predicted_x, predicted_y

# Itracing 함수
def itracing(frame):
    x, y, w, h = 200, 150, 240, 60  # 관심영역 설정
    cropped_frame = frame[y:y+h, x:x+w]
    predicted_x, predicted_y = predict_mouse_coordinates(cropped_frame)

    if predicted_x is not None and predicted_y is not None:
        cv2.circle(frame, (predicted_x, predicted_y), 5, (0, 0, 255), -1)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return frame

# 얼굴 인식 및 추적 함수
def face_recognition_mode(frame, target_encodings, user_name, threshold=0.39):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    if len(face_encodings) == 0:  # 얼굴 인식에 실패한 경우
        return frame

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        distances = face_recognition.face_distance(target_encodings, face_encoding)
        if len(distances) == 0:
            continue  # 인식된 얼굴이 없는 경우 건너뜀

        min_distance = min(distances)

        if min_distance < threshold:
            label = f"{user_name} ({min_distance:.2f})"
            color = (0, 255, 0)
        else:
            label = f"Unknown ({min_distance:.2f})"
            color = (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame

# 메인 함수
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 유저 이미지 경로를 하드코딩
    user_name = "example_video"
    user_encodings = load_user_encodings(user_name, user_folder="test")

    mode_switch_time = time.time()
    current_mode = "face_recognition"  # 처음엔 얼굴 인식 모드로 시작

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전

        # 5초마다 모드 전환
        if time.time() - mode_switch_time > 5:
            current_mode = "itracing" if current_mode == "face_recognition" else "face_recognition"
            mode_switch_time = time.time()

        # 모드에 따라 다른 처리
        if current_mode == "face_recognition":
            frame = face_recognition_mode(frame, user_encodings, user_name)
        else:
            frame = itracing(frame)

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 유저 인코딩 불러오기 (하드코딩된 경로)
def load_user_encodings(user_name, user_folder='test'):
    person_folder = os.path.join(user_folder, user_name)
    person_images = glob(os.path.join(person_folder, '*.jpg'))  # 하드코딩된 경로에서 이미지 검색
    target_encodings = []
    for img_path in person_images:
        img = cv2.imread(img_path)
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        if len(face_encodings) > 0:
            target_encodings.append(face_encodings[0])
    return target_encodings

# 프로그램 실행
if __name__ == "__main__":
    main()
