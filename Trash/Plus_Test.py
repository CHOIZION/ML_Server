import os
from glob import glob
import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import time

# 얼굴 인식 모델 관련 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# 아이 트래킹 모델 로드
model = tf.keras.models.load_model('itracing.h5')

# 유저 이미지 불러오기 및 얼굴 임베딩 생성
def load_user_encodings(user_folder='videos/example_video'):
    person_folder = user_folder
    if not os.path.exists(person_folder):
        raise ValueError(f"User folder '{person_folder}' not found!")
   
    person_images = glob(os.path.join(person_folder, '*.jpg'))
    if len(person_images) == 0:
        raise ValueError(f"No images found in '{person_folder}'!")
   
    target_encodings = []
    for img_path in person_images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read: {img_path}. Check if the file exists and is a valid image.")
            continue

        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
       
        if len(face_encodings) > 0:
            target_encodings.append(face_encodings[0])
        else:
            print(f"No face found in image: {img_path}")

    if len(target_encodings) == 0:
        raise ValueError(f"No valid face encodings found in '{person_folder}'")

    return target_encodings

# 얼굴을 인식하고 미디어파이프 기반 얼굴 랜드마크 추적
def recognize_and_track_faces(frame, target_encodings, tracking, threshold=0.39):
    face_recognized = False
    if tracking:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                draw_face_landmarks(frame, face_landmarks.landmark, frame.shape[1], frame.shape[0], "User")
            face_recognized = True
        else:
            tracking = False
    else:
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(target_encodings, face_encoding)
            min_distance = min(distances)

            if min_distance < threshold:
                cv2.putText(frame, "User", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                tracking = True
                face_recognized = True
                break

    return tracking, face_recognized

def draw_face_landmarks(frame, landmarks, width, height, user_name):
    landmark_points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks]
    x_min, y_min = min(landmark_points, key=lambda p: p[0])[0], min(landmark_points, key=lambda p: p[1])[1]
    x_max, y_max = max(landmark_points, key=lambda p: p[0])[0], max(landmark_points, key=lambda p: p[1])[1]
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(frame, user_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 아이 트래킹을 위한 모델 예측 함수
def predict_mouse_coordinates(cropped_frame):
    cropped_image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
    cropped_image = cropped_image.resize((240, 60))
    input_data = img_to_array(cropped_image) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    predicted_coords = model.predict(input_data)
    predicted_coords = predicted_coords[0] * [1920, 1080]
    predicted_x, predicted_y = min(max(int(predicted_coords[0]), 0), 1920), min(max(int(predicted_coords[1]), 0), 1080)
    return predicted_x, predicted_y

# 메인 함수: 얼굴 인식과 아이 트래킹 모드를 전환하며 실행
def main():
    target_encodings = load_user_encodings()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    mode = 'face_recognition'
    tracking = False
    face_detected_time = None  # 얼굴 인식 시간을 저장하는 변수

    # 추출할 영역의 좌표 (아이 트래킹 모드에서 사용)
    x, y, w, h = 875, 330, 240, 60

    print(f"웹캠을 통한 얼굴 인식을 시작합니다. 'T'를 눌러 모드를 전환하고 'Q'를 눌러 종료하세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            break

        frame = cv2.flip(frame, 1)

        if mode == 'face_recognition':
            tracking, face_recognized = recognize_and_track_faces(frame, target_encodings, tracking)

            if face_recognized:
                if face_detected_time is None:
                    face_detected_time = time.time()
                elif time.time() - face_detected_time > 5:  # 5초 후에 아이 트래킹 모드로 전환
                    mode = 'eye_tracking'
                    print(f"모드를 {mode}로 전환합니다.")
            else:
                face_detected_time = None  # 얼굴을 놓치면 타이머 리셋

        elif mode == 'eye_tracking':
            cropped_frame = frame[y:y+h, x:x+w]
            predicted_x, predicted_y = predict_mouse_coordinates(cropped_frame)
            cv2.circle(frame, (predicted_x, predicted_y), 5, (0, 0, 255), -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            coord_text = f'Predicted Coordinates: ({predicted_x}, {predicted_y})'
            cv2.putText(frame, coord_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
