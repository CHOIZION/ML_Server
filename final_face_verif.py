import os
from glob import glob
import cv2
import numpy as np
import face_recognition
import mediapipe as mp

# 미디어파이프 FaceMesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# 유저 이미지 불러오기 및 얼굴 임베딩 생성
def load_user_encodings(user_name, user_folder='test'):
    person_folder = os.path.join(user_folder, user_name)
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
            target_encodings.append(face_encodings[0])  # 첫 번째 얼굴의 인코딩을 저장
        else:
            print(f"No face found in image: {img_path}")

    if len(target_encodings) == 0:
        raise ValueError(f"No valid face encodings found for {user_name}")

    return target_encodings

# 얼굴을 인식하고 미디어파이프 기반 얼굴 랜드마크 추적
def recognize_and_track_faces(video_capture, target_encodings, user_name, threshold=0.39):
    tracking = False  # 현재 얼굴 추적 상태를 나타냄
    print(f"웹캠을 통한 {user_name} 얼굴 인식을 시작합니다...")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            break

        # 프레임 좌우 반전
        frame = cv2.flip(frame, 1)

        # 프레임 크기
        height, width, _ = frame.shape

        # 현재 추적 중일 때는 미디어파이프를 사용해 랜드마크 추적
        if tracking:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 랜드마크 기반 바운딩 박스 그리기
                    draw_face_landmarks(frame, face_landmarks.landmark, width, height, user_name)
            else:
                tracking = False  # 얼굴을 찾지 못하면 추적 중단

        else:
            # 얼굴 인식 수행
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                distances = face_recognition.face_distance(target_encodings, face_encoding)
                min_distance = min(distances)

                if min_distance < threshold:
                    label = f"{user_name} ({min_distance:.2f})"
                    color = (0, 255, 0)

                    # 얼굴 인식 성공 시, 추적 모드 활성화
                    tracking = True
                    print(f"{user_name} 얼굴이 인식되었습니다. 랜드마크 추적을 시작합니다.")
                    cv2.putText(frame, user_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    break  # 추적 시작을 위해 얼굴 인식 중단
                else:
                    label = f"Unknown ({min_distance:.2f})"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def draw_face_landmarks(frame, landmarks, width, height, user_name):
    landmark_points = []
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        landmark_points.append((x, y))

    x_min = min(point[0] for point in landmark_points)
    y_min = min(point[1] for point in landmark_points)
    x_max = max(point[0] for point in landmark_points)
    y_max = max(point[1] for point in landmark_points)

    # 바운딩 박스 그리기
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # 유저 이름을 바운딩 박스 위에 표시
    cv2.putText(frame, user_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def compare_faces_with_webcam(target_encodings, user_name):
    video_capture = cv2.VideoCapture(0)
    try:
        recognize_and_track_faces(video_capture, target_encodings, user_name)
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

def main():
    target_user = '1'
    target_encodings = load_user_encodings(target_user)
    compare_faces_with_webcam(target_encodings, target_user)

if __name__ == "__main__":
    main()
