import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mediapipe as mp
import os
from glob import glob

FONT_PATH = "Font/Maplestory Light.ttf"

# Mediapipe 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def load_user_encodings(user_name, base_folder=None):
    # base_folder가 없을 경우, 상위 디렉토리에서 'test' 폴더를 찾도록 설정
    if base_folder is None:
        base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'test'))
    else:
        base_folder = os.path.abspath(base_folder)
    
    person_folder = os.path.join(base_folder, user_name)
    if not os.path.exists(person_folder):
        raise ValueError(f"User folder '{person_folder}' not found!")

    person_images = glob(os.path.join(person_folder, '*.jpg'))
    if len(person_images) == 0:
        raise ValueError(f"No images found in '{person_folder}'!")

    target_encodings = []
    for img_path in person_images:
        try:
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                print(f"Failed to read: {img_path}. Check if the file exists and is a valid image.")
                continue

            face_locations = face_recognition.face_locations(img)
            face_encodings = face_recognition.face_encodings(img, face_locations)

            if face_encodings:
                target_encodings.append(face_encodings[0])
            else:
                print(f"No face found in image: {img_path}")
        except Exception as e:
            print(f"Error loading image '{img_path}': {str(e)}")

    if not target_encodings:
        raise ValueError(f"No valid face encodings found in '{person_folder}'")

    return target_encodings

def recognize_faces(target_encodings, box_coordinates, user_name, threshold=0.39):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    tracking = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        if tracking:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    frame = draw_face_landmarks(frame, face_landmarks.landmark, frame.shape[1], frame.shape[0], user_name)
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
                    frame = draw_text_with_pillow(frame, user_name, (left, top - 30), font_size=20)
                    tracking = True
                    box_coordinates[:] = [left, top, right, bottom]  # 얼굴 바운딩 박스 좌표 공유 메모리에 기록
                    break

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def draw_text_with_pillow(img, text, position, font_size=20, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        print("폰트를 찾을 수 없습니다. 기본 폰트로 대체합니다.")
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_face_landmarks(frame, landmarks, width, height, user_name):
    landmark_points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks]
    x_min, y_min = min(landmark_points, key=lambda p: p[0])[0], min(landmark_points, key=lambda p: p[1])[1]
    x_max, y_max = max(landmark_points, key=lambda p: p[0])[0], max(landmark_points, key=lambda p: p[1])[1]
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return draw_text_with_pillow(frame, user_name, (x_min, y_min - 30), font_size=20)
