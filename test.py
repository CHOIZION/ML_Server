import os
import urllib
from glob import glob
import urllib.parse
import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import img_to_array
import time
import sys
import torch
import pathlib
import requests
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# YOLO 경로 오류 방지용
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 폰트 경로 지정
FONT_PATH = "ML/Font/Maplestory Light.ttf"

# 얼굴 인식 모델 설정
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# 아이 트래킹 모델 로드
eye_tracking_model = tf.keras.models.load_model('ML/Models/itracing_.h5')

# YOLO 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='ML/Models/yolov5x_train_best.pt')
model.conf = 0.3
model.iou = 0

# YOLO 경로 오류 방지용
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 'light' 감지 모델 로드
model_light = torch.hub.load(
    'ultralytics/yolov5', 'custom', path='ML/Models/light_best_ yolo5.pt', force_reload=True
)
model_light.conf = 0.8  # Confidence threshold for 'light'
model_light.iou = 0.8   # IoU threshold for 'light'

# 'light' 감지 설정
min_width_light = 50
min_height_light = 50
max_width_light = 500
max_height_light = 500
min_aspect_ratio_light = 1.1

# 'fan' 감지 모델 로드
model_fan = torch.hub.load(
    'ultralytics/yolov5', 'custom', path='ML/Models/best_fan_yolo5t.pt', force_reload=True
)
model_fan.conf = 0.8  # Confidence threshold for 'fan'
model_fan.iou = 0.8   # IoU threshold for 'fan'

# 'fan' 감지 설정
min_width_fan = 50
min_height_fan = 50
max_width_fan = 500
max_height_fan = 500
min_aspect_ratio_fan = 1.2

# 리사이즈 비율 및 임계값 설정
resize_rate = 1
iris_x_threshold, iris_y_threshold = 0.15, 0.16  # 눈동자가 중앙에서 얼마나 벗어나야 상태 바뀜으로 인정할 것인지

# 제스처 관련 변수 초기화
prev_iris_status = 'Center'
gesture_start_time = None
blink_start_time = None
received_gestures = []  # 서버로부터 받은 gestureName 리스트

# 유저 이미지 불러오기 및 얼굴 임베딩 생성
def load_user_encodings(user_name):
    person_folder = os.path.join('ML/test', user_name)
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

# 얼굴을 인식하고 미디어파이프 기반 얼굴 랜드마크 추적
def recognize_and_track_faces(frame, target_encodings, tracking, user_name, threshold=0.39):
    face_recognized = False
    x_min, y_min, x_max, y_max = None, None, None, None
    if tracking:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                frame, x_min, y_min, x_max, y_max = draw_face_landmarks(frame, face_landmarks.landmark, frame.shape[1], frame.shape[0], user_name)
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
                face_recognized = True
                x_min, y_min, x_max, y_max = left, top, right, bottom
                break

    return tracking, face_recognized, frame, x_min, y_min, x_max, y_max

def draw_text_with_pillow(img, text, position, font_size=20, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except IOError:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_face_landmarks(frame, landmarks, width, height, user_name):
    landmark_points = [(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks]
    x_min, y_min = min(landmark_points, key=lambda p: p[0])[0], min(landmark_points, key=lambda p: p[1])[1]
    x_max, y_max = max(landmark_points, key=lambda p: p[0])[0], max(landmark_points, key=lambda p: p[1])[1]
    
    # 바운딩 박스에 패딩 추가
    padding_x = int((x_max - x_min) * 0.2)  # 가로 방향으로 20% 패딩
    padding_y = int((y_max - y_min) * 0.2)  # 세로 방향으로 20% 패딩

    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(width, x_max + padding_x)
    y_max = min(height, y_max + padding_y)
    
    # 바운딩 박스 그리기 (동공 위치 계산에는 영향을 주지 않음)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    frame = draw_text_with_pillow(frame, user_name, (x_min, y_min - 30), font_size=20)
    
    return frame, x_min, y_min, x_max, y_max

# YOLO 기반 눈동자 추적 처리
def yolo_process(img):
    yolo_results = model(img)
    df = yolo_results.pandas().xyxy[0]
    obj_list = []
    for i in range(len(df)):
        obj_confi = round(df['confidence'][i], 2)
        obj_name = df['name'][i]
        x_min = int(df['xmin'][i])
        y_min = int(df['ymin'][i])
        x_max = int(df['xmax'][i])
        y_max = int(df['ymax'][i])
        obj_dict = {
            'class': obj_name,
            'confidence': obj_confi,
            'xmin': x_min,
            'ymin': y_min,
            'xmax': x_max,
            'ymax': y_max
        }
        obj_list.append(obj_dict)
    return obj_list

# 서버로 객체 정보를 전송하고 응답을 받는 함수 수정
def send_object_to_server(device_name):
    global received_gestures
    url = 'http://192.168.0.77:8080/et/eyeTracking/getDevice'

    # 객체 이름을 한국어로 매핑
    device_name_mapping = {
        'fan': '선풍기',
        'light': '전등'
    }
    # 매핑된 한국어 이름을 사용하거나, 매핑되지 않은 경우 원래 이름 사용
    device_name_korean = device_name_mapping.get(device_name, device_name)

    data = {'deviceName': device_name_korean}
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    
    try:
        print(f"전송할 device_name: {device_name_korean}")
        response = requests.post(url, data=data, headers=headers, timeout=10)
        response.raise_for_status()

        print(f"서버 응답: {response.json()}")
        response_data = response.json()
        features = response_data.get('features', [])
        received_gestures = [feature.get('gestureName') for feature in features]
        print(f"서버로부터 받은 제스처 리스트: {received_gestures}")

    except requests.exceptions.RequestException as e:
        print(f"서버로 객체를 전송하는 중 오류 발생: {str(e)}")

# 서버로 제스처를 전송하는 함수 수정 (POST 요청으로 변경)
def send_gesture_to_server(gesture_name):
    try:
        url = 'http://192.168.0.77:8080/et/eyeTracking/getGesture'  # 서버 주소에 맞게 변경하세요
        data = {'gestureName': gesture_name}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, json=data, headers=headers, timeout=10)
        if response.status_code == 200:
            print(f"제스처 '{gesture_name}'를 서버로 전송했습니다. 응답: {response.json()}")
        else:
            print(f"제스처를 서버로 전송하는 데 실패했습니다. 상태 코드: {response.status_code}")
    except Exception as e:
        print(f"제스처를 서버로 전송하는 중 오류 발생: {str(e)}")

# 아이 트래킹을 위한 마우스 좌표 예측 함수
def predict_mouse_coordinates(cropped_frame):
    cropped_image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
    cropped_image = cropped_image.resize((240, 60))
    input_data = img_to_array(cropped_image) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    predicted_coords = eye_tracking_model.predict(input_data)
    predicted_coords = predicted_coords[0] * [1920, 1080]
    predicted_x, predicted_y = min(max(int(predicted_coords[0]), 0), 1920), min(max(int(predicted_coords[1]), 0), 1080)
    return predicted_x, predicted_y

def main():
    global prev_iris_status, gesture_start_time, blink_start_time, received_gestures
    if len(sys.argv) < 2:
        print("사용법: python final_face_verif.py <user_name>")
        sys.exit(1)
    user_name = sys.argv[1]
    user_name = urllib.parse.unquote(user_name)
    target_encodings = load_user_encodings(user_name)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    mode = 'face_recognition'
    tracking = False
    face_detected_time = None
    eye_tracking_start_time = None
    iris_status = 'Center'  

    x, y, w, h = 875, 330, 240, 60

    current_object = None

    print(f"웹캠을 통한 얼굴 인식을 시작합니다. 'Q'를 눌러 종료하세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            break

        frame = cv2.flip(frame, 1)

        if mode == 'face_recognition':
            tracking, face_recognized, frame, x_min, y_min, x_max, y_max = recognize_and_track_faces(frame, target_encodings, tracking, user_name)

            if face_recognized:
                if face_detected_time is None:
                    face_detected_time = time.time()
                elif time.time() - face_detected_time > 5:
                    mode = 'eye_tracking'
                    eye_tracking_start_time = time.time()
                    print(f"모드를 {mode}로 전환합니다.")
            else:
                face_detected_time = None

        elif mode == 'eye_tracking':
            tracking, face_recognized, frame, x_min, y_min, x_max, y_max = recognize_and_track_faces(frame, target_encodings, tracking, user_name)
            
            if not face_recognized:
                print("얼굴 인식을 놓쳤습니다. 얼굴 인식 모드로 돌아갑니다.")
                mode = 'face_recognition'
                face_detected_time = None
                continue

            if x_min <= x <= x + w <= x_max and y_min <= y <= y + h <= y_max:
                cropped_frame = frame[y:y + h, x:x + w]
                predicted_x, predicted_y = predict_mouse_coordinates(cropped_frame)
                cv2.circle(frame, (predicted_x, predicted_y), 5, (0, 0, 255), -1)
                coord_text = f'Predicted Coordinates: ({predicted_x}, {predicted_y})'
                frame = draw_text_with_pillow(frame, coord_text, (10, 30), font_size=20)
                
                # 객체 감지 실행
                # 'light' 모델로 감지 실행
                results_light = model_light(frame)
                # 'light'에 대한 감지 처리
                light_detections = []
                for det in results_light.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = det[:6]
                    label = model_light.names[int(cls)]
                    if label == 'light':
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = height / width
                        if (
                            min_width_light <= width <= max_width_light and
                            min_height_light <= height <= max_height_light and
                            aspect_ratio >= min_aspect_ratio_light
                        ):
                            cv2.rectangle(
                                frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 2
                            )
                            cv2.putText(
                                frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                            )
                            light_detections.append({
                                'label': label,
                                'bbox': (int(x1), int(y1), int(x2), int(y2))
                            })
                # 'fan' 모델로 감지 실행
                results_fan = model_fan(frame)
                # 'fan'에 대한 감지 처리
                fan_detections = []
                for det in results_fan.xyxy[0]:
                    x1, y1, x2, y2, conf, cls = det[:6]
                    label = model_fan.names[int(cls)]
                    if label == 'fan':
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = height / width
                        if (
                            min_width_fan <= width <= max_width_fan and
                            min_height_fan <= height <= max_height_fan and
                            aspect_ratio >= min_aspect_ratio_fan
                        ):
                            cv2.rectangle(
                                frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                            )
                            cv2.putText(
                                frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                            )
                            fan_detections.append({
                                'label': label,
                                'bbox': (int(x1), int(y1), int(x2), int(y2))
                            })
                # 감지 결과 합치기
                detections = light_detections + fan_detections
                
                # 예측된 시선 지점이 어떤 객체 내에 있는지 확인
                gaze_in_object = False
                for detection in detections:
                    label = detection['label']
                    x1, y1, x2, y2 = detection['bbox']
                    if x1 <= predicted_x <= x2 and y1 <= predicted_y <= y2:
                        gaze_in_object = True
                        if current_object is None or current_object['label'] != label:
                            current_object = {'label': label, 'start_time': time.time()}
                            print(f"객체 '{label}'에 시선 시작")
                        else:
                            elapsed_time = time.time() - current_object['start_time']
                            if elapsed_time > 0.5:
                                print(f"객체 '{label}'를 0.5초 이상 바라봄. 서버로 전송 중...")
                                # 객체를 서버로 전송
                                send_object_to_server(label)
                                # 응답을 받은 후 'yolo_tracking' 모드로 전환
                                mode = 'yolo_tracking'
                                print(f"모드를 {mode}로 전환합니다.")
                                # 시선 추적 변수 초기화
                                current_object = None
                                break
                        break  # 시선이 객체 위에 있으므로 더 이상 확인하지 않음
                if not gaze_in_object:
                    current_object = None

            else:
                current_object = None

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
        elif mode == 'yolo_tracking':
            # 'yolo_tracking' 모드에서는 얼굴 인식 초록색 바운딩 박스와 YOLO 트래킹만 실행
            tracking, face_recognized, frame, x_min, y_min, x_max, y_max = recognize_and_track_faces(
                frame, target_encodings, tracking, user_name
            )

            if not face_recognized:
                print("얼굴 인식을 놓쳤습니다. 얼굴 인식 모드로 돌아갑니다.")
                mode = 'face_recognition'
                face_detected_time = None
                continue

            # 전체 프레임에 대해 YOLO 처리
            results = yolo_process(frame)
            eye_list = []
            iris_list = []

            # 검출된 객체 중 얼굴 바운딩 박스 내에 있는 것만 사용
            for result in results:
                xmin_resize = int(result['xmin'] / resize_rate)
                ymin_resize = int(result['ymin'] / resize_rate)
                xmax_resize = int(result['xmax'] / resize_rate)
                ymax_resize = int(result['ymax'] / resize_rate)

                # 객체가 얼굴 바운딩 박스 내에 있는지 확인
                if xmin_resize >= x_min and xmax_resize <= x_max and ymin_resize >= y_min and ymax_resize <= y_max:
                    if result['class'] == 'eyes':
                        eye_list.append({
                            'class': result['class'],
                            'confidence': result['confidence'],
                            'xmin': xmin_resize,
                            'ymin': ymin_resize,
                            'xmax': xmax_resize,
                            'ymax': ymax_resize
                        })
                        # 눈 바운딩 박스는 그리지 않음
                    elif result['class'] == 'iris':
                        iris_list.append({
                            'class': result['class'],
                            'confidence': result['confidence'],
                            'xmin': xmin_resize,
                            'ymin': ymin_resize,
                            'xmax': xmax_resize,
                            'ymax': ymax_resize
                        })
                        # 동공을 원으로 그리기
                        x_center = int((xmin_resize + xmax_resize) / 2)
                        y_center = int((ymin_resize + ymax_resize) / 2)
                        x_length = xmax_resize - xmin_resize
                        y_length = ymax_resize - ymin_resize
                        circle_r = int((x_length + y_length) / 4)
                        cv2.circle(frame, (x_center, y_center), circle_r, (255, 255, 255), 1)

            # 왼쪽 파트와 오른쪽 파트를 나눔
            if len(eye_list) == 2 and len(iris_list) == 2:
                left_part = []
                right_part = []
                if eye_list[0]['xmin'] > eye_list[1]['xmin']:
                    right_part.append(eye_list[0])
                    left_part.append(eye_list[1])
                else:
                    right_part.append(eye_list[1])
                    left_part.append(eye_list[0])
                if iris_list[0]['xmin'] > iris_list[1]['xmin']:
                    right_part.append(iris_list[0])
                    left_part.append(iris_list[1])
                else:
                    right_part.append(iris_list[1])
                    left_part.append(iris_list[0])

                left_x_iris_center = (left_part[1]['xmin'] + left_part[1]['xmax']) / 2
                left_x_per = (left_x_iris_center - left_part[0]['xmin']) / (left_part[0]['xmax'] - left_part[0]['xmin'])
                left_y_iris_center = (left_part[1]['ymin'] + left_part[1]['ymax']) / 2
                left_y_per = (left_y_iris_center - left_part[0]['ymin']) / (left_part[0]['ymax'] - left_part[0]['ymin'])

                right_x_iris_center = (right_part[1]['xmin'] + right_part[1]['xmax']) / 2
                right_x_per = (right_x_iris_center - right_part[0]['xmin']) / (right_part[0]['xmax'] - right_part[0]['xmin'])
                right_y_iris_center = (right_part[1]['ymin'] + right_part[1]['ymax']) / 2
                right_y_per = (right_y_iris_center - right_part[0]['ymin']) / (right_part[0]['ymax'] - right_part[0]['ymin'])

                avr_x_iris_per = (left_x_per + right_x_per) / 2
                avr_y_iris_per = (left_y_per + right_y_per) / 2

                if avr_x_iris_per < (0.5 - iris_x_threshold):
                    iris_status = 'Right'  # 좌우반전으로 수정
                elif avr_x_iris_per > (0.5 + iris_x_threshold):
                    iris_status = 'Left'  # 좌우반전으로 수정
                elif avr_y_iris_per < (0.5 - iris_y_threshold):
                    iris_status = 'Up'
                else:
                    iris_status = 'Center'
                # 텍스트를 프레임 위에 그리기
                cv2.putText(frame, f'Iris Direction: {iris_status}', (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (30, 30, 30), 2)
            else:
                iris_status = 'Blink'
                cv2.putText(frame, f'Iris Direction: {iris_status}', (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (30, 30, 30), 2)

            # 제스처 감지 및 서버로 전송
            if iris_status in ['Right', 'Left', 'Up']:
                if iris_status == prev_iris_status:
                    if gesture_start_time is None:
                        gesture_start_time = time.time()
                    else:
                        elapsed_time = time.time() - gesture_start_time
                        if elapsed_time >= 3:
                            gesture_name = iris_status
                            if gesture_name in received_gestures:
                                print(f"제스처 '{gesture_name}'를 {elapsed_time:.1f}초 동안 유지했습니다. 서버로 전송합니다.")
                                send_gesture_to_server(gesture_name)
                            else:
                                print(f"제스처 '{gesture_name}'는 서버로부터 받은 제스처 목록에 없습니다.")
                            gesture_start_time = None  # 타이머 리셋하여 중복 전송 방지
                else:
                    gesture_start_time = time.time()
            elif iris_status == 'Center':
                gesture_start_time = None  # Center로 돌아오면 타이머 리셋
            else:
                gesture_start_time = None  # 다른 상태일 경우 타이머 리셋

            prev_iris_status = iris_status

            # Blink 감지
            if iris_status == 'Blink':
                if blink_start_time is None:
                    blink_start_time = time.time()
                else:
                    blink_elapsed = time.time() - blink_start_time
                    if blink_elapsed >= 3:
                        if 'Blink' in received_gestures:
                            print("Blink 제스처 감지됨. 서버로 전송합니다.")
                            send_gesture_to_server('Blink')
                        else:
                            print("Blink 제스처는 서버로부터 받은 제스처 목록에 없습니다.")
                        blink_start_time = None
            else:
                blink_start_time = None

        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
