
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# 모델 로드
model = tf.keras.models.load_model('itracing.h5')

# 모델 예측 함수
def predict_mouse_coordinates(cropped_frame):
    # 입력 데이터 전처리
    cropped_image = Image.fromarray(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
    cropped_image = cropped_image.resize((240, 60))
    input_data = img_to_array(cropped_image) / 255.0  # 정규화 (0~1 범위)
    input_data = np.expand_dims(input_data, axis=0)  # 배치 차원 추가

    # 모델 예측
    predicted_coords = model.predict(input_data)

    # 예측 좌표 역정규화 (0~1 범위를 1920x1080으로 복구)
    predicted_coords = predicted_coords[0] * [1920, 1080]
    predicted_x, predicted_y = predicted_coords

    # 0~1920, 0~1080 범위로 제한
    predicted_x = min(max(int(predicted_x), 0), 1920)
    predicted_y = min(max(int(predicted_y), 0), 1080)

    return predicted_x, predicted_y

# 웹캠 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 추출할 영역의 좌표와 크기
x, y, w, h = 875, 330, 240, 60

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 좌우 반전
    frame = cv2.flip(frame, 1)

    # 관심영역 추출
    cropped_frame = frame[y:y+h, x:x+w]

    # 모델로 예측
    predicted_x, predicted_y = predict_mouse_coordinates(cropped_frame)

    # 화면에 예측된 좌표 표시 (점의 형태로)
    cv2.circle(frame, (predicted_x, predicted_y), 5, (0, 0, 255), -1)

    # 추출된 영역에 바운딩 박스 표시
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 왼쪽 상단에 예측된 마우스 좌표 출력
    coord_text = f'Predicted Coordinates: ({predicted_x}, {predicted_y})'
    cv2.putText(frame, coord_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 화면 표시
    cv2.imshow('Webcam', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
