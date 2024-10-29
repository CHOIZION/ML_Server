import cv2
import time

def process_code1(frame):
    # 첫 번째 코드: 화면에 텍스트 출력
    cv2.putText(frame, "Code 1 Running", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def process_code2(frame):
    # 두 번째 코드: 화면에 다른 텍스트 출력
    cv2.putText(frame, "Code 2 Running", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cap = cv2.VideoCapture(0)  # 웹캠 열기
switch_time = time.time()  # 전환 시간을 기록
current_code = 1  # 처음에 실행할 코드 선택

while True:
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        break

    # 5초마다 코드 전환
    if time.time() - switch_time > 5:
        current_code = 1 if current_code == 2 else 2  # 코드 전환
        switch_time = time.time()

    if current_code == 1:
        process_code1(frame)
    else:
        process_code2(frame)

    cv2.imshow('Webcam', frame)  # 화면 출력

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
