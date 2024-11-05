import sys
import os
from multiprocessing import Process, Array
import Face_Recognition
import ET_And_OD

def main(user_name):
    # test 폴더의 절대 경로 설정
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_folder = os.path.join(base_dir, 'test')  # test 폴더의 절대 경로

    try:
        # 사용자 인코딩 로드 (test 폴더의 절대 경로 전달)
        target_encodings = Face_Recognition.load_user_encodings(user_name, test_folder)
    except ValueError as e:
        print(f"Error loading encodings: {e}")
        return

    # 얼굴 바운딩 박스 좌표를 위한 공유 메모리
    box_coordinates = Array('i', [-1, -1, -1, -1])

    # 두 프로세스 병렬 실행
    face_recognition_process = Process(target=Face_Recognition.recognize_faces, args=(target_encodings, box_coordinates, user_name))
    eye_tracking_process = Process(target=ET_And_OD.eye_tracking, args=(box_coordinates,))

    face_recognition_process.start()
    eye_tracking_process.start()

    face_recognition_process.join()
    eye_tracking_process.join()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python main.py <user_name>")
        sys.exit(1)

    user_name = sys.argv[1]
    main(user_name)
