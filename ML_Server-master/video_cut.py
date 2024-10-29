import cv2
import os
import sys
import unicodedata

# 1. 프레임을 일정 간격으로 저장하는 함수
def capture_frames_from_video(input_video_path):
    # 비디오 파일명 추출 및 폴더명 생성
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    video_name = unicodedata.normalize('NFC', video_name.replace(" ", "_"))  # 공백을 밑줄로 변경하고 한글 정규화

    # 수정된 코드 (두 번째 영상일 때도 첫 번째 영상의 폴더에 저장)
    output_folder = os.path.abspath(os.path.join('test', video_name.split('_2')[0]))

    # 출력 폴더가 존재하지 않으면 생성
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"[디렉토리 생성] 출력 폴더를 생성했습니다: {output_folder}")
        else:
            print(f"[디렉토리 존재] 출력 폴더가 이미 존재합니다: {output_folder}")
    except Exception as e:
        print(f"[에러] 출력 폴더 생성 중 오류 발생: {str(e)}")
        sys.exit(1)

    # 비디오 캡처 객체 생성
    print(f"[디버그] 비디오 파일 경로: {input_video_path}")
    video_capture = cv2.VideoCapture(input_video_path)

    if not video_capture.isOpened():
        print(f"[에러] 비디오 파일을 열 수 없습니다: {input_video_path}")
        sys.exit(1)

    # 총 프레임 수 및 FPS 가져오기
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)

    if total_frames == 0 or fps == 0:
        print(f"[에러] 비디오 파일에서 총 프레임 수 또는 FPS를 가져올 수 없습니다. 파일이 손상되었거나 잘못된 파일일 수 있습니다: {input_video_path}")
        sys.exit(1)

    # 총 영상 길이(초)
    video_duration = total_frames / fps
    print(f"[정보] 총 프레임 수: {total_frames}, FPS: {fps}, 영상 길이(초): {video_duration}")

    # 영상의 5분의 1마다 캡처할 프레임 번호 계산
    capture_times = [video_duration * i / 5 for i in range(5)]
    print(f"[정보] 캡처할 시간대 (초): {capture_times}")

    # 프레임 캡처 및 저장
    image_index = 0

    for time_point in capture_times:
        frame_number = int(time_point * fps)

        # 현재 프레임 위치 설정
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        print(f"[프레임 설정] 설정된 프레임 번호: {frame_number}")

        ret, frame = video_capture.read()
        if not ret:
            print(f"[에러] 프레임을 읽어올 수 없습니다. 프레임 번호: {frame_number}")
            continue

        # 프레임 정보 출력
        print(f"[프레임 정보] 프레임 크기: {frame.shape}, 데이터 타입: {frame.dtype}")

        # 이미지 저장 경로 및 시도
        # 동일한 폴더에 여러 영상의 프레임을 저장할 수 있도록 파일 이름에 구분자를 추가
        if "_2.mp4" in input_video_path:
            image_name = os.path.join(output_folder, f'{video_name.split("_2")[0]}_2_{image_index}.jpg')
        else:
            image_name = os.path.join(output_folder, f'{video_name}_{image_index}.jpg')

        try:
            # 경로 인코딩 변경을 시도해 이미지 저장
            image_name = image_name.encode('utf-8').decode('utf-8')
            cv2.imencode('.jpg', frame)[1].tofile(image_name)
            print(f"[저장 성공] 이미지가 저장되었습니다: {image_name}")
        except Exception as e:
            print(f"[에러] 이미지 저장 중 오류 발생: {str(e)}")

        image_index += 1

    # 비디오 캡처 객체 해제
    video_capture.release()
    print(f"[완료] {video_name} 비디오에서 5분의 1마다 이미지를 저장했습니다.")

# 2. 메인 함수: 서버에서 비디오 경로를 전달받아 처리
if __name__ == "__main__":
    # 커맨드 라인에서 비디오 파일 경로를 받아 처리
    print(f"[디버그] 전달된 인자: {sys.argv}")
    if len(sys.argv) != 2:
        print("[에러] 전달된 인자 수가 잘못되었습니다.")
        print("사용법: python video_cut.py <비디오 파일 경로>")
        sys.exit(1)

    input_video_path = sys.argv[1]
    print(f"[디버그] 전달된 비디오 파일 경로: {input_video_path}")

    if not os.path.exists(input_video_path):
        print(f"[에러] 비디오 파일이 존재하지 않습니다: {input_video_path}")
        sys.exit(1)

    capture_frames_from_video(input_video_path)
