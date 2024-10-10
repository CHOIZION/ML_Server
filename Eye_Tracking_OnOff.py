# Eye_Tracking_OnOff.py

from fastapi import APIRouter
import subprocess
import logging
import os
import signal

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

router = APIRouter()

# 웹캠 프로세스를 저장할 변수
camera_process = None

@router.get("/On_Off")
def control_eye_tracking(name: str, is_camera_on: bool):
    global camera_process
    logging.info(f"Received request to control eye tracking: name={name}, is_camera_on={is_camera_on}")

    if is_camera_on:  # Boolean 타입의 True/False로 처리
        logging.info(f"Attempting to start eye tracking for {name}...")

        if camera_process is not None:
            logging.warning("Eye tracking is already running. Ignoring the request to start again.")
            return {"status": "already_running", "message": f"Eye tracking is already running for {name}"}

        # 카메라가 켜져 있으면 final_face_verif.py 실행
        try:
            logging.info(f"Executing final_face_verif.py with name={name}")
            camera_process = subprocess.Popen(["python", "final_face_verif.py", name])
            logging.info(f"Eye tracking started successfully for {name}")
            return {"status": "success", "message": f"Eye tracking started for {name}"}
        except Exception as e:
            logging.error(f"Error occurred while starting eye tracking for {name}: {str(e)}")
            return {"status": "error", "message": str(e)}
    else:
        if camera_process is None:
            logging.info("Webcam is already off. No action taken.")
            return {"status": "no_action", "message": f"Webcam is already off for {name}"}

        # 웹캠 프로세스를 종료
        logging.info(f"Stopping eye tracking for {name}...")
        try:
            camera_process.terminate()  # 웹캠 종료
            camera_process = None
            logging.info(f"Eye tracking stopped for {name}.")
            return {"status": "off", "message": f"Eye tracking not started for {name}"}
        except Exception as e:
            logging.error(f"Error occurred while stopping eye tracking for {name}: {str(e)}")
            return {"status": "error", "message": str(e)}
