import os
import base64
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
import uvicorn
from add_video import router as add_video_router  # 새로운 파일에서 라우터 가져오기

app = FastAPI()

video_chunks: Dict[str, List[str]] = {}

class VideoChunkRequest(BaseModel):
    Name: str
    videoChunk: str
    chunkIndex: int
    totalChunks: int

def add_base64_padding(base64_str: str) -> str:
    missing_padding = len(base64_str) % 4
    if missing_padding:
        base64_str += '=' * (4 - missing_padding)
    return base64_str

def decode_base64_to_video(base64_str: str, output_file: str):
    try:
        print(f"[디코딩] 비디오 디코딩 시작. 파일 경로: {output_file}")
        base64_str = add_base64_padding(base64_str)
        video_data = base64.b64decode(base64_str)
        with open(output_file, 'wb') as video_file:
            video_file.write(video_data)
        print(f"[디코딩] 비디오 디코딩 완료. 파일이 저장되었습니다: {output_file}")
        return output_file
    except Exception as e:
        print(f"[에러] Base64 디코딩 중 오류 발생: {str(e)}")
        raise ValueError(f"Base64 디코딩 중 오류 발생: {str(e)}")

@app.post("/upload-video/")
async def upload_video(request: VideoChunkRequest):
    try:
        Name = request.Name
        videoChunk = request.videoChunk
        chunkIndex = request.chunkIndex
        totalChunks = request.totalChunks

        print(f"[요청 수신] Name: {Name}, chunkIndex: {chunkIndex}, totalChunks: {totalChunks}")

        if chunkIndex < 0 or chunkIndex > totalChunks:
            print(f"[에러] 유효하지 않은 청크 인덱스: chunkIndex={chunkIndex}, totalChunks={totalChunks}")
            return {"status": "실패", "error": "유효하지 않은 청크 인덱스"}

        print(f"[디코딩] videoChunk 길이: {len(videoChunk)}")

        # 새로운 Name의 청크 리스트 초기화
        if Name not in video_chunks:
            print(f"[초기화] 새로운 Name 생성 - Name: {Name}, 총 청크 수: {totalChunks}")
            video_chunks[Name] = [None] * (totalChunks + 1)  # 인덱스 1부터 사용하기 위해 크기를 totalChunks + 1로 설정

        # 중복된 청크 처리 방지
        if video_chunks[Name][chunkIndex] is not None:
            print(f"[중복] 청크 {chunkIndex}는 이미 처리되었습니다 - Name: {Name}")
            return {"status": f"청크 {chunkIndex}는 이미 처리되었습니다."}

        # 청크 저장
        video_chunks[Name][chunkIndex] = videoChunk
        saved_chunks = len([chunk for chunk in video_chunks[Name][1:] if chunk is not None])
        print(f"[저장] 청크 {chunkIndex} 저장 완료 - 현재 저장된 청크 수: {saved_chunks}/{totalChunks}")

        # 모든 청크 수신 완료 시 처리
        if saved_chunks == totalChunks:
            print(f"[완료] 모든 청크 수신 완료 - Name: {Name}. 비디오 합치는 중...")
            # 인덱스 0번부터 시작하여 모든 청크를 합칩니다.
            full_video_base64 = ''.join(video_chunks[Name])
            full_video_base64 = add_base64_padding(full_video_base64)
            print(f"[합치기] 모든 청크 합치기 완료. 합쳐진 Base64 길이: {len(full_video_base64)}")

            directory_path = f"videos/{Name}"
            os.makedirs(directory_path, exist_ok=True)
            video_save_path = f"{directory_path}/{Name}.mp4"
            print(f"[저장 경로] 디렉토리 생성 완료. 저장 경로: {video_save_path}")

            decoded_video_path = decode_base64_to_video(full_video_base64, video_save_path)

            print(f"[정리] 청크 데이터 삭제 - Name: {Name}")
            del video_chunks[Name]

            # video_cut.py 호출 (쉘 명령으로 호출, 유니코드 경로 처리)
            try:
                print(f"[프레임 추출] video_cut.py 실행 중 - 파일 경로: {decoded_video_path}")
                subprocess.run(['python', 'video_cut.py', decoded_video_path], check=True)

                print(f"[프레임 추출] video_cut.py 실행 완료")
            except subprocess.CalledProcessError as e:
                print(f"[에러] video_cut.py 실행 중 오류 발생: {str(e)}")
                return {"status": "실패", "error": f"프레임 추출 중 오류 발생: {str(e)}"}

            return {"status": "모든 조각 처리 및 영상 저장 완료", "video_path": decoded_video_path}

        print(f"[진행 상황] 청크 저장 중 - 현재 청크 수: {saved_chunks}/{totalChunks}")
        return {"status": f"{chunkIndex}/{totalChunks} 조각 저장 완료"}

    except Exception as e:
        print(f"[에러] 처리 중 오류 발생: {str(e)}")
        return {"status": "실패", "error": str(e)}

# 새로운 `/add-video` 엔드포인트를 FastAPI 애플리케이션에 추가
app.include_router(add_video_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
