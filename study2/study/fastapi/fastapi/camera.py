import cv2

def get_stream_video():
    # camera 정의

    while True:
        cam = cv2.VideoCapture(0)
        
        # 카메라 값 불러오기
        success, frame = cam.read()

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            # frame을 byte로 변경 후 특정 식??으로 변환 후에
            # yield로 하나씩 넘겨준다.
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(frame) + b'\r\n')
            
            
# image.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# cv2 모듈 import
from camera import get_stream_video

# FastAPI객체 생성
app = FastAPI()

# openCV에서 이미지 불러오는 함수
def video_streaming():
    return get_stream_video()

# 스트리밍 경로를 /video 경로로 설정.
@app.get("/video")
def main():
    # StringResponse함수를 return하고,
    # 인자로 OpenCV에서 가져온 "바이트"이미지와 type을 명시
    return StreamingResponse(video_streaming(), media_type="multipart/x-mixed-replace; boundary=frame")