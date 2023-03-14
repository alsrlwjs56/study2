# import cv2
# import base64
# from fastapi import FastAPI

# app = FastAPI()

# # Initialize the USB camera
# cap = cv2.VideoCapture(1)

# @app.get("/")
# async def read_root():
#     ret, frame = cap.read()

#     # Encode the frame in base64 format
#     retval, buffer = cv2.imencode('.jpg', frame)
#     jpg_as_text = base64.b64encode(buffer).decode('utf-8')

#     return {"data": jpg_as_text}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import cv2
import base64
from fastapi import FastAPI
import time

app = FastAPI()

cap = cv2.VideoCapture(1)

def calculate_fps():
    num_frames = 120
    start = time.time()
    
    for i in range(0, num_frames):
        ret, frame = cap.read()

    end = time.time()

    seconds = end - start
    fps  = num_frames / seconds

    return fps

@app.get("/")
async def read_root():
    ret, frame = cap.read()

    fps = calculate_fps()

    retval, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    return {"data": jpg_as_text, "fps": fps}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


