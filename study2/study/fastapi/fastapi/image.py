from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}

# import requests

# url = "http://localhost:8000/uploadfile/"
# files = {"file": open("/home/funzin/study/test_fastapi/man2.jpg", "rb")}
# response = requests.post(url, files=files)

# print(response.json())

# curl -X POST -F "file=@/path/to/image.jpg" http://localhost:8000/uploadfile/

# {"filename":"image.jpg"} 

# main.py

# 라이브러리 import
# StreamingResponse를 가져와야함


