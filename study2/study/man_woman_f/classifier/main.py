import time
import uvicorn
from fastapi import FastAPI, File, UploadFile
import os
from model import GenderPredictionModel
import requests
import json
import cv2
import base64

model_path = os.path.join(os.getcwd(), "./men_women_weights_f.h5")
app = FastAPI()
model = GenderPredictionModel(model_path)

# img = cv2.imread("./man4.jpg")
# _, img_bytes = cv2.imencode('.jpg', img)
# img_base64 = base64.b64encode(img_bytes).decode()
# input_data = {'image': img_base64}
# input_data_json = json.dumps(input_data)
# url = "http://172.16.100.102:8080/json"
# response = requests.post(url, json=input_data_json)
# result_dict = json.loads(response.content)
# result = result_dict["result"]

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    img_base64 = base64.b64encode(contents).decode()
    data = {"image": img_base64, "other_data": "some other data"}
    json_data = json.dumps(data)
    json_data = json_data
    
    return {"filename": file.filename}

@app.post("/upload")
async def predict(file: UploadFile = File(...)):
    
    # Start timing
    start_time = time.time()

    # Read image as bytes
    contents = await file.read()

    # Preprocess the image
    img_array = model.preprocess_image(contents)

    # Predict output
    pred = model.predict_gender(img_array)

    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time

    # Return processing time and reference accuracy
    return {'processing_time': processing_time,
            # 'reference_acc': model.get_reference_accuracy(),
            'prediction': str(pred)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7999)
    
    
