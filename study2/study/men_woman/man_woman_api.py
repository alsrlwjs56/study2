import time
import uvicorn
from fastapi import FastAPI, File, UploadFile
from typing import List
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, MaxPool2D, Dropout
from keras.layers import BatchNormalization, Dropout

app = FastAPI()

# Load your model here
model = Sequential()
model.add(Conv2D(128,(3,3),input_shape=(150,150,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Conv2D(256,(3,3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.load_weights('/home/funzin/study/men_woman/men_women_weights_f.h5')
# Define reference accuracy
REFERENCE_ACC = 0.95

# Define endpoint for prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Start timing
    start_time = time.time()

    # Read image as bytes
    contents = await file.read()

    # Load image using PIL
    img = Image.open(io.BytesIO(contents))

    # Preprocess the image
    img = img.resize((150, 150))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict output 
    pred = model.predict(img_array)
    
    if pred >0.5 :
        pred = 'Male'
    else :
        pred = 'Female'
        
    # Convert prediction to JSON
    # output = {'prediction': str(pred)}

    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time

    # Return processing time and reference accuracy
    return {'processing_time': processing_time, 'reference_acc': REFERENCE_ACC, 'prediction': str(pred)}

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)