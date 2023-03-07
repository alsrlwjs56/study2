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
import os

model_path = os.path.join(os.getcwd(), "/home/funzin/study/men_woman/men_women_weights_f.h5")

class GenderPredictionModel:
    def __init__(self, model_path):
        # Load your model here
        self.model = Sequential()
        self.model.add(Conv2D(128,(3,3),input_shape=(150,150,3),activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D())
        self.model.add(Conv2D(256,(3,3),activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPool2D())
        self.model.add(Flatten())
        self.model.add(Dense(128,activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(256,activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1,activation='sigmoid'))
        self.model.load_weights(model_path)

        # Define reference accuracy
        self.reference_acc = 0.95

    def preprocess_image(self, contents):
        # Load image using PIL
        img = Image.open(io.BytesIO(contents))

        # Preprocess the image
        img = img.resize((150, 150))
        img_array = np.array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict_gender(self, img_array):
        # Predict output
        pred = self.model.predict(img_array)

        if pred > 0.5:
            return 'Male'
        else:
            return 'Female'

    def get_reference_accuracy(self):
        return self.reference_acc


app = FastAPI()
model = GenderPredictionModel(model_path)

@app.post("/predict")
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
            'reference_acc': model.get_reference_accuracy(),
            'prediction': str(pred)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)