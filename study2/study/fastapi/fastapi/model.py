from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from PIL import Image
from tensorflow.python.keras.models import Sequential, load_model
import numpy as np

model = load_model('./model4.h5')
input_shape = model.layers[0].input_shape
app = FastAPI()

@app.get('/')
def root_route():
    return {"error": "use GET /prediction instead of root route"}

@app.post('/prediction')
async def prediction_route(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = Image.open(BytesIO(contents))

    pil_image = pil_image.resize((input_shape[1], input_shape[2]))
    pil_image = pil_image.convert('L')

    numpy_array = np.array(pil_image)
    numpy_array = np.expand_dims(numpy_array, axis=-1)
    numpy_array = numpy_array / 255.0
    if numpy_array.sum() > 200:
        numpy_array = 1 - numpy_array
    prediction_array = np.array([numpy_array])

		# 예측 및 반환
    predictions = model.predict(prediction_array)
    # prediction = np.argmax(predictions[0])
    return {"result": int(predictions)}