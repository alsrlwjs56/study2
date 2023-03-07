from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, MaxPool2D, Dropout
from keras.layers import BatchNormalization, Dropout
from tensorflow.python.keras.models import Sequential, load_model
from PIL import Image
import io
import numpy as np
import cv2

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

    def load_reference_images(self):
        self.reference_images = {
            './man4.jpg': 'male'
            
            # Add more image paths and associated genders here
        }
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
    
    # def get_reference_accuracy(self):
    #     return self.reference_acc
    
    def get_reference_accuracy(self):
        total = 0
        correct = 0

        for image_path, gender in self.reference_images.items():
            # Load image from file path
            image = cv2.imread(image_path)

            # Preprocess the image
            image = self.preprocess_image(image)

            # Predict the gender
            pred = self.predict_gender(image)

            # Check if the prediction matches the reference gender
            if pred == gender:
                correct += 1

            total += 1

        # Calculate and return the account score
        accuracy = correct / total
        account_score = accuracy * 100

        return account_score
