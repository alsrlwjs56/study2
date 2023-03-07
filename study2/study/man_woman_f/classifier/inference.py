import argparse
import cv2
import numpy as np
from PIL import Image
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, MaxPool2D, Dropout
from keras.layers import BatchNormalization, Dropout

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='men_women_weights.h5', help='', required=False)
    parser.add_argument('--source', type=str, default='0', help='', required=False)
    parser.add_argument('--imgH', type=int, default=150, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=150, help='the width of the input image')
    parser.add_argument('--image_folder', type=str, default="mypic/mypic", required=False, help='path to image_folder which contains text images')
    args = parser.parse_args()
    return args

def pre_processing(dims, image):
    target_width, target_height = dims
    img = Image.open(image)
    img = img.resize((target_width, target_height))
    
    return img

def main():
    opt = get_argparse()
    if '.' in opt.source:
        pass
    else:
        opt.source = int(opt.source)
     
    cap = cv2.VideoCapture(opt.source)
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
    model.load_weights('./men_women_weights_f.h5')
    
    while cap.isOpened():
        r, f = cap.read()
        if r:
            resized_frame = cv2.resize(f, (150, 150))
            input_data = np.expand_dims(resized_frame, axis=0) / 255.0
            
            pred = model.predict(input_data)
            
            if pred < 0.5:
                gender = "Male"
                color = (255, 0, 0)  # Blue
            else:
                gender = "Female"
                color = (0, 0, 255)  # Red
                
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(f, (x, y), (x + w, y + h), color, 2)
                cv2.putText(f, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
            cv2.imshow("", f)
            
            if cv2.waitKey(1) == ord('q'):
                break
            
    cap.release()
    cv2.destroyAllWindows()        

if __name__=='__main__':
    main()