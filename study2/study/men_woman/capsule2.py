import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, MaxPool2D, Dropout
from keras.layers import BatchNormalization, Dropout
import time
import argparse
import os
from fz_logger import fz_logger

script_name = os.path.basename(__file__).split(".")[0]

# INIT fz_logger get instance
log = fz_logger.Logger(className=script_name, lvl="INFO", filePath="./log", save=True)
logger = log.initLogger()


def argparses():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,   default=1)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--input_size",   type=int,   default=150)
    args = parser.parse_args()
    return args

def load_data(opt):
    # load data
    x_train = np.load('/home/funzin/study/men_woman/train_x3.npy')
    y_train = np.load('/home/funzin/study/men_woman/train_y3.npy')
    x_test = np.load('/home/funzin/study/men_woman/test_x3.npy')
    y_test = np.load('/home/funzin/study/men_woman/test_y3.npy')
    mypic = np.load('./save/man.npy')
    return x_train, y_train, x_test, y_test, mypic

def build_model(opt):
    # build model
    model = Sequential()
    model.add(Conv2D(128,(3,3),input_shape=(opt.input_size,opt.input_size,3),activation='relu'))
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
    
    return model

def train_model(opt, x_train, y_train):
    # compile and train model
    model = build_model(opt)
    start = time.time()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    # log = model.fit(x_train, y_train, epochs=opt.epochs, batch_size=opt.batch_size, validation_split=0.2) 
    end = time.time() - start
    return model, log, end

def results(x_test, y_test, mypic, model):
    loss = model.evaluate(x_test, y_test)
    y_predict = model.predict(x_test)
    y_predict = np.round(abs(y_predict))
    acc = accuracy_score(y_test,y_predict)
    
    pred = model.predict(mypic)
    logger.info('pred: ', pred)
    if pred > 0.5 :
        print('Male')
    else :
        print('Female')
        
    print('loss : ',loss)
    print('acc스코어 : ', acc)
    
def main():
    opt = argparses()
    x_train, y_train, x_test, y_test, mypic = load_data(opt)
    
    model, log, end = train_model(opt=opt, x_train=x_train, y_train=y_train)
    print(f"train time : {end}")
    
    results(x_test, y_test, mypic, model)
    
    
if __name__=='__main__':
    main()