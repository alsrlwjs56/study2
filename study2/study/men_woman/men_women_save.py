import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, MaxPool2D, Dropout
from keras.layers import BatchNormalization, Dropout
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
import time
import os
from fz_logger import fz_logger
from tensorflow.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir="logs/{}".format('test'))

script_name = os.path.basename(__file__).split(".")[0]

# INIT fz_logger get instance
log = fz_logger.Logger(className=script_name, lvl="INFO", filePath="./log", save=True)
logger = log.initLogger()

# 1. 데이터
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     )

# xy_data = train_datagen.flow_from_directory(
#     './dataset',
#     # './data/',
#     target_size=(150, 150),
#     batch_size=5418,
#     class_mode='binary',
#     shuffle=True
# )

# mypic = train_datagen.flow_from_directory(
#     './mypic/',
#     # './mypic/',
#     target_size=(150, 150),
#     batch_size=1,
#     class_mode='binary',
#     shuffle=True
# )

# x = xy_data[0][0]
# y = xy_data[0][1]
# mypic = mypic[0][0]

# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42,stratify=y)

# # #save
# np.save('./train_x3.npy', arr =x_train)
# np.save('./train_y3.npy', arr =y_train)
# np.save('./test_x3.npy', arr =x_test)
# np.save('./test_y3.npy', arr =y_test)
# np.save('./man2.npy', arr=mypic)

# load
x_train = np.load('./train_x3.npy')
y_train = np.load('./train_y3.npy')
x_test = np.load('./test_x3.npy')
y_test = np.load('./test_y3.npy')
mypic = np.load('./save/man2.npy')

# 2. 모델
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

# 3. 컴파일, 훈련
start = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
log = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_split=0.2,callbacks=[tensorboard]) 
end = time.time() - start

# model save
# model.save_weights('./men_women_weights_f.h5')

# predict
pred = model.predict(mypic)
print('pred: ', pred)
if pred > 0.5 :
    print('Male')
else :
    print('Female')
    
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_predict = np.round(y_predict,0)
acc = accuracy_score(y_test, y_predict)
logger.info(y_test.shape)
logger.info(y_predict.shape)
print('accscore:', accuracy_score(y_test,np.round(abs(y_predict))))

loss = log.history['loss']
accuracy = log.history['accuracy']
val_loss = log.history['val_loss']
val_accuracy = log.history['val_accuracy']

print('loss: ', loss[-1])
print('accuracy: ', accuracy[-1])
print('val_loss: ', val_loss[-1])
print('val_accuracy: ', val_accuracy[-1])
print('걸린시간:',end)

plt.plot(log.history['accuracy'])
plt.plot(log.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize log for loss
plt.plot(log.history['loss'])
plt.plot(log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

exit()

# mpl.rcParams['axes.unicode_minus'] = False
# epochs = range(1, len(accuracy) + 1)
# plt.plot(epochs, accuracy, 'bo', label='Training acc')
# plt.plot(epochs, val_accuracy, 'b', label='Validation acc')
# plt.title('Accuracy')
# plt.legend()
# plt.figure()
 
# plt.plot(epochs, loss, 'ro', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Loss')
# plt.legend()
 
# plt.show()