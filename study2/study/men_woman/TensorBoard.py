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
from tensorflow.keras.callbacks import TensorBoard

tensorboard = TensorBoard(log_dir="logs/{}".format('test'))

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
# VGG16#################################################################################
# vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
# model = Sequential()
# model.add(vgg16)
# model.add(Conv2D(64, (2,2), input_shape=(150,150,3), activation='relu'))
# model.add(Conv2D(128, (3,3), activation='relu'))
# model.add(GlobalAveragePooling2D())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.save('./model3.h5')
# model.summary()

# model = Sequential()
# model.add(Conv2D(input_shape=(150,150,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
# model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# model.add(Flatten())
# model.add(Dense(4096,activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dense(4096,activation="relu"))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.summary()
########################################################################################
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
# model.load_weights('./men_women_weights3.h5')
# model.save_weights('./men_women_weights3.h5')
# 3. 컴파일, 훈련


start = time.time()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
log = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2,callbacks=[tensorboard]) 
end = time.time() - start

# # # 2-1 model.save & load
# model.save_weights('./men_women_weights2.h5')
# # # weights만 저장 및 불러오기
# # model.save_weights('./model_weights.h5')
# # model.load_weights('./model_weights.h5')

# # # model과 weights 동시에 저장 및 불러오기
# # model.load_model('model.h5')
# # model.save('./model0.h5')
# # model = load_model("./model.h5")

# 그래프


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
print('acc스코어 : ', acc)
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

mpl.rcParams['axes.unicode_minus'] = False
epochs = range(1, len(accuracy) + 1)

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

# 1 epochs
# pred:  [[0.5654171]]
# 여자
# loss:  1.2368522882461548
# accuracy:  0.5772319436073303
# val_loss:  0.6769571304321289
# val_accuracy:  0.5754716992378235
# 걸린시간: 240.77914834022522

# 2 epochs
# pred:  [[0.5301963]]
# 여자
# loss:  0.655275285243988
# accuracy:  0.6428908705711365
# val_loss:  0.6881383657455444
# val_accuracy:  0.5698113441467285
# 걸린시간: 482.6566777229309

# pred:  [[0.8631362]]
# 남자
# loss:  0.6142119765281677
# accuracy:  0.698157787322998
# val_loss:  0.7896648049354553
# val_accuracy:  0.5924528241157532
# 걸린시간: 728.4304859638214