from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os
import matplotlib.pyplot as plt
import preproccdataset

###compilation flags to make TF more efficient
os.environ["OMP_NUM_THREADS"] = "NUM_PARALLEL_EXEC_UNITS"
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'


class LogisticRegression(tf.keras.Model):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.dense = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        self.flatten_layer = tf.keras.layers.Flatten(input_shape= (200, 200))
    def call(self, x):
        x = self.flatten_layer(x)
        x = self.dense(x)
        return x

def main():
    (IMG_HEIGHT, IMG_WIDTH) = (200,200)
    batch_size = 64
    epochs = 70
    (train_data_gen, val_data_gen,test_data_gen) = preproccdataset.preprocess_forTF()

    model = LogisticRegression()
    ###complinig the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit_generator(
        train_data_gen,
      # steps_per_epoch=2291 // batch_size,
        epochs=epochs,
        validation_data=val_data_gen,
       # validation_steps=732 // batch_size
    )

    ###summary of the model after traning
    print('\nhistory dict:', history.history)
    ###saving the model and weights as a json and h5 files
   # json_str = model.to_json()
   # with open(r'C:\Users\amitai\Desktop\machine learning finale project\res\saved_model_FIRST', 'w') as outfile:
    #    json.dump(json.loads(json_str), outfile, indent=4)  # Save the json on a file
     #   model.save_weights(r'C:\Users\amitai\Desktop\machine learning finale project\res\saved_model_FIRST_w.h5', save_format="h5")
   # print("Saved model to disk")
    ###evaluating the model on the test data
    print('\n# Evaluate on test data')
    results_test = model.evaluate_generator(test_data_gen)
    print('test loss, test acc:', results_test)
    ####printing the model as a graph
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

main()