from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import glob
import os
import sys

def create_rel_path():
    abs_path = os.path.dirname(__file__)
    dirname_string = abs_path[:abs_path.find("\github")]
    dirname_path = os.path.normpath(abs_path)
    path_for_train = os.path.join(dirname_path , "dataset","traindata","traindata")
    path_for_validation = os.path.join(dirname_path , 'dataset','corss validation')
    path_for_test = os.path.join(dirname_path , "dataset","testdata","testdata")
    return (path_for_train,path_for_validation, path_for_test)

def preprocess_forTF():
    (path_for_train,path_for_validation, path_for_test) = create_rel_path()
    ###predeclared parameters for the learning
    IMG_HEIGHT = 200
    IMG_WIDTH = 200

    ###all data sets will use as train set, validation set and test set
    train_image_generator = ImageDataGenerator(
        rescale=1. / 255
    )  # Generator for our training data

    validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data
    test_image_generator = ImageDataGenerator(rescale=1. / 255)
    train_data_gen = train_image_generator.flow_from_directory(
                                                           directory=path_for_train,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
    val_data_gen = validation_image_generator.flow_from_directory(
                                                              directory=path_for_validation,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')
    test_data_gen = test_image_generator.flow_from_directory(
                                                         directory=path_for_test,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode='binary')
    return (train_data_gen, val_data_gen,test_data_gen)


def preprocess_forDir(path_to_save_in=r"C:\Users\amitai\Desktop\machine learning finale project\men-women-classification\logistic\dataset"):
    (path_for_train, path_for_validation, path_for_test) = create_rel_path()
    procc_folder(path_for_train, os.path.join(path_to_save_in,"ta"))
    procc_folder(path_for_validation, os.path.join(path_to_save_in,"tb"))
    procc_folder(path_for_test, os.path.join(path_to_save_in,"tc"))
    print("done")


def procc_folder(path_to_take_from, path_to_save_in):
    procc_folder_men(path_to_take_from, path_to_save_in)
    procc_folder_women(path_to_take_from, path_to_save_in)

def procc_folder_men(path_to_take_from, path_to_save_in):
    num = 1
    path_to_take_from = os.path.join(path_to_take_from,"men","*.jpg")
    for filename in glob.glob(path_to_take_from):
        im = Image.open(filename)
        im = im.resize((200, 200), Image.ANTIALIAS)
        n_path = os.path.join(path_to_save_in, str(num)) + '.' + 'jpg'
        im.save(n_path)
        num += 1
    print("saved ",num," images of men in",path_to_save_in)

def procc_folder_women(path_to_take_from, path_to_save_in):
    num = 1
    path_to_take_from = os.path.join(path_to_take_from,"women","*.jpg")
    for filename in glob.glob(path_to_take_from):
        im = Image.open(filename)
        im = im.resize((200, 200), Image.ANTIALIAS)
        n_path = os.path.join(path_to_save_in, str(num)) + '.' + 'jpg'
        im.save(n_path)
        num += 1
    print("saved ",num," images of women in",path_to_save_in)



