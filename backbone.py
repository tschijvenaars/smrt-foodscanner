import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Lambda
from sklearn.model_selection import train_test_split

def init_data():
    """
    Initializes all global path files.
    """
    global file_path_trainlabels
    global train_labels
    global file_path_trainimages
    file_path_trainlabels = os.getcwd() + '/data/train_labels.csv'
    train_labels = pd.read_csv(file_path_trainlabels, index_col=False)
    file_path_trainimages = os.getcwd() +'/data/train_set/train_set/'
    return [file_path_trainlabels,file_path_trainimages,train_labels]

def make_split(percentage):
    """
    Takes the percentage of test set size (validation set in our case). 
    Returns x_train, x_val, y_train and y_val as a list.
    """
    init_data()
    x_train, x_val, y_train, y_val = train_test_split(train_labels['img_name'], train_labels['label'], test_size=percentage)
    x_train.reset_index(drop=True, inplace=True)
    x_val.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    return [x_train,x_val,y_train,y_val]

def data_export(predicted_image_list,predicted_score_list):
    """
    Takes a numpy list of predicted images names and predicted score list.
    Then it exports the data to a .csv file within the working directory.
    """
    np.asarray(predicted_image_list)
    np.asarray(predicted_score_list)
    out_dataset = pd.DataFrame({'img_name': predicted_image_list, 'label': predicted_score_list}, columns=['img_name', 'label'])
    out_dataset.to_csv('model_output.csv',index=False)

def make_image_sets(height,width,x_trainname,x_valname,ipm="nearest"):
    x_trainimgs = []
    x_valimgs = []

    for i in x_trainname:
        img = keras.preprocessing.image.load_img(
        file_path_trainimages + i, target_size=(height, width), interpolation = ipm
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        x_trainimgs.append(img_array)
        
    for i in x_valname:
        img = keras.preprocessing.image.load_img(
        file_path_trainimages + i, target_size=(height, width), interpolation = ipm
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        x_valimgs.append(img_array)
    
    x_trainimgs = np.asarray(x_trainimgs)
    x_valimgs = np.asarray(x_valimgs)
    return [x_trainimgs,x_valimgs]

def get_scores(test_model, height, width, file_path_testimages, test_labels, ipm="nearest"):
    scores = []

    for i in test_labels['img_name']:
        img = keras.preprocessing.image.load_img(
        file_path_testimages + i, target_size=(height, width), interpolation = ipm
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = test_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        scores.append(np.argmax(score))
    
    return scores
    
def get_test_paths():
    file_path_testimages = os.getcwd() +'/data/test_set/test_set/'
    file_path_testlabels = os.getcwd() + '/data/sample.csv'
    test_labels = pd.read_csv(file_path_testlabels, index_col=False)
    return [file_path_testimages,file_path_testlabels,test_labels]