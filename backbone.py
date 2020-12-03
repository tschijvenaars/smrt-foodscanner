import numpy as np
import pandas as pd
import os
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
    return [x_train,x_val,y_train,y_val]

def data_export(predicted_image_list,predicted_score_list):
    """
    Takes a numpy list of predicted images names and predicted score list.
    Then it exports the data to a .csv file within the working directory.
    """
    out_dataset = pd.DataFrame({'img_name': predicted_image_list, 'label': predicted_score_list}, columns=['img_name', 'label'])
    out_dataset.to_csv('model_output.csv',index=False)
