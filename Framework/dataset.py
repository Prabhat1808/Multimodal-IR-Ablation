import os
import os.path as path
import numpy as np
import random


"""
    Dataset is the superclass which contains high level variables:
    x_{train,val,test}, y_{train,val,test}
    dir_{train,val,test}
    loader : function to read the dataset
    preprocess : function to preprocess the datasets
    normalize : normalize train, val and test datasets
    Note: while normalizing:
        1. stats obtained from train can be used to normalize val and test
        2. val and test can be normalized independently
        This needs to be fixed.... #LOOK

    Example code is provided for NUS-wide
"""
class Dataset:

    def __init__(self, directories, loader, preprocess, normalize):
        self.dir_train, self.dir_val, self.dir_test = directories
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.loader = loader
        self.preprocess = preprocess
        self.normalize = normalize
