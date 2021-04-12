"""
    This file extracts features for raw images or texts from various datasets.
    For example it stores image features and text features into variables
    declared in inputMatrices class. All feature extraction related code must
    reside here.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import normalize
from numpy import linalg
from os import listdir
from os.path import join
from sklearn.manifold import TSNE

epsilon = 0.00000001
def zeroMeanRow(x):
    return x - np.mean(x, 1).reshape((-1, 1))
def zeroMean(x, isTest=False, train_mean=None):
    if (isTest == False):
        col_mean = np.mean(x, 0)
        return x - col_mean, col_mean
    else:
        return x - train_mean
def l2Normalize(x):
    return normalize(x, 'l2')

def tsne(x, num_samples, filename, title):
    x_embedded = TSNE().fit_transform(x[:num_samples, :])
    plt.scatter(x_embedded[:, 0], x_embedded[:, 1], s=2)
    plt.title(title + ', {} samples'.format(num_samples))
    plt.savefig(filename)
    plt.clf()

class inputMatrices:
    """
        This is a super class. This class stores input matrices:
        e.g. image features, text features, labels for training
        and testing dataset. It abstracts out the process of
        getting these matrices and only cares for the matrices.
    """
    def __init__(self):
        """ Declares various input matrices. 
            E.g. train_xv for storing training image features
        """
        self.train_xv = None
        self.train_xt = None
        self.train_y = None
        self.test_xv = None
        self.test_xt = None
        self.test_y = None
        self.xv_anchors = None
        self.xt_anchors = None
        # helps in map calculation
        self.num_relevant_items = None

    def zeroMeanL2Normalize(self):
        """apply zero mean L2 norm to input matrices"""
        print('Applying zero mean L2 norm normalization...')
        self.train_xv = zeroMeanRow(self.train_xv)
        self.train_xv = l2Normalize(self.train_xv)
        self.test_xv = l2Normalize(zeroMeanRow(self.test_xv))

        self.train_xt = zeroMeanRow(self.train_xt)
        self.train_xt = l2Normalize(self.train_xt)
        self.test_xt = l2Normalize(zeroMeanRow(self.test_xt))

    def zeroMeanUnitVariance(self):
        """
            apply zero mean unit variance to input matrices
            Note: If in one of the dimension, standard deviation is
                  zero then it is replaced with zero to avoid 
                  divide by zero error.
        """
        print('Applying zero mean unit variance normalization...')
        xv_mean = np.mean(self.train_xv, 0)
        xt_mean = np.mean(self.train_xt, 0)
        xv_std = np.std(self.train_xv, 0)
        xv_std = np.where(xv_std==0, 1.0, xv_std)
        xt_std = np.std(self.train_xt, 0)
        xt_std = np.where(xt_std==0, 1.0, xt_std)

        self.train_xv = (self.train_xv - xv_mean)/xv_std
        self.test_xv = (self.test_xv - xv_mean)/xv_std
        self.train_xt = (self.train_xt - xt_mean)/xt_std
        self.test_xt = (self.test_xt - xt_mean)/xt_std

    def findNumRelevantItems(self):
        """
            This funtion finds number of relevant items for every query
            and stores it in self.num_relevant_items variable. This
            will be used in MAP and recall calculation later in model.py file.
            self.num_relevant_items.shape = (#test samples, )
        """
        similarity_matrix = np.matmul(self.train_y, self.test_y.T)
        similarity_matrix = np.where(similarity_matrix>0, 1.0, 0.0)
        self.num_relevant_items = np.sum(similarity_matrix, axis = 0)

    def stats(self):
        """
            This funtion prints type and shape of all input matrices on STDOUT.
            In is necessary that input features are of float type.
        """
        print("Train image features:", self.train_xv.shape, type(self.train_xv[0, 0]))
        print("Train text features:", self.train_xt.shape, type(self.train_xt[0, 0]))
        print("Test image features:", self.test_xv.shape,  type(self.test_xv[0, 0]))
        print("Test text features:", self.test_xt.shape, type(self.test_xt[0, 0]))
        print("Train labels:", self.train_y.shape, type(self.train_y[0, 0]))
        print("Test labels:", self.test_y.shape, type(self.test_y[0, 0]))

def loadFromNpy(filename):
    """load numpy variable from file"""
    return np.load(filename, allow_pickle = True)[()]

def dumpToNpy(dt, filename):
    """dump numpy variable to file"""
    np.save(filename, dt)

class NUS_WIDE(inputMatrices):
    """
        This class is a derived class of inputMatrices class. This
        class aims at storing the input matrices in inputMatrices
        class. Update following class variables: train_xv, test_xv, 
        train_xt, test_xt, train_y, test_y.
        
        inputs:
            dirpath_xv: path to BoW_int.dat file
            dirpath_xt: path to AllTags1k.txt file
            dirpath_y: path to AllTags81.txt file
            num_concepts: consider num_concepts number of most frequent labels only
            test_percent: how many percent of total dataset goes to test data
            eliminate_unlabeled_samples: True if samples with no label associated with it
                                             has to be discarded.
                                        False otherwise
    """
    def __init__(self, dirpath_xv, dirpath_xt, dirpath_y,\
                 num_concepts=81, test_percent=20, eliminate_unlabeled_samples=False):
        inputMatrices.__init__(self)
        print('reading input matrices...')
        image_features = np.loadtxt(dirpath_xv + 'BoW_int.dat')
        text_features = np.loadtxt(dirpath_xt + 'AllTags1k.txt')
        ground_truths = np.loadtxt(dirpath_y + 'AllTags81.txt')

        print('considering only {} most frequent labels...'.format(num_concepts))
        # consider num_concepts number of most frequent labels
        freq = np.matmul(np.ones((1, ground_truths.shape[0])), ground_truths)[0]
        topk_concepts = np.sort(np.argsort(-1*freq)[:num_concepts]) # based on freq
        ground_truths = ground_truths[:, topk_concepts]

        # removing examples with no label associated with it
        if (eliminate_unlabeled_samples):
            concept_count = np.matmul(ground_truths, np.ones((ground_truths.shape[1], 1))).reshape((1, -1))[0]
            valid_item_indices = np.where(concept_count != 0)[0]
            ground_truths = ground_truths[valid_item_indices, :]
            image_features = image_features[valid_item_indices, :]
            text_features = text_features[valid_item_indices, :]

        print('spliting into training and testing data...')
        num_test_samples = (ground_truths.shape[0] * test_percent)//100
        # sampling n random numbers from a range without replacement
        test_indices = sorted(random.sample(range(0, ground_truths.shape[0]), num_test_samples))
        train_indices = [i for i in range(0, ground_truths.shape[0]) if i not in test_indices]
        self.train_xv = image_features[train_indices, :]
        self.test_xv = image_features[test_indices, :]
        self.train_xt = text_features[train_indices, :]
        self.test_xt = text_features[test_indices, :]
        self.train_y = ground_truths[train_indices, :]
        self.test_y = ground_truths[test_indices, :]

class NUS_WIDE_Lite(inputMatrices):
    """
        This class is a derived class of inputMatrices class. This
        class aims at storing the input matrices in inputMatrices
        class.
        
        inputs:
            dirpath_xv : path to .dat image files
            dirpath_xt : path to Lite_Tags1k_*.txt files
            dirpth_y   : path to Lite_Tags81_*.txt files
    """
    def __init__(self, dirpath_xv, dirpath_xt, dirpath_y):
        inputMatrices.__init__(self)
        self.updateTrainTestImageFeatures(dirpath_xv)
        self.updateTrainTestTextFeatures(dirpath_xt)
        self.updateTrainTestLabels(dirpath_y)

    def updateTrainTestImageFeatures(self, dirpath_xv):
        print('Reading image features...')
        train_feature_file = 'Normalized_CORR_Lite_Train.dat'
        test_feature_file = 'Normalized_CORR_Lite_Test.dat'
        self.train_xv = np.loadtxt(dirpath_xv + train_feature_file)
        self.test_xv = np.loadtxt(dirpath_xv + test_feature_file)

    def updateTrainTestTextFeatures(self, dirpath_xt):
        print('Reading text features...')
        train_text_file = 'Lite_Tags1k_Train.txt'
        test_text_file = 'Lite_Tags1k_Test.txt'
        self.train_xt = np.loadtxt(dirpath_xt + train_text_file)
        self.test_xt = np.loadtxt(dirpath_xt + test_text_file)

    def updateTrainTestLabels(self, dirpath_y):
        print('Reading ground truth labels...')
        train_label_file = 'Lite_Tags81_Train.txt'
        test_label_file = 'Lite_Tags81_Test.txt'
        self.train_y = np.loadtxt(dirpath_y + train_label_file)
        self.test_y = np.loadtxt(dirpath_y + test_label_file)

# following main function is for debugging purpose
if __name__ == '__main__':
    # reading lite version of nus-wide dataset
    #data = NUS_WIDE_Lite('/mnt/f/mtp/dataset/dataset/nus_wide/NUS-WIDE-Lite/NUS-WIDE-Lite_features/',\
    #                     '/mnt/f/mtp/dataset/dataset/nus_wide/NUS-WIDE-Lite/NUS-WIDE-Lite_tags/',\
    #                     '/mnt/f/mtp/dataset/dataset/nus_wide/NUS-WIDE-Lite/NUS-WIDE-Lite_tags/')
    
    # reading full version of nuswide dataset
    data = NUS_WIDE('/mnt/f/mtp/dataset/dataset/nus_wide/NUS-WIDE/',\
                    '/mnt/f/mtp/dataset/dataset/nus_wide/NUS-WIDE/',\
                    '/mnt/f/mtp/dataset/dataset/nus_wide/NUS-WIDE/')
    data.stats()
