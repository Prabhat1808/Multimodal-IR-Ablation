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
from sklearn.decomposition import PCA
from numpy import linalg
from os import listdir
from os.path import join
from sklearn.manifold import TSNE

epsilon = 0.00000001

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
        self.X1_train = None
        self.X2_train = None
        self.Y_train = None
        self.X1_test = None
        self.X2_test = None
        self.Y_test = None
        # helps in map calculation
        self.num_relevant_items = None

    def findNumRelevantItems(self):
        """
            This funtion finds number of relevant items for every query
            and stores it in self.num_relevant_items variable. This
            will be used in MAP and recall calculation later in model.py file.
            self.num_relevant_items.shape = (#test samples, )
        """
        similarity_matrix = np.matmul(self.Y_train, self.Y_test.T)
        similarity_matrix = np.where(similarity_matrix>0, 1.0, 0.0)
        self.num_relevant_items = np.sum(similarity_matrix, axis = 0)

    def stats(self):
        """
            This funtion prints type and shape of all input matrices on STDOUT.
            In is necessary that input features are of float type.
        """
        print("Train image features:", len(self.X1_train), self.X1_train[0].shape, type(self.X1_train[0][0, 0]))
        print("Train text features:", len(self.X2_train), self.X2_train[0].shape, type(self.X2_train[0][0, 0]))
        print("Test image features:", self.X1_test.shape,  type(self.X1_test[0, 0]))
        print("Test text features:", self.X2_test.shape, type(self.X2_test[0, 0]))
        print("Train labels:", self.Y_train.shape, type(self.Y_train[0, 0]))
        print("Test labels:", self.Y_test.shape, type(self.Y_test[0, 0]))

def loadFromNpy(filename):
    """load numpy variable from file"""
    return np.load(filename, allow_pickle = True)[()]

def dumpToNpy(dt, filename):
    """dump numpy variable to file"""
    np.save(filename, dt)

def chunkify(M, chunk_size):
    """
        Given a matrix M of m samples, make chunks of size
        chunk_size. Return list of chunks.
    """
    num_chunks = (M.shape[0] // chunk_size) + (M.shape[0] % chunk_size != 0)
    M_chunked = []
    for i in range(num_chunks):
        M_chunked.append(M[i*chunk_size:i*chunk_size + chunk_size, :])
    return M_chunked

class xmedianet(inputMatrices):
    def __init__(self, dirpath):
        chunk_size = 5000
        inputMatrices.__init__(self)
        identity = np.eye(200)
        print('Reading labels...')
        self.Y_test = []
        with open(dirpath + 'img_test_list.txt', 'r') as infile:
            while (True):
                line = infile.readline()
                if (line == None) or (len(line.split()) != 2): break
                self.Y_test.append(identity[int(line.split()[1])-1].tolist())
        self.Y_test = np.array(self.Y_test)
        with open(dirpath + 'img_train_list.txt', 'r') as infile:
            self.Y_train = []
            while (True):
                line = infile.readline()
                if (line == None) or (len(line.split()) != 2): break
                self.Y_train.append(identity[int(line.split()[1])-1].tolist())
        self.Y_train = np.array(self.Y_train)
             
        #cate_fea = np.loadtxt(dirpath + 'cate_fea.txt') # 200 x 300
        print('preparing test features...')
        self.X1_test = np.loadtxt(dirpath + 'img_test_fea.txt') # 8K x 4096
        self.X2_test = np.loadtxt(dirpath + 'txt_test_fea.txt') # 8K x 300
        
        print('preparing train features...')
        img_train_fea = np.loadtxt(dirpath + 'img_train_fea.txt') # 32K x 4096
        text_train_fea = np.loadtxt(dirpath + 'txt_train_fea.txt') # 32K x 300

        print('Applying dimensionality reduction 4096D -> 128D...')
        pca = PCA(n_components = 128)
        img_train_fea = pca.fit_transform(img_train_fea)
        self.X1_test = pca.transform(self.X1_test)

        print('Shuffling the training data...')
        indices = [[i] for i in range(img_train_fea.shape[0])]
        random.shuffle(indices)
        indices = np.array(indices).flatten()
        img_train_fea = img_train_fea[indices, :]
        text_train_fea = text_train_fea[indices, :]
        self.Y_train = self.Y_train[indices, :]

        print('Chunkifying the training data...')
        self.X1_train = chunkify(img_train_fea, chunk_size)
        self.X2_train = chunkify(text_train_fea, chunk_size)
#data = xmedianet('/mnt/f/mtp/dataset/dataset/xmedianet/')

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
            num_test_samples: number samples to consider in test set
            eliminate_unlabeled_samples: True if samples with no label associated with it
                                             has to be discarded.
                                        False otherwise
            samples_per_chunk: number of samples in one chunk.
    """
    def __init__(self, dirpath_xv, dirpath_xt, dirpath_y,\
                 num_concepts=10, num_test_samples=2000, num_train_samples=(186577-2000), eliminate_unlabeled_samples=True, samples_per_chunk=5000):
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
        # sampling n random numbers from a range without replacement
        test_indices = sorted(random.sample(range(0, ground_truths.shape[0]), num_test_samples))
        train_indices = [i for i in range(0, ground_truths.shape[0]) if i not in test_indices]
        #eligible_train_indices = [i for i in range(0, ground_truths.shape[0]) if i not in test_indices] #INP
        #train_indices = sorted(random.sample(eligible_train_indices, num_train_samples)) #IMP
        print('#train, #test samples: {}, {}'.format(len(train_indices), len(test_indices)))
        X1_train = image_features[train_indices, :]
        X2_train = text_features[train_indices, :]
        self.Y_train = ground_truths[train_indices, :]
        self.X1_test = image_features[test_indices, :]
        self.X2_test = text_features[test_indices, :]
        self.Y_test = ground_truths[test_indices, :]

        # creating chunks
        print('creating chunks of atmost {} samples for training features only...'.format(samples_per_chunk))
        total_samples = X1_train.shape[0]
        num_chunks = (total_samples//samples_per_chunk) + ((total_samples % samples_per_chunk) != 0)
        print('{} number of chunks will be created'.format(num_chunks))
        curr_index = 0
        self.X1_train = []
        self.X2_train = []
        for i in range(num_chunks):
            self.X1_train.append( X1_train[curr_index: curr_index + samples_per_chunk, :].copy().tolist())
            self.X2_train.append( X2_train[curr_index: curr_index + samples_per_chunk, :].copy().tolist())
            curr_index += samples_per_chunk

        # deleting redundent data
        del X1_train
        del X2_train

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
        self.X1_train = np.loadtxt(dirpath_xv + train_feature_file)
        self.X1_test = np.loadtxt(dirpath_xv + test_feature_file)

    def updateTrainTestTextFeatures(self, dirpath_xt):
        print('Reading text features...')
        train_text_file = 'Lite_Tags1k_Train.txt'
        test_text_file = 'Lite_Tags1k_Test.txt'
        self.X2_train = np.loadtxt(dirpath_xt + train_text_file)
        self.X2_test = np.loadtxt(dirpath_xt + test_text_file)

    def updateTrainTestLabels(self, dirpath_y):
        print('Reading ground truth labels...')
        train_label_file = 'Lite_Tags81_Train.txt'
        test_label_file = 'Lite_Tags81_Test.txt'
        self.Y_train = np.loadtxt(dirpath_y + train_label_file)
        self.Y_test = np.loadtxt(dirpath_y + test_label_file)

# following main function is for debugging purpose
#if __name__ == '__main__':
    # reading lite version of nus-wide dataset
    #data = NUS_WIDE_Lite('/mnt/f/mtp/dataset/dataset/nus_wide/NUS-WIDE-Lite/NUS-WIDE-Lite_features/',\
    #                     '/mnt/f/mtp/dataset/dataset/nus_wide/NUS-WIDE-Lite/NUS-WIDE-Lite_tags/',\
    #                     '/mnt/f/mtp/dataset/dataset/nus_wide/NUS-WIDE-Lite/NUS-WIDE-Lite_tags/')
    
    # reading full version of nuswide dataset
#    data = NUS_WIDE('/mnt/f/mtp/dataset/dataset/nus_wide/NUS-WIDE/',\
#                    '/mnt/f/mtp/dataset/dataset/nus_wide/NUS-WIDE/',\
#                    '/mnt/f/mtp/dataset/dataset/nus_wide/NUS-WIDE/')
#    data.stats()
