import os
import os.path as path
import numpy as np
import random
import sys
import time


class Parameters:
    def __init__(self, values):
        self.values = values
        # This should ideally be stored in a dictionary format,
        # to allow for easy access and referencing
        self.size = sys.getsizeof(values)


def map_rank(traingnd, testgnd, hamming_rank):
    """ 
        This funtion returns map@all metric score.
        hamming_rank : numtrain x numtest
        *gnd : numsamples x labelsize
    """
    numtrain, numtest = hamming_rank.shape
    apall = np.zeros((numtrain, numtest))
    patk = np.zeros((numtrain, numtest))
    aa = np.array([i+1 for i in range(numtrain)])
    for i in range(numtest):
        y = hamming_rank[:, i]
        new_label = np.array([0 for j in range(numtrain)])
        relevant_indices = (np.matmul(traingnd, testgnd[i, :].reshape((-1, 1))) > 0).reshape(-1)
        new_label[relevant_indices] = 1
        xx = np.cumsum(new_label[y])
        x = xx * new_label[y]
        p = x / aa #precision@k
        patk[:, i] = p.copy()
        p = np.cumsum(p)
        mask = (p != 0)
        p[mask] = p[mask]/xx[mask]
        apall[:, i] = p.copy()
    pre = np.mean(patk, axis=1)
    mAP = np.mean(apall, axis=1)
    return mAP, pre

class Model:
    def __init__(self, 
                    training_function,
                    hyperparams,
                    dataset_obj,
                    params,
                    params_verification,
                    prediction_function,
                    evaluation_metrics
                    ):
        # training_function -> function that acts on the input data and returns a Parameters object
        # The subsequent inputs should be consistent with the inputs required by the training function
        # hyperparams -> Dictionary of iterations, learning rate or any other vars.
        # params -> an object of class Parameters, which contains the matrices, weights and any other parameters,
        #           in the correct shape and data type, as per the training function
        # params_verification -> ensures consistency b/w params and output of training function
        # prediction_function -> takes params and datapoint(s) as input to generate output(s)
        # evaluation_metrics -> list of evaluation metrics to be calculated
        #                       Output from prediction function should be in the required format
        #                       Implementation for those not provided by the framework
        self.train = training_function
        self.hyperparams = hyperparams
        # dataset_object has to be of type Dataset
        self.dataset_obj = dataset_obj
        self.params = params
        self.params_verification = params_verification
        self.prediction_function = prediction_function
        self.evaluation_metrics = evaluation_metrics
        self.stats = self.initialize_stats()
        self.logs = []
        self.results ={}

    def initialize_stats(self):
        stats = {
            'data_stats' : self.dataset_obj.get_stats(),
            'params_size' : self.params.size,
            'epochs' : 0,
            'training_time' : 0,
            'loss_history' : {},
            # Dictionary, where multiple lists are stored, for each kind of loss being monitored
            'inference_time' : 0,
            # Inference time per sample
            'metrics' : {}
        }
        return stats #TODO: changed

    def train_model(self):
        # returns model stats as well
        # note starting and ending time before training_function() call
        # keep track of hyperparams
        # keep track of the size Params object and the number of params therein
        
        start = time.time()
        params, losses, logs = self.train(
                                    self.dataset_obj,
                                    self.params,
                                    self.hyperparams
                                )
        end = time.time()

        self.params = params
        self.stats['loss_history'] = losses
        self.logs.append(logs)
        self.stats['training_time'] = (end-start)

    # tag = train, val, test. Or anything else, but then custom dataset and provisions in prediction_function are needed
    def predict(self, tag):
        """ calls user-defined prediction_function

            user must return the results in the following form"
            type(results) = dict
            results.keys() : ['itot_ranked_results', 'ttoi_ranked_results']
            and matrices corresponding to the each key.
        """
        start = time.time()
        n_samples, results, logs = self.prediction_function(self.dataset_obj, self.params, tag)
        end = time.time()

        self.results[tag] = results #TODO: ''.format(tag)
        self.stats['prediction_time'] = (end-start)/n_samples
        self.logs.append(logs)

    def evaluate(self, train_labels, test_labels):
        """ calculate mAP, recall, pecision
            This funtion will make use of the "results" stored in
            the predict part above.

            args:
                train_labels : #samples x #classes matrix. 
                                It should be a binary matrix saying 
                                given sample belongs what labels.
                test_labels :  #samples x #classes matrix
                                It should be a binary matrix saying 
                                given sample belongs what labels.
        """
        mAP_itot, pre_itot = map_rank(train_labels,test_labels, self.results['test']['itot_ranked_results'].T)
        mAP_ttoi, pre_ttoi = map_rank(train_labels, test_labels, self.results['test']['ttoi_ranked_results'].T)
        print('image to text mAP@max: \n', np.max(mAP_itot), np.argmax(mAP_itot))
        print('image to text P@max: \n', np.max(pre_itot), np.argmax(pre_itot))
        print('text to image mAP@max: \n', np.max(mAP_ttoi), np.argmax(mAP_ttoi))
        print('text to image P@max: \n', np.max(pre_ttoi), np.argmax(pre_ttoi))

    def get_stats(self):
        return self.stats

    def plot_curves(self):
        pass

    def save_stats(self):
        pass