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
        start = time.time()
        n_samples, results, logs = self.prediction_function(self.dataset_obj, self.params, tag)
        end = time.time()

        self.results['tag'] = results #TODO: ''.format(tag)
        self.stats['prediction_time'] = (end-start)/n_samples
        self.logs.append(logs)

    def evaluate(self):
        pass

    def get_stats(self):
        return self.stats

    def plot_curves(self):
        pass

    def save_stats(self):
        pass