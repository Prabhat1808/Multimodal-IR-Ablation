import os
import os.path as path
import numpy as np
import random
import os.path as path
import sys
import time
import json
import matplotlib.pyplot as plt

class Comparator:
    def __init__(self, files):
        self.stats = [np.load(f, allow_pickle=True)[()] for f in files]
        self.labels = [path.basename(f).split('.')[0] for f in files]
        self.comparisions = {}
        self.outdir = '_V_'.join(self.labels)
        if not path.exists(self.outdir):
            os.makedirs(self.outdir)

    # creates a bar plot
    # supported tags - params_size, training_time, inference_time
    # other tags can also be used, but consistency has to be checked by the user
    def createBarPlot(self, tag):
        labels = self.labels
        param_sizes = [s[tag] for s in self.stats]
        plt.title(tag)
        plt.bar(labels, param_sizes)
        outfile = path.join(self.outdir, '{}.jpeg'.format(tag))
        plt.savefig(outfile)
        plt.close()
    
    # supported tags: loss_histry, metrics
    # If the metric object is 2-level, then a subtag can be provided as well
    # For example -> self.stats1[tag][subtag]
    def createLinePlot(self, tag, subtag=''):
        for i in range(len(self.labels)):
            label = self.labels[i]
            line = self.stats[i][tag]
            if subtag != '':
                line = line[subtag]
            indices = [j for j in range(1,len(line)+1)]
            plt.plot(indices, line, label=label)       

        plt.title(tag+ ' ' + subtag)
        plt.ylabel(tag + ' ' + subtag)
        plt.xlabel('i')
        plt.legend()
        outfile = path.join(self.outdir, '{}.jpeg'.format(tag+' '+subtag))
        plt.savefig(outfile)
        plt.close()

    # precision vs recall
    # dataset stats - TODO Dwijesh
    # multiple model comparisions - DONE