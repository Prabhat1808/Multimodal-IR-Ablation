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
    def __init__(self, file1, file2):
        self.stats1 = np.load(file1, allow_pickle=True)[()]
        self.stats2 = np.load(file2, allow_pickle=True)[()]
        self.label1 = path.basename(file1).split('.')[0]
        self.label2 = path.basename(file2).split('.')[0]
        self.comparisions = {}
        self.outdir = '{}_V_{}'.format(self.label1, self.label2)
        if not path.exists(self.outdir):
            os.makedirs(self.outdir)

    # creates a bar plot
    # supported tags - params_size, training_time, inference_time
    # other tags can also be used, but consistency has to be checked by the user
    def createBarPlot(self, tag):
        labels = [self.label1, self.label2]
        param_sizes = [self.stats1[tag], self.stats2[tag]]
        plt.title(tag)
        plt.bar(labels, param_sizes)
        outfile = path.join(self.outdir, '{}.jpeg'.format(tag))
        plt.savefig(outfile)
        plt.close()
    
    # supported tags: loss_histry, metrics
    # If the metric object is 2-level, then a subtag can be provided as well
    # For example -> self.stats1[tag][subtag]
    def createLinePlot(self, tag, subtag=''):       
        line1 = self.stats1[tag]
        if subtag != '':
            line1 = line1[subtag]
        indices1 = [i for i in range(1, len(line1)+1)]
        plt.plot(indices1, line1, label=self.label1)
        
        line2 = self.stats2[tag]
        if subtag != '':
            line2 = line2[subtag]
        indices2 = [i for i in range(1, len(line2)+1)]

        plt.title(tag+ ' ' + subtag)
        plt.ylabel(tag + ' ' + subtag)
        plt.xlabel('i')
        plt.plot(indices2, line2, label=self.label2)
        plt.legend()
        outfile = path.join(self.outdir, '{}.jpeg'.format(tag+' '+subtag))
        plt.savefig(outfile)
        plt.close()