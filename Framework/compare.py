import os
import os.path as path
import numpy as np
import random
import sys
import time
import json
import matplotlib.pyplot as plt

class Comparator:
    def __init__(self, file1, file2):
        self.stats1 = np.load(file1, allow_pickle=True)[()]
        self.stats2 = np.load(file2, allow_pickle=True)[()]
        self.label1 = file1.split('.')[0]
        self.label2 = file2.split('.')[0]
        self.comparisions = {}
        

    def compareParams(self):
