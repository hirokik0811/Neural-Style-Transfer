'''
Created on Oct 9, 2017

@author: kwibu
'''
import numpy as np

class clustor:
    def __init__(self, data=None, imsize=32*32, n_labels=62):
        self.imsize = imsize
        self.n_labels = n_labels
        if data==None:
            self.images = np.empty([0, imsize])
            self.labels = np.empty([0, n_labels])
            self.n_data = 0
        else:
            self.images = data[:, :imsize]
            self.lables = data[:, imsize:]
            self.n_data = data.shape[0]
            
    def row_stack(self, target):
        self.images = np.vstack((self.images, target[:, :self.imsize]))
        self.labels = np.vstack((self.labels, target[:, self.imsize:]))
        self.n_data = self.images.shape[0]
        
class dataset:
    def __init__(self, data, imsize=32*32, n_labels=62, test_rate=0.3):
        self.train = clustor()
        self.test = clustor()
        for ds in data:
            np.random.shuffle(ds)
            self.train.row_stack(ds[:int(ds.shape[0]*(1-test_rate)), :])
            self.test.row_stack(ds[:int(ds.shape[0]*test_rate), :])
            
    def next_batch(self, n_batches):
        indices = np.random.randint(self.train.n_data, size=n_batches)
        return [self.train.images[indices], self.train.labels[indices]]