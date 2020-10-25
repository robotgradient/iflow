import os, sys, time
import numpy as np
import scipy.io as spio
import torch
from iflow.dataset.generic_dataset import Dataset


directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..','data')) + '/LASA_dataset/'


class LASA():
    def __init__(self, filename, device=torch.device('cpu')):

        ## Define Variables and Load trajectories ##
        self.filename = filename
        self.dim = 2
        self.device = device
        mat = spio.loadmat(directory + filename + '.mat', squeeze_me=True)
        self.trajs_real=[]
        for demo_i in mat['demos']:
            x = demo_i[0]
            y = demo_i[1]
            tr_i = np.stack((x,y))
            self.trajs_real.append(tr_i.T)
        trajs_np = np.asarray(self.trajs_real)
        self.n_trajs = trajs_np.shape[0]
        self.trj_length = trajs_np.shape[1]
        self.n_dims  = trajs_np.shape[2]

        ## Normalize trajectories ##
        trajs_np = np.reshape(trajs_np, (self.n_trajs * self.trj_length, self.n_dims))
        self.mean = np.mean(trajs_np,axis=0)
        self.std = np.std(trajs_np, axis=0)
        self.trajs_normalized = self.normalize(self.trajs_real)

        ## Build Train Dataset
        self.train_data = []
        for i in range(self.trajs_normalized.shape[0]):
            self.train_data.append(self.trajs_normalized[i, ...])
        self.dataset = Dataset(trajs=self.train_data, device=device)

    def normalize(self, X):
        Xn = (X - self.mean)/self.std
        return Xn

    def unormalize(self, Xn):
        X = Xn*self.std + self.mean
        return X


if __name__ == "__main__":
    filename = 'Spoon'
    device = torch.device('cpu')
    lasa = LASA(filename, device)
    print(lasa)