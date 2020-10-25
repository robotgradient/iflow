import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from iflow.dataset.generic_dataset import Dataset


directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..','data')) + '/POURING_dataset/'


class POURING():
    def __init__(self, device = torch.device('cpu')):
        ## Define Variables and Load trajectories ##
        self.type = type
        self.dim = 6
        self.dt = .01

        trj_filename = 'Pouring.npy'
        trajs_np = np.load(os.path.join(directory, trj_filename),allow_pickle=True)

        self.trajs_real=[]
        for i in range(trajs_np.shape[0]):
            self.trajs_real.append(trajs_np[i])
        self.n_trajs = trajs_np.shape[0]
        self.n_dims = trajs_np[0].shape[-1]

        ####### Normalize Trajectories #########
        trajs_np = self.full_trajs()
        self.mean = self.compute_mean(trajs_np)
        self.std = self.compute_std(trajs_np)
        self.trajs_normalized = self.normalize(self.trajs_real)

        ###### Build Train Dataset #######
        self.train_data = self.trajs_normalized
        self.dataset = Dataset(trajs=self.train_data, device=device, steps=10)


    def full_trajs(self):
        trajs = np.zeros((0, self.dim))
        for trj in self.trajs_real:
            trajs = np.concatenate((trajs, trj), 0)
        return trajs

    def compute_mean(self, trajs):
        mean_xyz = np.mean(trajs[:,:3].flatten())
        mean_rpy = np.mean(trajs[:,3:].flatten())
        mean = np.array([mean_xyz,mean_xyz,mean_xyz, mean_rpy, mean_rpy, mean_rpy])
        return mean

    def compute_std(self,trajs):
        std_xyz = np.std(trajs[:,:3].flatten())
        std_rpy = np.std(trajs[:,3:].flatten())
        std = np.array([std_xyz,std_xyz,std_xyz, std_rpy, std_rpy, std_rpy])
        return std

    def normalize(self, X_l):
        X_n_l = []
        for X in X_l:
            Xn = (X - self.mean) / self.std
            X_n_l.append(Xn)
        return X_n_l

    def unormalize(self, Xn_l):
        X_l = []
        for Xn in Xn_l:
            X = Xn * self.std + self.mean
            X_l.append(X)
        return X_l



if __name__ == "__main__":
    pouring_data = POURING(device=None)
    print(pouring_data)
