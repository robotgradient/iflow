import numpy as np
import matplotlib.pyplot as plt
import torch
from iflow.utils.generic import to_numpy


class TestClass():
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.N = 100

    def points_evolution(self):
        x0 = torch.ones(1, 3)
        trj_n = self.dynamics.generate_trj(x0, T=self.N, noise=False)
        trj_n_np = to_numpy(trj_n)

        plt.plot(trj_n_np[:, 0], '*')
        plt.show()

    def noise_forward_evaluation(self):
        x0 = torch.ones(100, 3)
        trj_n = self.dynamics.generate_trj(x0, T=self.N, noise=True)
        trj_n_np = to_numpy(trj_n)
        for i in range(10):
            plt.plot(trj_n_np[:, i, 0])
        plt.show()

    def noise_backward_evaluation(self):
        x0 = torch.ones(1, 3)*0.1
        trj_n = self.dynamics.generate_trj(x0, T=self.N, noise=False)
        x0 = torch.ones(100, 3) * trj_n[-1, 0, :]
        trj_n = self.dynamics.generate_trj(x0, T=self.N, noise=True, reverse=True)
        trj_n_np = to_numpy(trj_n)
        for i in range(100):
            plt.plot(trj_n_np[:, i, 0])
        plt.show()

    def forward_density(self):
        x0 = torch.ones(1, 3)
        tr_mu, tr_var = self.dynamics.generate_trj_density(x0, self.N, reverse=False)
        print(tr_mu.shape)
        print(tr_var.shape)
        tr_mu = to_numpy(tr_mu)
        tr_var = to_numpy(tr_var)

        l_trj = tr_mu[:, 0, 0] - 3 * np.sqrt(tr_var[:, 0, 0, 0])
        h_trj = tr_mu[:, 0, 0] + 3 * np.sqrt(tr_var[:, 0, 0, 0])

        t = np.linspace(0, tr_mu.shape[0], tr_mu.shape[0])
        plt.plot(t, tr_mu[:, 0, 0])
        plt.fill_between(t, l_trj, h_trj, alpha=0.3)

        x0 = torch.ones(30, 3)
        trj_n = self.dynamics.generate_trj(x0, T=self.N, noise=True, reverse=False)
        trj_n_np = to_numpy(trj_n)
        for i in range(10):
            plt.plot(trj_n_np[:, i, 0])
        plt.show()
    
    def backward_density(self):
        x0 = torch.ones(1, 3) * 0.01
        tr_mu, tr_var = self.dynamics.generate_trj_density(x0, self.N, reverse=True)
        print(tr_mu.shape)
        print(tr_var.shape)
        tr_mu = to_numpy(tr_mu)
        tr_var = to_numpy(tr_var)

        l_trj = tr_mu[:, 0, 0] - 3 * np.sqrt(tr_var[:, 0, 0, 0])
        h_trj = tr_mu[:, 0, 0] + 3 * np.sqrt(tr_var[:, 0, 0, 0])

        t = np.linspace(0, tr_mu.shape[0], tr_mu.shape[0])
        plt.plot(t, tr_mu[:, 0, 0])
        plt.fill_between(t, l_trj, h_trj, alpha=0.3)
        ### Random TRJS ####
        x0 = torch.ones(30, 3) * 0.01
        trj_n = self.dynamics.generate_trj(x0, T=self.N, noise=True, reverse=True)
        trj_n_np = to_numpy(trj_n)
        for i in range(10):
            plt.plot(trj_n_np[:, i, 0])
        plt.show()