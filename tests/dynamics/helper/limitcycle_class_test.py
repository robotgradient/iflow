import numpy as np
import matplotlib.pyplot as plt
import torch
from iflow.utils.generic import to_numpy


class TestClass():
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.N = 100
        self.dim = dynamics.dim

    def points_evolution(self):
        x0 = torch.ones(1, self.dim)
        trj_n = self.dynamics.generate_trj(x0, T=self.N, noise=False)
        trj_n_np = to_numpy(trj_n)

        fig, axs = plt.subplots(self.dim)
        for i in range(self.dim):
            axs[i].plot(trj_n_np[:, 0, i],'*')
        plt.show()

        plt.plot(trj_n_np[:, 0, 0], trj_n_np[:, 0, 1])
        plt.show()

    def noise_forward_evaluation(self):
        x0 = torch.ones(100, 3)
        trj_n = self.dynamics.generate_trj(x0, T=4*self.N, noise=True)
        trj_n_np = to_numpy(trj_n)

        fig, axs = plt.subplots(self.dim)
        for i in range(self.dim):
            for j in range(100):
                axs[i].plot(trj_n_np[:, j, i])
        plt.show()

        for j in range(100):
            plt.plot(trj_n_np[:, j, 0], trj_n_np[:, j, 1])
        plt.show()

    def noise_backward_evaluation(self):
        x0 = torch.ones(100, 3)
        trj_n = self.dynamics.generate_trj(x0, T=4*self.N, noise=True, reverse=True)
        trj_n_np = to_numpy(trj_n)

        fig, axs = plt.subplots(self.dim)
        for i in range(self.dim):
            for j in range(100):
                axs[i].plot(trj_n_np[:, j, i])
        plt.show()

        for j in range(100):
            plt.plot(trj_n_np[:, j, 0], trj_n_np[:, j, 1])
        plt.show()

    def conditional_prob_forward(self):
        step = 20
        x0 = torch.ones(1, self.dim)
        trj_n = self.dynamics.generate_trj(x0, T=self.N, noise=False)
        x0 = trj_n[:-step, 0 , :]
        x1 = trj_n[step:, 0, :]

        log_prob_x0_x1 = self.dynamics.conditional_log_prob(x0,x1, T=step, reverse=False)
        print('True Steps prob: {}'.format(torch.mean(log_prob_x0_x1)))
        log_prob_x0_x1 = self.dynamics.conditional_log_prob(x0,x1, T=1, reverse=False)
        print('Less Steps prob: {}'.format(torch.mean(log_prob_x0_x1)))
        log_prob_x0_x1 = self.dynamics.conditional_log_prob(x0,x1, T=50, reverse=False)
        print('More Steps prob: {}'.format(torch.mean(log_prob_x0_x1)))

    def forward_density(self):
        x0 = torch.randn(1, self.dim)
        tr_mu, tr_var = self.dynamics.generate_trj_density(x0, self.N, reverse=False)
        tr_mu = to_numpy(tr_mu)
        tr_var = to_numpy(tr_var)
        print(tr_mu.shape)
        print(tr_var.shape)

        fig, axs = plt.subplots(self.dim)
        for i in range(self.dim):
            l_trj = tr_mu[:, 0, i] - 3 * np.sqrt(tr_var[:, 0, i, i])
            h_trj = tr_mu[:, 0, i] + 3 * np.sqrt(tr_var[:, 0, i, i])

            t = np.linspace(0, tr_mu.shape[0], tr_mu.shape[0])
            axs[i].plot(t, tr_mu[:, 0, i])
            axs[i].fill_between(t, l_trj, h_trj, alpha=0.3)
        plt.show()

    def backward_density(self):
        x0 = torch.randn(1, self.dim)
        tr_mu, tr_var = self.dynamics.generate_trj_density(x0, self.N, reverse=True)
        tr_mu = to_numpy(tr_mu)
        tr_var = to_numpy(tr_var)
        print(tr_mu.shape)
        print(tr_var.shape)

        fig, axs = plt.subplots(self.dim)
        for i in range(self.dim):
            l_trj = tr_mu[:, 0, i] - 3 * np.sqrt(tr_var[:, 0, i, i])
            h_trj = tr_mu[:, 0, i] + 3 * np.sqrt(tr_var[:, 0, i, i])

            t = np.linspace(0, tr_mu.shape[0], tr_mu.shape[0])
            axs[i].plot(t, tr_mu[:, 0, i])
            axs[i].fill_between(t, l_trj, h_trj, alpha=0.3)
        plt.show()