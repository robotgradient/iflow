import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.distributions as tdist
import iflow.dataset.static_data as toy_data


class TestClass():
    def __init__(self, model, dataname='moons'):
        self.model = model
        ########## Optimization ################
        self.params = list(model.parameters())
        self.optimizer = optim.Adamax(self.params, lr=0.001, weight_decay=0.)
        self.log_freq = 50
        self.plot_interval = 50
        self.batch_size = 100
        self.data_name = dataname

    def compute_loss(self):
        # load data
        x = toy_data.inf_train_gen(self.data_name, batch_size=self.batch_size)
        x = torch.from_numpy(x).type(torch.float32)
        zero = torch.zeros(x.shape[0], 1).to(x)
        # transform to z
        z, delta_logp = self.model(x, zero)
        # compute log q(z)
        dist = tdist.MultivariateNormal(loc=z.new_zeros(z.shape), covariance_matrix=torch.eye(2))
        logpz = dist.log_prob(z)
        logpx = logpz + delta_logp[:, 0]
        #print('Loss trj: pz {} , logJ {}'.format(torch.mean(logpz), torch.mean(delta_logp[:, 0])))
        loss = -torch.mean(logpx)
        return loss

    def train(self):
        for itr in range(0, 100000):
            self.optimizer.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            self.optimizer.step()
            if itr % self.log_freq == 0:
                with torch.no_grad():
                    self.model.eval()
                    loss = self.compute_loss()
                    print('Iter {:04d}  | Loss {})'.format(
                        itr, loss))
                    self.model.train()

            if itr % self.plot_interval == 0:
                pass
                with torch.no_grad():
                    self.model.eval()

                    z_test = np.random.randn(5000, 2)
                    # z_test = np.random.rand(5000,dim)
                    z_test = torch.from_numpy(z_test).float()
                    x_test = self.model(z_test, reverse=True)
                    x_test = x_test.detach().cpu().numpy()

                    plt.clf()
                    fig = plt.figure(1)
                    plt.hist2d(x_test[:, 0], x_test[:, 1], bins=200)
                    plt.draw()
                    plt.pause(0.001)

                    self.model.train()
