import iflow.dataset.static_data as toy_data

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.distributions as tdist

import iflow.model as model

dim = 2
depth = 30

log_freq = 50
plot_interval = 50

batch_size = 100

data_name = 'moons'

# ######### FLOW MODEL ###########
def main_layer(dim):
    return model.flows.CouplingLayer(dim)

# def main_layer(dim):
#      return model.flows.NaiveScale(dim)

def construct_model(dim):
    chain = []
    for i in range(10):
        chain.append(main_layer(dim))
        chain.append(model.flows.RandomPermutation(dim))
        #chain.append(model.flows.LULinear(dim))
    chain.append(main_layer(dim))
    return model.SequentialFlow(chain)


def compute_loss(model):

    # load data
    x = toy_data.inf_train_gen(data_name, batch_size=batch_size)
    #x = np.random.randn(100,2)*5


    x = torch.from_numpy(x).type(torch.float32)
    zero = torch.zeros(x.shape[0], 1).to(x)

    # transform to z
    z, delta_logp = model(x, zero)

    # compute log q(z)
    dist = tdist.MultivariateNormal(loc=z.new_zeros(z.shape), covariance_matrix=torch.eye(dim))

    logpz = dist.log_prob(z)

    logpx = logpz  + delta_logp[:,0]
    print('Loss trj: pz {} , logJ {}'.format(torch.mean(logpz), torch.mean(delta_logp[:,0])))

    loss = -torch.mean(logpx)
    return loss

if __name__ == '__main__':
    ### Model Definition
    model = construct_model(dim)

    ########## Optimization ################
    params = list(model.parameters())

    optimizer = optim.Adamax(params, lr =0.001, weight_decay= 0.)
    #######################################

    best_loss = float('inf')
    for itr in range(0, 100000):

        optimizer.zero_grad()

        loss = compute_loss(model)

        loss.backward()
        optimizer.step()

        if itr % log_freq == 0:
            with torch.no_grad():
                model.eval()
                loss = compute_loss(model)
                print('Iter {:04d}  | Loss {})'.format(
                        itr, loss))
                model.train()

        if itr % plot_interval == 0:
            pass
            with torch.no_grad():
                model.eval()

                z_test = np.random.randn(5000,dim)
                #z_test = np.random.rand(5000,dim)
                z_test = torch.from_numpy(z_test).float()
                x_test = model(z_test, reverse=True)
                x_test = x_test.detach().cpu().numpy()

                plt.clf()
                fig = plt.figure(1)
                plt.hist2d(x_test[:,0], x_test[:,1],bins=200)
                plt.draw()
                plt.pause(0.001)

                model.train()




