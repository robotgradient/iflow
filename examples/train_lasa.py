import os, sys, time
import torch
import torch.optim as optim
from iflow.dataset import lasa_dataset
from torch.utils.data import DataLoader
from iflow import model
from iflow.trainers import goto_dynamics_train
from iflow.utils import to_numpy, to_torch
from iflow.visualization import visualize_latent_distribution, visualize_vector_field, visualize_2d_generated_trj
from iflow.test_measures import log_likelihood, iros_evaluation


percentage = .99
batch_size = 100
depth = 10
## optimization ##
lr = 0.001
weight_decay = 0.
## training variables ##
nr_epochs = 1000
## filename ##
filename = 'Sshape'

######### GPU/ CPU #############
#device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

#### Invertible Flow model #####
def main_layer(dim):
    return  model.CouplingLayer(dim)


def create_flow_seq(dim, depth):
    chain = []
    for i in range(depth):
        chain.append(main_layer(dim))
        chain.append(model.RandomPermutation(dim))
        chain.append(model.LULinear(dim))
    chain.append(main_layer(dim))
    return model.SequentialFlow(chain)


if __name__ == '__main__':
    ########## Data Loading #########
    data = lasa_dataset.LASA(filename = filename)
    dim = data.dim
    params = {'batch_size': batch_size, 'shuffle': True}
    dataloader = DataLoader(data.dataset, **params)
    ######### Model #########
    dynamics = model.TanhStochasticDynamics(dim, dt=0.01, T_to_stable=2.5)
    #dynamics = model.LinearStochasticDynamics(dim, dt=0.01, T_to_stable=2.5)
    flow = create_flow_seq(dim, depth)
    iflow = model.ContinuousDynamicFlow(dynamics=dynamics, model=flow, dim=dim).to(device)
    ########## Optimization ################
    params = list(flow.parameters()) + list(dynamics.parameters())
    optimizer = optim.Adamax(params, lr = lr, weight_decay= weight_decay)
    #######################################
    for i in range(nr_epochs):
        ## Training ##
        for local_x, local_y in dataloader:
            dataloader.dataset.set_step()
            optimizer.zero_grad()
            loss = goto_dynamics_train(iflow, local_x, local_y)
            loss.backward(retain_graph=True)
            optimizer.step()

        ## Validation ##
        if i%10 == 0:
            with torch.no_grad():
                iflow.eval()

                visualize_2d_generated_trj(data.train_data, iflow, device, fig_number=2)
                visualize_latent_distribution(data.train_data, iflow, device, fig_number=1)
                visualize_vector_field(data.train_data, iflow, device, fig_number=3)
                iros_evaluation(data.train_data, iflow, device)

                ## Prepare Data ##
                step = 20
                trj = data.train_data[0]
                trj_x0 = to_torch(trj[:-step,:], device)
                trj_x1 = to_torch(trj[step:,:], device)
                log_likelihood(trj_x0, trj_x1, step, iflow, device)
                print('The Variance of the latent dynamics are: {}'.format(torch.exp(iflow.dynamics.log_var)))
                print('The Velocity of the latent dynamics are: {}'.format(iflow.dynamics.Kv[0,0]))












