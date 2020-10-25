import os, sys
import torch
import torch.optim as optim
from iflow.dataset import iros_dataset
from torch.utils.data import DataLoader
from iflow.utils import makedirs
from iflow import model
from iflow.trainers import cycle_dynamics_train
from iflow.utils.generic import to_torch
import numpy as np
from iflow.visualization import visualize_vector_field, visualize_2d_generated_trj, save_vector_field
from iflow.test_measures.log_likelihood import cycle_log_likelihood


percentage = .99
batch_size = 100
depth = 15
## optimization ##
lr = 0.001
weight_decay = 0.
## training variables ##
nr_epochs = 1000
filename = 'RShape'
save_folder = 'experiments'

dir_save = os.path.join(os.path.dirname(__file__),save_folder)
makedirs(dir_save)

######### GPU/ CPU #############
#device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

#### Invertible Flow model #####
def main_layer(dim):
    return  model.ResNetCouplingLayer(dim)


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
    data = iros_dataset.IROS(filename=filename)
    dim = data.dim
    T_period = (2*np.pi)/data.w
    params = {'batch_size': batch_size, 'shuffle': True}
    dataloader = DataLoader(data.dataset, **params)
    ######### Model #########
    lsd = model.LinearLimitCycle(dim, device, dt=data.dt, T_period=T_period)
    flow = create_flow_seq(dim, depth)
    iflow = model.ContinuousDynamicFlow(dynamics=lsd, model=flow, dim=dim).to(device)
    ########## Optimization ################
    params = list(flow.parameters()) + list(lsd.parameters())
    optimizer = optim.Adamax(params, lr = lr, weight_decay= weight_decay)
    #######################################
    for i in range(nr_epochs):
        # Training
        for local_x, local_y in dataloader:
            dataloader.dataset.set_step()
            optimizer.zero_grad()
            loss = cycle_dynamics_train(iflow, local_x, local_y)
            loss.backward(retain_graph=True)
            optimizer.step()

        ## Validation ##
        if i%1 == 0:
            with torch.no_grad():
                iflow.eval()

                #visualize_2d_generated_trj(data.train_data, iflow, device, fig_number=2)
                #visualize_vector_field(data.train_data, iflow, device, fig_number=3)

                step = 20
                trj = data.train_data[0]
                trj_x0 = to_torch(trj[:-step,:], device)
                trj_x1 = to_torch(trj[step:,:], device)
                phase = to_torch(data.train_phase_data[0][:-step], device)
                cycle_log_likelihood(trj_x0, trj_x1, phase, step, iflow, device)

                fig_name = filename + str(i) + '.png'
                save_filename = os.path.join(dir_save, fig_name)
                save_vector_field(data.train_data, iflow, device, save_fig=save_filename)












