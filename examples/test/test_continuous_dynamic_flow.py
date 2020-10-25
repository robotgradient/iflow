import numpy as np
import torch
import matplotlib.pyplot as plt


from iflow import model


## Arguments ##
depth = 70

######### FLOW MODEL ###########
def main_layer(dim):
    return model.flows.CouplingLayer(dim)

def construct_model(dim):
    chain = []
    for i in range(depth):
        chain.append(main_layer(dim))
        chain.append(model.flows.RandomPermutation(dim))
        chain.append(model.flows.LULinear(dim))
    chain.append(main_layer(dim))
    return model.SequentialFlow(chain)


if __name__ == '__main__':

    ### Stable Dynamic Flows ###
    lsd = model.dynamics.LinearStochasticDynamics(dim=2)
    flows = construct_model(dim=2)
    cont_flow = model.ContinuousDynamicFlow(model=flows, dynamics=lsd)

    ## Generate random data
    x0 = torch.rand(10,2)

    z0, logJ = cont_flow(x0)
    print(logJ)
    ## The more it compress the data, the bigger the logJ



    x0_np = x0.numpy()
    z0_np = z0.detach().numpy()

    plt.plot(x0_np[:,0], x0_np[:,1], '*')
    plt.plot(z0_np[:,0], z0_np[:,1], '*')
    plt.show()

    ##








