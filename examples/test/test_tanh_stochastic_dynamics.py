import matplotlib.pyplot as plt
import numpy as np
import torch
from iflow.utils.generic import to_numpy

from iflow.model.dynamics.tanh_stochastic_dynamics import TanhStochasticDynamics


if __name__ == '__main__':
    lsd = TanhStochasticDynamics(3, dt=0.01 ,T_to_stable=2.5)
    N = lsd.N_to_stable
    print('number of samples to Stability {}'.format(N))

    x0 = torch.rand(10,3)
    x1 = lsd.step_forward(x0)

    if x1.shape==torch.Size([10, 3]):
        print('True')

    x1, logJ = lsd.forward(x0)
    print(x1, logJ)

    ### GENERATED POINTS ###
    x0 = torch.ones(1,3)
    trj_n = lsd.generate_trj(x0, T=N, noise=False)
    trj_n_np = to_numpy(trj_n)

    plt.plot(trj_n_np[:,0],'*')
    plt.show()

    ### Evolve N steps
    xn = lsd.evolve(x0, T=20)
    print('xn shape {}'.format(xn.shape))

    ### Evolve N steps Trajectory
    x0 = torch.ones(100,3)
    trj_n = lsd.generate_trj(x0, T=N , noise=True)
    trj_n_np = to_numpy(trj_n)
    for i in range(10):
        plt.plot(trj_n_np[:,i,0])
    plt.show()

    trj_n = lsd.generate_trj(x0, T=N , noise=False)
    x0 = torch.ones(100,3)*trj_n[-1,0,:]
    trj_n = lsd.generate_trj(x0, T=N , noise=True, reverse=True)
    trj_n_np = to_numpy(trj_n)
    for i in range(100):
        plt.plot(trj_n_np[:,i,0])
    plt.show()

    ### Distribution Forward
    x0 = torch.ones(1, 3)

    dist = lsd.conditional_distribution(x0, N )
    print('mean: {} , var: {}'.format(dist.loc , dist.variance))

    x0 = torch.ones(1,3)*0.01
    dist_b = lsd.conditional_distribution(x0, N , reverse=True)
    print('Backward. mean: {} , var: {}'.format(dist.loc , dist.variance))

    n_samples = 100

    samples = dist.sample((n_samples,))
    samples = to_numpy(samples)

    samples_b = dist_b.sample((n_samples,))
    samples_b = to_numpy(samples_b)


    ### Evolve N steps Trajectory
    x0 = torch.ones(30,3)*0.01
    trj_n = lsd.generate_trj(x0, T=N , noise=True, reverse=True)
    trj_n_np = to_numpy(trj_n)
    for i in range(10):
        plt.plot(trj_n_np[:,i,0])
    x = np.ones(n_samples)*N
    plt.plot(x,samples_b[:,0,0], '*')
    plt.show()

    x0 = torch.ones(30,3)
    trj_n = lsd.generate_trj(x0, T=N , noise=True)
    trj_n_np = to_numpy(trj_n)
    for i in range(10):
        plt.plot(trj_n_np[:,i,0])

    x = np.ones(n_samples)*N
    plt.plot(x,samples[:,0,0], '*')
    plt.show()


    ### Noise Trajectory ###
    x0 = torch.ones(1,3)
    tr_mu, tr_var = lsd.generate_trj_density(x0, N , reverse=False)
    print(tr_mu.shape)
    print(tr_var.shape)
    tr_mu = to_numpy(tr_mu)
    tr_var = to_numpy(tr_var)

    l_trj = tr_mu[:,0,0] - 3*np.sqrt(tr_var[:,0,0,0])
    h_trj = tr_mu[:,0,0] + 3*np.sqrt(tr_var[:,0,0,0])


    t = np.linspace(0, tr_mu.shape[0], tr_mu.shape[0])
    plt.plot(t,tr_mu[:,0,0] )
    plt.fill_between(t, l_trj, h_trj, alpha=0.3)


    ### Random TRJS ####
    x0 = torch.ones(30, 3)
    trj_n = lsd.generate_trj(x0, T=N, noise=True, reverse=False)
    trj_n_np = to_numpy(trj_n)
    for i in range(10):
        plt.plot(trj_n_np[:, i, 0])
    plt.show()



    x0 = torch.ones(1,3)*0.01
    tr_mu, tr_var = lsd.generate_trj_density(x0, N , reverse=True)
    print(tr_mu.shape)
    print(tr_var.shape)
    tr_mu = to_numpy(tr_mu)
    tr_var = to_numpy(tr_var)

    l_trj = tr_mu[:,0,0] - 3*np.sqrt(tr_var[:,0,0,0])
    h_trj = tr_mu[:,0,0] + 3*np.sqrt(tr_var[:,0,0,0])


    t = np.linspace(0, tr_mu.shape[0], tr_mu.shape[0])
    plt.plot(t,tr_mu[:,0,0] )
    plt.fill_between(t, l_trj, h_trj, alpha=0.3)
    ### Random TRJS ####
    x0 = torch.ones(30, 3)*0.01
    trj_n = lsd.generate_trj(x0, T=N, noise=True, reverse=True)
    trj_n_np = to_numpy(trj_n)
    for i in range(10):
        plt.plot(trj_n_np[:, i, 0])
    plt.show()


