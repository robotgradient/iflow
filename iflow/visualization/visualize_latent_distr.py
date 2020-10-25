import torch
import numpy as np
from iflow.utils.generic import to_numpy, to_torch
import matplotlib.pyplot as plt


def visualize_latent_distribution(val_trj, iflow, device, fig_number=1):
    n_trj = len(val_trj)
    dim = val_trj[0].shape[-1]
    ## Store the latent trajectory in a list ##
    val_z_trj= []
    val_mu_trj = []
    val_var_trj = []

    plt.figure(fig_number, figsize=(20,int(10*dim))).clf()
    fig, axs = plt.subplots(dim, 1, num=fig_number)
    for i in range(len(val_trj)):
        y_trj = to_torch(val_trj[i],device)
        z_trj, _ = iflow(y_trj)
        z_trj = to_numpy(z_trj)
        val_z_trj.append(z_trj)

        z0 = to_torch(z_trj[0,:],device)
        trj_mu, trj_var = iflow.dynamics.generate_trj_density(z0[None,:], T = val_z_trj[i].shape[0])
        val_mu_trj.append(to_numpy(trj_mu))
        val_var_trj.append(to_numpy(trj_var))

        for j in range(val_trj[i].shape[-1]):
            t = np.linspace(0,val_z_trj[i].shape[0], val_z_trj[i].shape[0])
            axs[j].plot(t,val_z_trj[i][:,j])
            l_trj = val_mu_trj[i][:,0,j] - np.sqrt(val_var_trj[i][:,0, j, j] )
            h_trj = val_mu_trj[i][:,0,j]  + np.sqrt(val_var_trj[i][:,0, j, j] )
            axs[j].fill_between(t,l_trj, h_trj, alpha=0.1)

    plt.draw()
    plt.pause(0.001)


def visualize_vector_field(val_trj, iflow, device, fig_number=1):
    _trajs = np.zeros((0, 2))
    for trj in val_trj:
        _trajs = np.concatenate((_trajs, trj),0)
    min = _trajs.min(0) - 0.5
    max = _trajs.max(0) + 0.5

    n_sample = 100

    x = np.linspace(min[0], max[0], n_sample)
    y = np.linspace(min[1], max[1], n_sample)

    xy = np.meshgrid(x, y)
    h = np.concatenate(xy[0])
    v = np.concatenate(xy[1])
    hv = torch.Tensor(np.stack([h, v]).T).float()
    if device is not None:
        hv = hv.to(device)

    hv_t1 = iflow.evolve(hv, T=3)
    hv = hv.detach().cpu().numpy()
    hv_t1 = hv_t1.detach().cpu().numpy()

    vel = (hv_t1 - hv)

    vel_x = np.reshape(vel[:, 0], (n_sample, n_sample))
    vel_y = np.reshape(vel[:, 1], (n_sample, n_sample))
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2)
    speed = speed/np.max(speed)

    fig = plt.figure(fig_number, figsize=(10, 10))
    plt.clf()
    ax = plt.gca()

    plt.streamplot(xy[0], xy[1], vel_x, vel_y, color=speed, density=[0.5, 1])
    for i in range(len(val_trj)):
        plt.plot(val_trj[i][:,0], val_trj[i][:,1], 'b')
    plt.draw()
    plt.pause(0.05)


def save_vector_field(val_trj, iflow, device, save_fig, fig_number=1):
    _trajs = np.zeros((0, 2))
    for trj in val_trj:
        _trajs = np.concatenate((_trajs, trj),0)
    min = _trajs.min(0) - 0.5
    max = _trajs.max(0) + 0.5

    n_sample = 100

    x = np.linspace(min[0], max[0], n_sample)
    y = np.linspace(min[1], max[1], n_sample)

    xy = np.meshgrid(x, y)
    h = np.concatenate(xy[0])
    v = np.concatenate(xy[1])
    hv = torch.Tensor(np.stack([h, v]).T).float()
    if device is not None:
        hv = hv.to(device)

    hv_t1 = iflow.evolve(hv, T=3)
    hv = hv.detach().cpu().numpy()
    hv_t1 = hv_t1.detach().cpu().numpy()

    vel = (hv_t1 - hv)

    vel_x = np.reshape(vel[:, 0], (n_sample, n_sample))
    vel_y = np.reshape(vel[:, 1], (n_sample, n_sample))
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2)
    speed = speed/np.max(speed)

    fig = plt.figure(fig_number, figsize=(10, 10))
    plt.clf()
    ax = plt.gca()

    plt.streamplot(xy[0], xy[1], vel_x, vel_y, color=speed, density=[0.5, 1])
    for i in range(len(val_trj)):
        plt.plot(val_trj[i][:,0], val_trj[i][:,1], 'b')

    plt.savefig(save_fig)
