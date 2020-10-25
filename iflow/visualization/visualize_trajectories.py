import torch
import matplotlib.pyplot as plt


def visualize_trajectories(val_trajs, iflow, device, fig_number=1):
    dim = val_trajs[0].shape[1]

    plt.figure(fig_number, figsize=(20, int(10 * dim))).clf()
    fig, axs = plt.subplots(dim, 1, num=fig_number)

    for trj in val_trajs:
        n_trj = trj.shape[0]
        y0 = trj[0, :]
        y0 = torch.from_numpy(y0[None, :]).float().to(device)
        traj_pred = iflow.generate_trj( y0, T=n_trj)
        traj_pred = traj_pred.detach().cpu().numpy()

        for j in range(dim):
            axs[j].plot(trj[:,j],'b')
            axs[j].plot(traj_pred[:,j],'r')
    plt.draw()
    plt.pause(0.001)


def visualize_2d_generated_trj(val_trj, iflow, device, fig_number=1):
    n_trj = len(val_trj)
    dim = val_trj[0].shape[-1]

    plt.figure(fig_number).clf()
    fig = plt.figure(figsize=(15, 15), num=fig_number)
    for i in range(len(val_trj)):
        y_0 = torch.from_numpy(val_trj[i][:1, :]).float().to(device)
        trj_y = iflow.generate_trj(y_0, T=val_trj[i].shape[0])
        trj_y = trj_y.detach().cpu().numpy()

        plt.plot(trj_y[:,0], trj_y[:,1], 'g')
        plt.plot(val_trj[i][:,0], val_trj[i][:,1], 'b')
    plt.draw()
    plt.pause(0.001)

