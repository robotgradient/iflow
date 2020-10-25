import torch


def goto_dynamics_train(iflow, x, y):
    ## Separate Data ##
    y0 = x
    y1 = y[0]
    step = y[1][0]
    yN = y[2]
    t = y[3]
    ## Evolve dynamics backwards ##
    x_0, log_det_J_x0 = iflow(y0)
    x_1, log_det_J_x1 = iflow(y1)
    p_x0_x1 = iflow.dynamics.conditional_distribution(x_1, T=step, reverse=True)
    log_p_z0 = p_x0_x1.log_prob(x_0)
    loss_trj = log_p_z0 + log_det_J_x0.squeeze()

    ########## Last step #############
    yN = yN[:1,:]
    x_n, log_det_J_xn = iflow(yN)
    log_p_xn = iflow.dynamics.compute_stable_log_px(x_n)
    loss_end = log_p_xn + log_det_J_xn.squeeze()

    #### Complete Loss is composed between the stable loss and the trajectory loss
    loss_total = torch.mean(loss_trj) + torch.mean(loss_end)
    return -loss_total


def cycle_dynamics_train(iflow, x, y):
    ## Separate Data ##
    y0 = x
    y1 = y[0]
    step = y[1][0]
    phase = y[2]
    ## Evolve dynamics forward ##
    x_0, log_det_J_x0 = iflow(y0)
    x_1, log_det_J_x1 = iflow(y1)

    ### Forward Conditioning ###
    log_p_z1 = iflow.dynamics.cartesian_cond_log_prob(x_0, x_1, T=step)
    log_trj = log_p_z1 + log_det_J_x1.squeeze()

    ### Stable Point ###
    log_p_z0 = iflow.dynamics.stable_log_prob(x_0, ref_phase=phase)
    log_stable = log_p_z0 + log_det_J_x0.squeeze()

    log_total = torch.mean(log_stable) + torch.mean(log_trj)
    return -log_total



