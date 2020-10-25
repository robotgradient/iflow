import torch


def log_likelihood(val_data_y0, val_data_y1, step, iflow, device):
    x_0, log_det_J_x0 = iflow(val_data_y0)
    x_1, _ = iflow(val_data_y1)
    p_x0_x1 = iflow.dynamics.conditional_distribution(x_1, T=step, reverse=True)
    log_p_z0 = p_x0_x1.log_prob(x_0)
    log_likeli = log_p_z0 + log_det_J_x0.squeeze()
    print('Validation data Log Likelihood: {} | log pz : {} | log detJ : {}'.format(torch.mean(log_likeli), torch.mean(log_p_z0), torch.mean(log_det_J_x0[:, 0])))


def cycle_log_likelihood(val_data_y0, val_data_y1,phase, step, iflow, device):
    ## Separate Data ##
    y0 = val_data_y0
    y1 = val_data_y1
    ## Evolve dynamics forward ##
    x_0, log_det_J_x0 = iflow(y0)
    x_1, log_det_J_x1 = iflow(y1)

    ### Forward Conditioning ###
    log_p_z1 = iflow.dynamics.cartesian_cond_log_prob(x_0, x_1, T=step)
    log_trj = log_p_z1 + log_det_J_x1.squeeze()

    ### Stable Point ###
    log_p_z0 = iflow.dynamics.stable_log_prob(x_0, ref_phase=phase)
    log_stable = log_p_z0 + log_det_J_x0.squeeze()

    print('Validation data Conditional Log Likelihood : {} | log pz : {} | log detJ : {}'.format(torch.mean(log_trj), torch.mean(log_p_z1), torch.mean(log_det_J_x1[:, 0])))
    print('Validation data Stable Log Likelihood : {} | log pz : {} | log detJ : {}'.format(torch.mean(log_stable), torch.mean(log_p_z0), torch.mean(log_det_J_x0[:, 0])))
