import numpy as np
import similaritymeasures
import torch


def seds_metric(vel_ref_trj_l, vel_pred_tr_l, r=0.6, q=0.4, epsilon = 0.0001):

    n_trj = len(vel_ref_trj_l)

    error = 0
    for vel_pred_tr, vel_ref_trj in zip(vel_pred_tr_l,vel_ref_trj_l):


        total_value = 0
        for i in range(0,vel_pred_tr.shape[0]):
            norm_pred = np.sum(np.abs(vel_pred_tr[i,0]))
            norm_real = np.sum(np.abs(vel_ref_trj[i,0]))

            vel_p = vel_pred_tr[i,:]
            vel_r = vel_ref_trj[i,:]

            dist_x = vel_r - vel_p
            dist_mean = np.matmul(dist_x.T,dist_x)

            q_value = dist_mean/(norm_pred*norm_real+epsilon)

            dist_ang = np.matmul(vel_r.T,vel_p)
            r_value = (1 - dist_ang/((norm_pred*norm_real+epsilon)))**2
            value_t = r*r_value + q*q_value

            total_value += np.sqrt(value_t)

        total_value_n = total_value/vel_pred_tr.shape[0]
        error += total_value_n

    error_n = error/ n_trj
    return error_n


def squared_mean_error(ref_trj_l, pred_tr_l):

    n_trj = len(pred_tr_l)

    error = 0
    for pred_tr, ref_trj in zip(pred_tr_l, ref_trj_l):

        l2_norm_tr = 0
        length = pred_tr.shape[0]
        for i in range(0,length):
            x_pred = pred_tr[i,:]
            x_real = ref_trj[i,:]

            dist = x_pred - x_real
            l2_norm = np.linalg.norm(dist)
            l2_norm_tr += l2_norm

        l2_norm_tr_n = l2_norm_tr/length

        error += l2_norm_tr_n

    error_n = error/ n_trj
    return error_n


def area(X):
    n_points = len(X)

    d = 0
    for i in range(0,n_points):
        x0 = X[i]
        if i== (n_points-1):
            x1 = X[0]
        else:
            x1 = X[i+1]

        d_point = x0[0]*x1[1] - x0[1]*x1[0]
        d += d_point

    A = np.abs(d)/2
    return A


def mean_swept_error(ref_trj_l ,pred_tr_l):
    n_trj = len(pred_tr_l)

    error = 0
    for pred_tr, ref_trj in zip(pred_tr_l, ref_trj_l):
        ar = 0
        lenght = ref_trj.shape[0]
        for i in range(0,lenght-1):
            X = []
            X.append(pred_tr[i, :])
            X.append(pred_tr[i + 1, :])
            X.append(ref_trj[i+1,:])
            X.append(ref_trj[i,:])

            A = area(X)
            ar += A

        error += ar

    error_n = error / n_trj
    return error_n


def area_between_error(ref_trj_l, pred_tr_l):
    n_trj = len(pred_tr_l)

    error = 0
    for pred_tr, ref_trj in zip(pred_tr_l, ref_trj_l):
        area = similaritymeasures.area_between_two_curves(ref_trj,pred_tr)
        error += area
    error_n = error/n_trj
    return error_n


def mean_frechet_error(ref_trj_l, pred_tr_l):
    n_trj = len(pred_tr_l)

    error = 0
    for pred_tr, ref_trj in zip(pred_tr_l, ref_trj_l):
        frecht_d = similaritymeasures.frechet_dist(ref_trj,pred_tr)
        error += frecht_d
    error_n = error/n_trj
    return error_n


def dtw_distance(ref_trj_l, pred_tr_l):
    n_trj = len(pred_tr_l)

    error = 0
    for pred_tr, ref_trj in zip(pred_tr_l, ref_trj_l):
        dtw_dist, d = similaritymeasures.dtw(ref_trj,pred_tr)
        error+= dtw_dist
    error_n = error/n_trj
    return error_n


def iros_evaluation(val_trajs, iflow, device):
    dim = val_trajs[0].shape[1]

    ### Generate Predicted Trajectories ###
    predicted_trajs = []
    for trj in val_trajs:
        n_trj = trj.shape[0]
        y0 = trj[0, :]
        y0 = torch.from_numpy(y0[None, :]).float().to(device)
        traj_pred = iflow.generate_trj( y0, T=n_trj)
        traj_pred = traj_pred.detach().cpu().numpy()
        predicted_trajs.append(traj_pred)

    print('#### IROS EVALUATION ####')
    error_mean = squared_mean_error(val_trajs, predicted_trajs)
    print('The mean Error is: {}'.format(error_mean))
    error_frechet = mean_frechet_error(val_trajs, predicted_trajs)
    print('The Frechet Distance is: {}'.format(error_frechet))
    error_dtw = dtw_distance(val_trajs, predicted_trajs)
    print('The DTW Distance is: {}'.format(error_dtw))
    error_swept = mean_swept_error(val_trajs, predicted_trajs)
    print('The Swept Area error is: {}'.format(error_swept))
    error_area = area_between_error(val_trajs, predicted_trajs)
    print('The Area error is: {}'.format(error_area))
    print('##########################')

