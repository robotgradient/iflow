import numpy as np
import torch
import torch.nn as nn

from iflow.model.dynamics.generic_dynamic import LimitCycleDynamicModel

from iflow.utils.math import block_diag


class LinearLimitCycle(LimitCycleDynamicModel):
    ''' The Dynamic model is composed of two parts. A limit '''
    def __init__(self, dim, device=None, dt = 0.001, requires_grad=True, T_to_stable=1., T_period= 1.):
        super().__init__(dim, device, dt, requires_grad)

        #### Deterministic Dynamics ####
        self.N_to_stable = int(T_to_stable/dt)
        self.N_to_period = int(T_period/dt)

        self.T_period = T_period
        self.T_to_stable = T_to_stable

        ## Set the dynamics: N STEPS ##
        _x0 = 1.
        _xn = 0.01
        _adt_1 = _xn**(1/self.N_to_stable)
        _a = (_adt_1 - 1)/dt
        A = torch.eye(dim-2)*_a

        ## Limit Cycle Velocities ##
        w = -(2*np.pi)/(self.N_to_period*dt)

        self.r_des = nn.Parameter(torch.ones(1),requires_grad=False)
        self.v_r = nn.Parameter(torch.ones(1)*10,requires_grad=False)
        self.w = nn.Parameter(torch.ones(1)*w,requires_grad=False)
        self.A = nn.Parameter(A*10,requires_grad=False)

        ## Variance in Linear Dynamics
        _std = 0.1
        self.log_var = nn.Parameter(torch.ones(dim)*np.log(_std ** 2)).to(device).requires_grad_(requires_grad)

    @property
    def var(self):
        return torch.diag(torch.exp(self.log_var))

    def forward(self, x, logpx=None, reverse=False):
        if not reverse:
            y = self.transform(x, reverse=reverse)
            log_abs_det_J = torch.log(y[:,0])
        else:
            y = self.transform(x, reverse=reverse)
            log_abs_det_J = -torch.log(x[:,0])

        if logpx is None:
            return y, logpx
        else:
            return y, logpx + log_abs_det_J.unsqueeze(1)

    def transform(self, x, reverse=False):
        y = x.clone()
        ## reverse to cartesian // forward to polar ##
        if reverse:
            y[..., 0] = x[..., 0] * torch.cos(x[..., 1])
            y[..., 1] = x[..., 0] * torch.sin(x[..., 1])
        else:
            r = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
            theta = torch.atan2(x[..., 1], x[..., 0])
            y[..., 0] = r
            y[..., 1] = theta
        return y

    def velocity(self, x):
        vel_r = -self.v_r*(x[:,0] - self.r_des)
        vel_theta = self.w*torch.ones(x.shape[0]).to(x)
        if self.A.shape[0]!=0:
            vel_z = torch.matmul(self.A, x[:,2:].T).T
            return torch.cat((vel_r[:,None], vel_theta[:,None], vel_z),1)
        else:
            return torch.cat((vel_r[:,None], vel_theta[:,None]),1)

    def first_Taylor_dyn(self, x):
        vel_r =  torch.cat(x.shape[0]*[-self.v_r.unsqueeze(0)[None,...]],0)
        vel_theta = torch.cat(x.shape[0]*[torch.zeros(1,1)[None,...]],0)
        vel_z = torch.cat(x.shape[0]*[self.A[None,...]],0)
        l_vel = [vel_r,vel_theta,vel_z]
        vel_mat = block_diag(l_vel)
        return vel_mat





