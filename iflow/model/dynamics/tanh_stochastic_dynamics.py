import numpy as np
import torch
import torch.nn as nn
from iflow.utils.math import kronecker_product
import torch.distributions as tdist

from iflow.model.dynamics.generic_dynamic import DynamicModel


def inverse_tanh(x):
    return 0.2 * torch.log((1+x)/(1-x))

class TanhStochasticDynamics(DynamicModel):
    r"""
    Tanh Dynamics. The dynamics will evolve with constant velocity until the robot is close to the steady-state point.
    Then, the dynamics will evolve like a Linear System.

    .. math::
        \dot{x} = - K_v \tanh (K_x x)

    """
    def __init__(self, dim, device=None, dt=0.01, requires_grad=True, T_to_stable=1.):
        super().__init__(dim, device, dt, requires_grad)

        ## Deterministic Dynamics
        self.N_to_stable = int(T_to_stable / dt)


        ## We can approximate the dynamics between 1/Kp to infinity as constant velocity Kp
        kp = 100
        x_n = 1/kp
        x_0 = 1.
        kv = (x_n-x_0)/(self.N_to_stable * self.dt)

        ## Set the dynamics: N STEPS
        self.Kv = nn.Parameter(torch.eye(dim) * kv).to(device).requires_grad_(requires_grad)
        self.Kp = torch.eye(dim) * kp

        ## Variance in Linear Dynamics
        _std = 0.1
        self.log_var = nn.Parameter(torch.ones(dim)*np.log(_std ** 2)).to(device).requires_grad_(requires_grad)

    @property
    def var(self):
        return torch.diag(torch.exp(self.log_var))

    def velocity(self, x):
        return torch.matmul(self.Kv, torch.tanh(torch.matmul(self.Kp, x.T))).T

    def first_Taylor_dyn(self, x):
        ## Approximate the velocity with respect to the constant velocity case
        ## todo: Approximate the taylor approximation given the current state (Constant vel / Linear vel)
        return torch.cat(x.shape[0]*[self.Kv[None,...]])

    def compute_stable_log_px(self, x_n):
        r'''
        compute_stable_log_px will provide the density p(x_n), when n->infinity. We assume stable dynamics
        mu_inf  = torch.zeros(dim)

            .. math::
        var_inf =  \sum_{i=1}^{inf} A^{i} var (A.T)^{i} = \frac{var}{I - AA.T}


        The variance can be solved analytically: P - APA.T = VAR --> P = (I - conj(A) (x) A ) vec(X) = vec(Q)
        vec(ABC) = (C.T (X) A)vec(B)

        :return:
        log_px(t-> infty)
        '''

        A = self.Kv*self.dt + torch.eye(self.Kv.shape[0]).to(x_n)

        I_n2 = torch.eye(self.dim ** 2).to(x_n)
        AXA = kronecker_product(A, A)

        stable_var = self.var * self.dt
        var_vec = stable_var.reshape(-1)

        I_AXA = I_n2 - AXA

        I_AXA_inv = I_AXA.inverse()

        var_inf_vec = torch.matmul(I_AXA_inv, var_vec)
        var_inf = var_inf_vec.reshape((self.dim, self.dim))

        mu_inf = torch.zeros_like(x_n)

        stable_dist = tdist.MultivariateNormal(loc=mu_inf, covariance_matrix=var_inf)
        log_px_n = stable_dist.log_prob(x_n)

        return log_px_n














