import torch
import torch.distributions as tdist
from iflow.utils.math import kronecker_product
from iflow.model.dynamics.generic_dynamic import DynamicModel


class LinearStochasticDynamics(DynamicModel):

    def __init__(self, dim, device=None, dt = 0.001, requires_grad=True, T_to_stable=1.):
        super().__init__(dim, device, dt, requires_grad)

        ## Deterministic Dynamics
        self.N_to_stable = int(T_to_stable/dt)

        ## Set the dynamics: N STEPS
        _x0 = 1.
        _xn = 0.01
        _adt_1 = _xn**(1/self.N_to_stable)

        _a = (_adt_1 - 1)/dt
        self.A = nn.Parameter(torch.eye(dim)*_a).to(device).requires_grad_(requires_grad)

        ## Variance in Linear Dynamics
        _std = 0.1
        self.log_var = nn.Parameter(torch.ones(dim) * np.log(_std ** 2)).to(device).requires_grad_(requires_grad)

    @property
    def var(self):
        return torch.diag(torch.exp(self.log_var))

    def velocity(self, x):
        return torch.matmul(self.A, x.T).T

    def first_Taylor_dyn(self, x):
        return torch.cat(x.shape[0]*[self.A[None,...]])

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

        A = self.A * self.dt + torch.eye(self.A.shape[0]).to(x_n)

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











