import torch
import torch.nn as nn
import torch.distributions as tdist
from iflow.densities import AngleNormal


class DynamicModel(nn.Module):
    def __init__(self, dim, device=None, dt = 0.01, requires_grad=True):
        super().__init__()
        self.dim = dim
        self.device = device
        self.dt = dt

    def forward(self, x, logp=None, reverse=False):
        return x, logp

    def backward(self, z, logp=None):
        return z, logp

    def velocity(self,x):
        raise NotImplementedError('Velocity must be Implemented in the inherited Method')

    def first_Taylor_dyn(self, x):
        raise NotImplementedError('first_Taylor_dyn must be Implemented in the inherited Method')

    def step_forward(self, xt0, noise=False):
        vel = self.velocity(xt0)

        mu = vel * self.dt + xt0
        if noise == True:
            var_step = self.var * self.dt
            mv_dist = tdist.MultivariateNormal(mu, var_step)
            xt1 = mv_dist.rsample()
        else:
            xt1 = mu
        return xt1

    def step_backwards(self, xt1, noise=False):
        vel = -self.velocity(xt1)

        mu_b = vel * self.dt + xt1
        if noise == True:
            var_b = self.var * self.dt
            mv_dist = tdist.MultivariateNormal(mu_b, var_b)
            xt0 = mv_dist.rsample()
        else:
            xt0 = mu_b
        return xt0

    def evolve(self, xti, T=1, reverse=False, noise=False):
        xt0 = xti
        if reverse == False:
            for i in range(T):
                xt1 = self.step_forward(xt0, noise=noise)
                xt0 = xt1
        else:
            for i in range(T):
                xt1 = self.step_backwards(xt0, noise=noise)
                xt0 = xt1
        return xt0

    def generate_trj(self, xti, T=1, reverse=False, noise=False):
        xt0 = xti
        trx = xt0[None, ...]
        if not reverse:
            for i in range(T - 1):
                xt1 = self.step_forward(xt0, noise=noise)
                xt0 = xt1
                trx = torch.cat((trx, xt0[None, ...]))
        else:
            for i in range(T - 1):
                xt1 = self.step_backwards(xt0, noise=noise)
                xt0 = xt1
                trx = torch.cat((trx, xt0[None, ...]))
        return trx

    def generate_trj_density(self, xti, T=1, reverse=False):
        _mu = xti
        _var = torch.zeros(xti.shape[0], self.dim, self.dim).to(xti)
        tr_mean = _mu[None, ...]
        tr_var = _var[None, ...]
        if not reverse:
            for i in range(T - 1):
                Ad = self.first_Taylor_dyn(_mu) * self.dt + torch.eye(self.dim).to(xti)
                _mu = self.velocity(_mu) * self.dt + _mu
                _var = torch.bmm(torch.bmm(Ad, _var), Ad) + self.var * self.dt

                tr_mean = torch.cat((tr_mean, _mu[None, ...]))
                tr_var = torch.cat((tr_var, _var[None, ...]))
        else:
            for i in range(T - 1):
                Ad = -self.first_Taylor_dyn(_mu) * self.dt + torch.eye(self.dim).to(xti)
                _mu = -self.velocity(_mu) * self.dt + _mu
                _var = torch.bmm(torch.bmm(Ad, _var), Ad) + self.var * self.dt

                tr_mean = torch.cat((tr_mean, _mu[None, ...]))
                tr_var = torch.cat((tr_var, _var[None, ...]))
        return tr_mean, tr_var

    def conditional_distribution(self, xti, T=1, reverse=False):
        _mu = xti
        _var = torch.zeros(xti.shape[0], self.dim, self.dim).to(xti)
        if not reverse:
            for i in range(T):
                Ad = self.first_Taylor_dyn(_mu) * self.dt + torch.eye(self.dim).to(xti)
                _mu = self.velocity(_mu) * self.dt + _mu
                _var = torch.bmm(torch.bmm(Ad, _var), Ad) + self.var * self.dt
        else:
            for i in range(T):
                Ad = -self.first_Taylor_dyn(_mu) * self.dt + torch.eye(self.dim).to(xti)
                _mu = -self.velocity(_mu) * self.dt + _mu
                _var = torch.bmm(torch.bmm(Ad, _var), Ad) + self.var * self.dt
        return tdist.MultivariateNormal(_mu, _var)


class LimitCycleDynamicModel(nn.Module):

    def __init__(self, dim, device=None, dt=0.01, requires_grad=True):
        super().__init__()
        self.dim = dim
        self.device = device
        self.dt = dt

    def forward(self, x, logpx=None, reverse=False):
        raise NotImplementedError('Forward function has to be implemented')

    def transform(self, x, reverse=False):
        raise NotImplementedError('Implement transformation from cycle to cartesian')

    def velocity(self, x):
        raise NotImplementedError('Velocity must be Implemented in the inherited Method')

    def first_Taylor_dyn(self, x):
        raise NotImplementedError('first_Taylor_dyn must be Implemented in the inherited Method')

    def step_forward(self, xt0, noise=False):
        ## Go To Polar
        xt0 = self.transform(xt0, reverse=False)
        ##Evolve
        vel = self.velocity(xt0)
        mu = vel * self.dt + xt0
        if noise == True:
            var_step = self.var * self.dt
            mv_dist = tdist.MultivariateNormal(mu, var_step)
            xt1 = mv_dist.rsample()
        else:
            xt1 = mu
        ## Go To Cartesian
        xt1 = self.transform(xt1, reverse=True)
        return xt1

    def step_backwards(self, xt1, noise=False):
        ## Go To Polar
        xt1 = self.transform(xt1, reverse=False)
        ##Evolve
        vel = -self.velocity(xt1)

        mu_b = vel * self.dt + xt1
        if noise == True:
            var_b = self.var * self.dt
            mv_dist = tdist.MultivariateNormal(mu_b, var_b)
            xt0 = mv_dist.rsample()
        else:
            xt0 = mu_b
        ## Go To Cartesian
        xt0 = self.transform(xt0, reverse=True)
        return xt0

    def evolve(self, xti, T=1, reverse=False, noise=False):
        ##Evolve
        xt0 = xti
        if reverse == False:
            for i in range(T):
                xt1 = self.step_forward(xt0, noise=noise)
                xt0 = xt1
        else:
            for i in range(T):
                xt1 = self.step_backwards(xt0, noise=noise)
                xt0 = xt1
        return xt0

    def generate_trj(self, xti, T=1, reverse=False, noise=False):
        xt0 = xti
        trx = xt0[None, ...]
        if not reverse:
            for i in range(T - 1):
                xt1 = self.step_forward(xt0, noise=noise)
                xt0 = xt1
                trx = torch.cat((trx, xt0[None, ...]))
        else:
            for i in range(T - 1):
                xt1 = self.step_backwards(xt0, noise=noise)
                xt0 = xt1
                trx = torch.cat((trx, xt0[None, ...]))
        return trx

    def generate_trj_density(self, xti, T=1, reverse=False):
        '''
        Generate Trajectory Density will create a trajectory in the polar coordinate space.
        '''
        _mu = xti
        _var = torch.zeros(xti.shape[0], self.dim, self.dim).to(xti)
        tr_mean = _mu[None, ...]
        tr_var = _var[None, ...]
        if not reverse:
            for i in range(T - 1):
                Ad = self.first_Taylor_dyn(_mu) * self.dt + torch.eye(self.dim).to(xti)
                _mu = self.velocity(_mu) * self.dt + _mu
                _var = torch.bmm(torch.bmm(Ad, _var), Ad) + self.var * self.dt

                tr_mean = torch.cat((tr_mean, _mu[None, ...]))
                tr_var = torch.cat((tr_var, _var[None, ...]))
        else:
            for i in range(T - 1):
                Ad = -self.first_Taylor_dyn(_mu) * self.dt + torch.eye(self.dim).to(xti)
                _mu = -self.velocity(_mu) * self.dt + _mu
                _var = torch.bmm(torch.bmm(Ad, _var), Ad) + self.var * self.dt

                tr_mean = torch.cat((tr_mean, _mu[None, ...]))
                tr_var = torch.cat((tr_var, _var[None, ...]))
        return tr_mean, tr_var

    def conditional_distribution(self, xti, T=1, reverse=False):
        '''
        Conditional Distribution will compute the distribution in the Polar Space
        '''
        _mu = xti
        _var = torch.zeros(xti.shape[0], self.dim, self.dim).to(xti)
        if not reverse:
            for i in range(T):
                Ad = self.first_Taylor_dyn(_mu) * self.dt + torch.eye(self.dim).to(xti)
                _mu = self.velocity(_mu) * self.dt + _mu
                _var = torch.bmm(torch.bmm(Ad, _var), Ad) + self.var * self.dt
        else:
            for i in range(T):
                Ad = -self.first_Taylor_dyn(_mu) * self.dt + torch.eye(self.dim).to(xti)
                _mu = -self.velocity(_mu) * self.dt + _mu
                _var = torch.bmm(torch.bmm(Ad, _var), Ad) + self.var * self.dt

        dists = []
        dist_r = tdist.Normal(loc=_mu[:,0], scale=torch.sqrt(_var[:,0,0]))
        dists.append(dist_r)
        dist_w = AngleNormal(loc=_mu[:,1], scale=torch.sqrt(_var[:,1,1]))
        dists.append(dist_w)
        if self.dim ==3:
            dist_z = tdist.Normal(loc=_mu[:,2], scale=_var[:,2,2])
            dists.append(dist_z)
        elif self.dim>3:
            dist_z = tdist.MultivariateNormal(loc=_mu[:,2:], scale=_var[:,2:,2:])
            dists.append(dist_z)
        return dists

    def cartesian_conditional_distribution(self, xti, T=1, reverse=False):
        _mu = xti
        _var = torch.zeros(xti.shape[0], self.dim, self.dim).to(xti)
        if not reverse:
            _mu = self.evolve(_mu, T=T)
            var = self.var*self.dt*T
            _var = var
        return tdist.MultivariateNormal(_mu, _var)

    def cartesian_cond_log_prob(self, xt0, xt1, T=1, reverse=False):
        '''
        Compute the Cartesian Conditional Distribution given as input xt0 and xt1
        '''
        if not reverse:
            dist = self.cartesian_conditional_distribution(xt0, T=T, reverse=reverse)
            log_px = dist.log_prob(xt1)
        else:
            dist = self.cartesian_conditional_distribution(xt1, T=T, reverse=reverse)
            log_px = dist.log_prob(xt0)
        return log_px

    def conditional_log_prob(self, xt0, xt1, T=1, reverse=False):
        '''
        Compute the Conditional Distribution given as input xt0 and xt1
        '''
        zeros = torch.zeros(xt0.shape[0],1).to(xt0)
        z0, log_J0 = self.forward(xt0, zeros)
        z1, log_J1 = self.forward(xt1, zeros)

        if not reverse:
            dists = self.conditional_distribution(z0, T=T, reverse=reverse)
            log_px = log_J1.squeeze()
            log_px += dists[0].log_prob(z1[:,0])
            log_px += dists[1].log_prob(z1[:,1])
            if self.dim == 3:
                log_px += dists[2].log_prob(z1[:,2])
            elif self.dim > 3:
                log_px += dists[2].log_prob(z1[:,2:])
        else:
            dists = self.conditional_distribution(z1, T=T, reverse=reverse)
            log_px = log_J0.squeeze()
            log_px += dists[0].log_prob(z0[:, 0])
            log_px += dists[1].log_prob(z0[:, 1])
            if self.dim == 3:
                log_px += dists[2].log_prob(z0[:, 2])
            elif self.dim > 3:
                log_px += dists[2].log_prob(z0[:, 2:])
        return log_px

    def final_distribution(self, x , ref_phase=None):
        dists = []
        ##radius final distribution
        _mu_r = self.r_des
        _var_r = 0.01*torch.ones(1).to(x)
        dist_r = tdist.Normal(loc=_mu_r, scale=_var_r)
        dists.append(dist_r)
        ##

        return dists

    def cartesian_final_distribution(self, x, ref_phase):
        sin_x = torch.sin(ref_phase)
        cos_x = torch.cos(ref_phase)
        _mu = torch.cat([sin_x[:,None],cos_x[:,None]],1)
        _var = 0.1*torch.eye(x.shape[1]).to(x)

        if x.shape[1]>2:
            _mu_z = torch.zeros(x.shape[0], x.shape[1]-2).to(x)
            _mu = torch.cat([_mu, _mu_z],1)

        dist = tdist.MultivariateNormal(loc=_mu, covariance_matrix=_var)
        return dist

    def stable_log_prob(self, x, ref_phase = None):

        if ref_phase is not None:
            dist = self.cartesian_final_distribution(x, ref_phase)
            logpx = dist.log_prob(x)
            return logpx

        else:
            zeros = torch.zeros(x.shape[0], 1).to(x)
            z, log_J = self.forward(x, zeros)

            dists = self.final_distribution(x, ref_phase)
            log_px = log_J.squeeze()
            log_px += dists[0].log_prob(z[:, 0])
        return log_px
















