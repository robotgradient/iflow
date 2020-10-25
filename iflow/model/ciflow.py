import torch
import torch.nn as nn


class ContinuousDynamicFlow(nn.Module):
    def __init__(self, model, dynamics, dim=2, context_dim=2, device=None, dt=0.01):
        super().__init__()
        self.device = device
        self.flow = model
        self.flow_backward, self.flow_forward = self.get_transforms(model)
        self.dynamics = dynamics

    def get_transforms(self, model):

        def sample_fn(z, logpz=None, context=None):
            if logpz is not None:
                return model(z, logpz, reverse=True)
            else:
                return model(z,  reverse=True)

        def density_fn(x, logpx=None, context=None):
            if logpx is not None:
                return model(x, logpx, reverse=False)
            else:
                return model(x, reverse=False)

        return sample_fn, density_fn

    def forward(self, yt, context=None):
        zero = torch.zeros(yt.shape[0], 1).to(yt)
        xt, log_detjacobians = self.flow_forward(yt, zero, context=context)
        #zt, log_p = self.dynamics(xt,log_detjacobians)
        return xt, log_detjacobians

    def generate_trj(self, y0, T=100, noise=False, reverse=False):
        z0 = self.flow_forward(y0)
        trj_z = self.dynamics.generate_trj(z0, T=T, reverse = reverse, noise = noise)
        trj_y = self.flow_backward(trj_z[:, 0, :])
        return trj_y

    def evolve(self, y0, T=100, noise=False, reverse=False):
        z0 = self.flow_forward(y0)
        z1 = self.dynamics.evolve(z0, T=T, reverse=reverse, noise=noise)
        y1 = self.flow_backward(z1)
        return y1
