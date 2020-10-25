import math
from numbers import Number
import torch
from torch.distributions import Normal


class AngleNormal(Normal):

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Number) else self.scale.log()

        diff_theta = value - self.loc
        diff = torch.atan2(torch.sin(diff_theta),torch.cos(diff_theta))

        return -((diff) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))



if __name__ == "__main__":
    dist = AngleNormal(loc=7., scale=.5)

    samples = torch.linspace(-10,10,100)
    log_prob = dist.log_prob(samples)

    import matplotlib.pyplot as plt
    plt.plot(samples, log_prob)
    plt.show()