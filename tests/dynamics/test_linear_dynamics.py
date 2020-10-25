import matplotlib.pyplot as plt
import numpy as np
import torch

from iflow.model.dynamics.linear_stochastic_dynamics import LinearStochasticDynamics
from .helper.dynamics_class_test import TestClass

dynamics = LinearStochasticDynamics(3, dt=0.01 ,T_to_stable=2.5)
model = TestClass(dynamics)


def test_points():
    model.points_evolution()


def test_noise_forward():
    model.noise_forward_evaluation()


def test_noise_backward():
    model.noise_backward_evaluation()


def test_density_forward():
    model.forward_density()


def test_density_backward():
    model.backward_density()


test_points()
test_noise_forward()
test_noise_backward()
test_density_forward()
test_density_backward()