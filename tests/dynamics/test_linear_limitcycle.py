from iflow.model.dynamics.linear_limit_cycle import LinearLimitCycle
from .helper.limitcycle_class_test import TestClass

dynamics = LinearLimitCycle(3, dt=0.01 ,T_to_stable=1.)
model = TestClass(dynamics)


def test_points_eval():
    model.points_evolution()


def test_forward_noise():
    model.noise_forward_evaluation()


def test_backwards_noise():
    model.noise_backward_evaluation()


def test_conditional_forward():
    model.conditional_prob_forward()


def test_forward_density():
    model.forward_density()


def test_backward_density():
    model.backward_density()


test_points_eval()
test_forward_noise()
test_backwards_noise()

test_conditional_forward()
test_forward_density()
test_backward_density()