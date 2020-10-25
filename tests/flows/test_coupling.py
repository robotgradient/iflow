import iflow.model as model
from .helper.flow_class_test import TestClass

dim = 2
depth = 30


def main_layer(dim):
    return model.flows.CouplingLayer(dim)


def construct_model(dim):
    chain = []
    for i in range(10):
        chain.append(main_layer(dim))
        chain.append(model.flows.RandomPermutation(dim))
        chain.append(model.flows.LULinear(dim))
    chain.append(main_layer(dim))
    return model.SequentialFlow(chain)


def test_train():
    model = construct_model(dim)
    test = TestClass(model)
    test.train()

test_train()



