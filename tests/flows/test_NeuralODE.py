import iflow.model as model
from .helper.flow_class_test import TestClass

dim = 2

layer_type = 'hyper'
nonlinearity = 'tanh'
hidden_dim = '64-64-64'
hidden_dims = tuple(map(int, hidden_dim.split("-")))
divergence_fn = 'brute_force'
residual = False
rademacher = False

time_length = .5
train_T = True
solver = 'dopri5'

num_blocks=1


def main_layer():
    diffeq = model.cflows.ODEnet(
        hidden_dims=hidden_dims,
        input_shape=(dim,),
        strides=None,
        conv=False,
        layer_type=layer_type,
        nonlinearity=nonlinearity,
    )
    odefunc = model.cflows.ODEfunc(
        diffeq= diffeq,
        divergence_fn= divergence_fn,
        residual= residual,
        rademacher= rademacher,
    )
    cnf = model.cflows.CNF(
        odefunc=odefunc,
        T= time_length,
        train_T= train_T,
        regularization_fns=None,
        solver= solver,
    )
    return cnf


def construct_model(dim):
    chain = [main_layer() for _ in range(num_blocks)]
    return model.SequentialFlow(chain)


def test_train():
    model = construct_model(dim)
    test = TestClass(model)
    test.train()

test_train()



