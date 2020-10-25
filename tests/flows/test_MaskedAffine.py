from torch.nn import functional as F
import iflow.model as model
from .helper.flow_class_test import TestClass

dim = 2
depth = 6
hidden_features = 256
num_transform_blocks = 2
dropout_probability = .25
use_batch_norm = 0

dataname = 'checkerboard'

def main_layer(dim):
    return model.flows.MaskedAffineAutoregressiveTransform(
            features=dim,
            hidden_features=hidden_features,
            context_features=None,
            num_blocks=num_transform_blocks,
            use_residual_blocks=True,
            random_mask=False,
            activation=F.relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
    )


def construct_model(dim):
    chain = []
    for i in range(depth):
        chain.append(main_layer(dim))
        chain.append(model.flows.RandomPermutation(dim))
    chain.append(main_layer(dim))
    return model.SequentialFlow(chain)


def test_train():
    model = construct_model(dim)
    test = TestClass(model, dataname=dataname)
    test.train()

test_train()


