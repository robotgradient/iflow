import torch
from iflow.utils.typechecks import is_nonnegative_int, is_positive_int


def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors.
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2


def block_diag(matrixs):
    size=0
    for m in matrixs:
        size+= m.shape[-1]
    batch = matrixs[0].shape[0]
    index=0
    block_mask = torch.zeros(batch,size,size).to(matrixs[0])
    for m in matrixs:
        mask = torch.zeros(batch,size,size).to(m)
        mask[:, index:index+m.shape[-1], index:index+m.shape[-1]] = m
        block_mask += mask
        index +=m.shape[-1]
    return block_mask


def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if not is_nonnegative_int(num_batch_dims):
        raise TypeError('Number of batch dimensions must be a non-negative integer.')
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def tile(x, n):
    if not is_positive_int(n):
        raise TypeError('Argument \'n\' must be a positive integer.')
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1
