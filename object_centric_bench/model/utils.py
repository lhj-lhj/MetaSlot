from itertools import product
import statistics

import math

import numpy as np
import torch as pt
import torch.nn.functional as ptnf


def lecun_normal_(tensor, scale=1.0):
    """timm.models.layers.lecun_normal_, for conv/transposed-conv"""

    def _calculate_fanin_and_fanout(tensor):
        ndim = tensor.dim()
        assert ndim >= 2
        receptive_field_size = np.prod(tensor.shape[2:]) if ndim > 2 else 1
        fan_in = tensor.size(1) * receptive_field_size
        fan_out = tensor.size(0) * receptive_field_size
        return fan_in, fan_out

    def _trunc_normal_(tensor, mean, std, a, b):
        norm_cdf = lambda x: (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
        assert not ((mean < a - 2 * std) or (mean > b + 2 * std))
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)

    fan_in, fan_out = _calculate_fanin_and_fanout(tensor)
    denom = fan_in
    variance = scale / denom
    std = math.sqrt(variance) / 0.87962566103423978
    with pt.no_grad():
        _trunc_normal_(tensor, 0, 1.0, -2, 2)
        tensor.mul_(std).add_(0)


def sin_pos_enc(seq_len, d_model):
    """Sinusoidal absolute positional encoding."""
    inv_freq = 1.0 / (10000 ** (pt.arange(0.0, d_model, 2.0) / d_model))
    pos_seq = pt.arange(seq_len - 1, -1, -1).type_as(inv_freq)
    sin_inp = pt.outer(pos_seq, inv_freq)
    pos_emb = pt.cat([sin_inp.sin(), sin_inp.cos()], dim=-1)
    return pos_emb.unsqueeze(0)  # [1, L, C]


def find_groups(base=1024, g=4, r=0.5):
    """
    base:
    g: num_group
    r: radius of root value in percentage
    """
    x = np.power(base, 1 / g)  # root value
    print(x)
    xs = np.arange(np.round(x * (1 - r)), np.round(x * (1 + r)) + 1).astype("int")
    xs = xs.astype("int")  # root neighbors
    print(xs)

    combins = np.array(list(product(xs, repeat=g)))
    print(combins)

    flag = np.abs(np.prod(combins, axis=1) - base)
    assert flag.ndim == 1
    combin0 = combins[flag == 0]
    print("equal", combin0)

    flag1 = np.mean(np.abs(combin0 - x), axis=1)
    combin1 = combin0[np.argmin(flag1)]
    print("least", combin1, np.min(flag1))
    return combin1


def get_subclass_method_keys(obj, superclass):
    return [
        attr
        for attr in dir(obj)
        if callable(getattr(obj, attr)) and not hasattr(superclass, attr)
    ]
