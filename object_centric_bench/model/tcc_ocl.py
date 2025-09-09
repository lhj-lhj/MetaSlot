from einops import rearrange
import torch.nn as nn


class TCCOCL(nn.Module):

    def __init__(self, aggr, dim=2):
        super().__init__()
        self.aggr = aggr
        assert dim == 2
        self.dim = dim

    def forward(self, input):
        b, t, n, c = input.shape
        x = rearrange(input, "b t n c -> (b t) n c")
        x = self.aggr(x)  # (b*t,n,c)
        x = rearrange(x, "(b t) n c -> b t n c", b=b)
        x = x.mean(self.dim)  # (b,t,c)
        return x
