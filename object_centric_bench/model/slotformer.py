from itertools import combinations

from einops import rearrange
import torch as pt
import torch.nn as nn

from .utils import sin_pos_enc


class Rollouter(nn.Module):
    """"""

    def __init__(self, entity_dim, old_len, t_pe="sin", mid_dim=256, transit=None):
        super().__init__()
        self.proj_i = nn.Linear(entity_dim, mid_dim)
        self.t_pe = __class__.build_pos_enc(
            t_pe, old_len, mid_dim  # TODO change old_len to +inf XXX
        )
        self.transit = transit
        # self.s_pe = self.build_pos_enc(s_pe, num_slot, d_model)
        self.proj_o = nn.Linear(mid_dim, entity_dim)
        self.old_len = old_len

    def forward(self, input: pt.Tensor, new_len: int) -> pt.Tensor:
        """
        - input: historical slots in shape (b,t,n,c), where t=old_len
        - output: predicted slots in shape (b,t,n,c), where t=new_len
        """
        b, t, n, c = input.shape
        assert t <= self.old_len

        olds = input.flatten(1, 2)  # (b,t*n,c)

        t_pe = self.t_pe[:, :t, None, :].repeat(b, 1, n, 1).flatten(1, 2)
        # s_pe = self.s_pe[:, None, :, :].repeat(b, t0, 1, 1).flattent(1, 2)
        pe = t_pe  # + s_pe

        news = []
        for _ in range(new_len):
            x = self.proj_i(olds)  # TODO change to x=x+proj(pe); x=mlp(x)
            x = x + pe
            x = self.transit(x)
            new = self.proj_o(x[:, -n:, :])
            news.append(new)
            olds = pt.cat([olds[:, n:, :], new], dim=1)

        output = pt.stack(news, dim=1)
        return output

    @staticmethod
    def build_pos_enc(pos_enc, input_len, d_model):
        """Positional Encoding of shape [1, L, D]."""
        # ViT, BEiT etc. all use zero-init learnable pos enc
        if pos_enc == "learnable":
            pos_embed = nn.Parameter(pt.zeros(1, input_len, d_model))
        # in SlotFormer, we find out that sine P.E. is already good enough
        elif "sin" in pos_enc:  # 'sin', 'sine'
            pos_embed = nn.Parameter(  # (b=1,t,c)
                sin_pos_enc(input_len, d_model), requires_grad=False
            )
        else:
            raise "NotImplemented"
        return pos_embed


class RelationNetwork(nn.Module):
    """
    https://arxiv.org/abs/1706.01427
    """

    def __init__(self, in_dim, embed_dim, out_dim, aggr="max"):
        super().__init__()
        self.relat = nn.Linear(in_dim * 2, embed_dim)
        assert aggr in ["max", "mean"]
        if aggr == "max":
            self.aggr = lambda _: pt.max(_, dim=1)[0]
        else:
            self.aggr = lambda _: pt.mean(_, dim=1)
        self.logit = nn.Linear(embed_dim, out_dim)

    def forward(self, input):
        """
        input: shape=(b,n,c)
        output: shape=(b,c)
        """
        b, n, c = input.shape
        device = input.device
        comb = pt.combinations(pt.arange(n, dtype=pt.long, device=device), 2)
        entitiz = input[:, comb, :]  # (b,m,2,c)
        entitiz_ = rearrange(entitiz, "b m r c -> b m (r c)")
        relat = self.relat(entitiz_)  # (b,m,c)
        relat = self.aggr(relat)  # (b,c)
        logit = self.logit(relat)  # (b,c)
        return logit


class SlotFormer(nn.Module):
    """"""

    def __init__(self, old_len, rollout, readout=None, aggr="max", decode=None):
        super().__init__()
        # encoder: parts of ocl, returns slots
        #   encode, correct, mediat.encode/quant
        # rollouter: ...
        # decoder: parts of ocl, returns decoding
        #   decode (mlp/cnn)
        #   mediat.decode
        self.old_len = old_len
        self.rollout = rollout
        self.readout = readout
        assert aggr in ["max", "mean"]
        if aggr == "max":
            self.aggr = lambda _: pt.max(_, dim=1)[0]
        else:
            self.aggr = lambda _: pt.mean(_, dim=1)
        self.decode = decode

    def forward(self, input, use_dec=False, new_len: int = None) -> tuple:
        """
        - input: in shape (b,t,n,c)
        - future: in shape (b,t,n,c)
        - decode: in shape (b,t,c,h,w)
        - segment: in shape (b,t,h,w)
        """
        b, t, n, c = input.shape
        history = input[:, : self.old_len, :, :]
        if new_len is None:
            new_len = t - self.old_len
        future = self.rollout(history, new_len)
        readout = None
        if self.readout:
            if self.training:
                entitiz = input
            else:
                entitiz = pt.cat([history, future], dim=1)
            entitiz_ = rearrange(entitiz, "b t n c -> (b t) n c")
            logit_ = self.readout(entitiz_)
            logit = rearrange(logit_, "(b t) c -> b t c", b=b)
            readout = self.aggr(logit)  # (b,c) or (b,t,c)
        decode = None
        if self.decode and use_dec:
            raise NotImplementedError
        return future, readout, decode
