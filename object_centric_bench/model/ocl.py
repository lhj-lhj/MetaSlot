import math

from einops import rearrange, repeat
from scipy.optimize import linear_sum_assignment
import numpy as np
import timm
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf

from .basic import MLP
from .vaez import VQVAEZ, QuantiZ

class SlotAttention(nn.Module):
    """TODO XXX modularization/cgv: correct the wrong implementation!"""

    def __init__(
        self, num_iter, embed_dim, ffn_dim, dropout=0, kv_dim=None, trunc_bp=None
    ):
        """
        - dropout: only works in self.ffn; a bit is beneficial
        """
        super().__init__()
        kv_dim = kv_dim or embed_dim
        assert trunc_bp in ["bi-level", None]
        self.num_iter = num_iter
        self.trunc_bp = trunc_bp
        self.norm1q = nn.LayerNorm(embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm1kv = nn.LayerNorm(kv_dim)
        self.proj_k = nn.Linear(kv_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(kv_dim, embed_dim, bias=False)
        # self.dropout = nn.Dropout(dropout)  # always bad for attention
        self.rnn = nn.GRUCell(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = MLP(embed_dim, [ffn_dim, embed_dim], None, dropout)

    def forward(self, input, query, smask=None, num_iter=None):
        """
        input: in shape (b,h*w,c)
        query: in shape (b,n,c)
        smask: slots' mask, shape=(b,n), dtype=bool
        """
        b, n, c = query.shape
        self_num_iter = num_iter or self.num_iter
        kv = self.norm1kv(input)
        k = self.proj_k(kv)
        v = self.proj_v(kv)
        q = query
        for _ in range(self_num_iter):
            if _ + 1 == self_num_iter:
                if self.trunc_bp == "bi-level":  # BO-QSA
                    q = q.detach() + query - query.detach()
            x = q
            q = self.norm1q(q)
            q = self.proj_q(q)
            u, a = __class__.inverted_scaled_dot_product_attention(q, k, v, smask)
            y = self.rnn(u.flatten(0, 1), x.flatten(0, 1)).view(b, n, -1)
            z = self.norm2(y)
            q = y + self.ffn(z)  # droppath on ffn seems harmful
        return q, a

    @staticmethod
    def inverted_scaled_dot_product_attention(q, k, v, smask=None, eps=1e-5):
        scale = q.size(2) ** -0.5  # temperature
        logit = pt.einsum("bqc,bkc->bqk", q * scale, k)
        if smask is not None:
            logit = logit.where(smask[:, :, None], -pt.inf)
        a0 = logit.softmax(1)  # inverted: softmax over query  # , logit.dtype
        a = a0 / (a0.sum(2, keepdim=True) + eps)  # re-normalize over key
        # a = self_dropout(a)
        o = pt.einsum("bqv,bvc->bqc", a, v)
        return o, a0

    """@staticmethod
    def inverted_scaled_dot_product_attention(q, k, v, eps=1e-5, h=4):
        q = rearrange(q, "b q (h d) -> (b h) q d", h=h)
        k = rearrange(k, "b k (h d) -> (b h) k d", h=h)
        v = rearrange(v, "b k (h d) -> (b h) k d", h=h)
        scale = q.size(2) ** -0.5  # temperature
        logit = pt.einsum("bqc,bkc->bqk", q * scale, k)
        a0 = logit.softmax(1)  # inverted: softmax over query  # , logit.dtype
        a = a0 / (a0.sum(2, keepdim=True) + eps)  # re-normalize over key
        # a = self_dropout(a)
        o = pt.einsum("bqv,bvc->bqc", a, v)
        o = rearrange(o, "(b h) q d -> b q (h d)", h=h)
        return o, a0"""

class CartesianPositionalEmbedding2d(nn.Module):
    """"""

    def __init__(self, resolut, embed_dim):
        super().__init__()
        self.pe = nn.Parameter(
            __class__.meshgrid(resolut)[None, ...], requires_grad=False
        )
        self.project = nn.Linear(4, embed_dim)

    @staticmethod
    def meshgrid(resolut, low=-1, high=1):
        assert len(resolut) == 2
        yx = [pt.linspace(low, high, _ + 1) for _ in resolut]
        yx = [(_[:-1] + _[1:]) / 2 for _ in yx]
        grid_y, grid_x = pt.meshgrid(*yx)
        return pt.stack([grid_y, grid_x, 1 - grid_y, 1 - grid_x], 2)

    def forward(self, input):
        """
        input: in shape (b,h,w,c)
        output: in shape (b,h,w,c)
        """
        max_h, max_w = input.shape[1:3]
        output = input + self.project(self.pe[:, :max_h, :max_w, :])
        return output


class LearntPositionalEmbedding(nn.Module):
    """Support any dimension. Must be channel-last.
    PositionalEncoding: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, resolut: list, embed_dim: int):
        super().__init__()
        self.resolut = resolut
        self.embed_dim = embed_dim
        self.pe = nn.Parameter(pt.zeros(1, *resolut, embed_dim), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input, retp=False):
        """
        input: in shape (b,*r,c)
        output: in shape (b,*r,c)
        """
        max_r = ", ".join([f":{_}" for _ in input.shape[1:-1]])
        # TODO XXX support variant length
        # pe = timm.layers.pos_embed.resample_abs_pos_embed(self.pe, ...)
        # pe = self.pe[:, :max_resolut, :]
        pe = eval(f"self.pe[:, {max_r}, :]")
        output = input + pe
        if retp:
            return output, pe
        return output

    def extra_repr(self):
        return f"{self.resolut}, {self.embed_dim}"


class NormalSeparat(nn.Module):
    """Separate gaussians as queries."""

    def __init__(self, num, dim):
        super().__init__()
        self.num = num
        self.dim = dim
        self.mean = nn.Parameter(pt.empty(1, num, dim))
        self.logstd = nn.Parameter(
            (pt.ones(1, num, dim) * dim**-0.5).log()
        )  # scheduled std cause nan in dinosaur; here is learnt
        nn.init.xavier_uniform_(self.mean[0, :, :])  # very important

    def forward(self, b):
        smpl = self.mean.expand(b, -1, -1)
        if self.training:
            randn = pt.randn_like(smpl)
            smpl = smpl + randn * self.logstd.exp()
        return smpl

    def extra_repr(self):
        return f"1, {self.num}, {self.dim}"


class NormalShared(nn.Module):
    """Shared gaussian as queries."""

    # TODO new trick: Conditional Random Initialization

    def __init__(self, num, dim):
        super().__init__()
        self.num = num
        self.dim = dim
        self.mean = nn.Parameter(pt.empty(1, 1, dim))
        self.logstd = nn.Parameter(pt.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.mean)
        nn.init.xavier_uniform_(self.logstd)

    def forward(self, b, n=None):
        self_num = self.num
        if n is not None:
            self_num = n
        smpl = self.mean.expand(b, self_num, -1)
        randn = pt.randn_like(smpl)
        smpl = smpl + randn * self.logstd.exp()
        return smpl


class VQVAE(nn.Module):
    """
    Oord et al. Neural Discrete Representation Learning. NeurIPS 2017.

    reconstruction loss and codebook alignment/commitment (quantization) loss
    """

    def __init__(self, encode, decode, codebook):
        super().__init__()
        self.encode = encode  # should be encoder + quantconv
        self.decode = decode  # should be decoder + postquantconv
        self.codebook = codebook

    def forward(self, input):
        """
        input: image; shape=(b,c,h,w)
        """
        encode = self.encode(input)
        zsoft, zidx = self.codebook.match(encode, False)
        quant = self.codebook(zidx).permute(0, 3, 1, 2)  # (b,h,w,c) -> (b,c,h,w)
        quant = (
            quant  # .to(encode.dtype)  # TODO added after gdr, ogdr; add fp32 to diffuz
        )
        quant2 = __class__.grad_approx(encode, quant)
        decode = None
        if self.decode:
            decode = self.decode(quant2)
        return encode, zidx, quant, decode

    @staticmethod
    def grad_approx(z, q, nu=0):  # nu=1 always harmful; maybe smaller nu???
        """
        straight-through gradient approximation

        synchronized:
        Straightening Out the Straight-Through Estimator: Overcoming Optimization Challenges in Vector Quantized Networks
        """
        assert nu >= 0
        q = z + (q - z).detach()  # delayed: default
        if nu > 0:
            q += nu * (q - q.detach())  # synchronized
        return q


class Codebook(nn.Module):
    """
    clust: always negative
    replac: always positive
    sync: always negative
    """

    def __init__(self, num_embed, embed_dim):
        super().__init__()
        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.templat = nn.Embedding(num_embed, embed_dim)
        n = self.templat.weight.size(0)  # good to vqvae pretrain but bad to dvae
        self.templat.weight.data.uniform_(-1 / n, 1 / n)

    def forward(self, input):
        """
        input: indexes in shape (b,..)
        output: in shape (b,..,c)
        """
        output = self.templat(input)
        return output

    def match(self, encode, sample: bool, tau=1, detach="encode"):
        return __class__.match_encode_with_templat(
            encode, self.templat.weight, sample, tau, detach
        )

    @pt.no_grad()
    def cluster(self, latent, max_iter=100):  # always harmful
        """Straightening Out the Straight-Through Estimator: Overcoming Optimization Challenges in Vector Quantized Networks

        latent: shape=(b,c,h,w)
        """
        assert self.training  # if not self.training: return
        if not hasattr(self, "cluster_flag"):  # only once
            self.cluster_flag = pt.zeros([], dtype=pt.bool, device=latent.device)
        if self.cluster_flag:
            return
        self.cluster_flag.data[...] = True
        latent = latent.permute(0, 2, 3, 1).flatten(0, -2)  # (b,h,w,c)
        n, c = latent.shape
        if n < self.num_embed:
            raise f"warmup samples should >= codebook size: {n} vs {self.num_embed}"
        print("clustering...")
        assign, centroid = __class__.kmeans_pt(
            latent, self.num_embed, max_iter=max_iter
        )
        self.templat.weight.data[...] = centroid

    @pt.no_grad()
    def replace(self, latent, zidx, rate=1, rho=1e-2, timeout=4096, cluster=0.5):
        """Straightening Out the Straight-Through Estimator: Overcoming Optimization Challenges in Vector Quantized Networks

        latent: shape=(b,c,h,w)
        zidx: shape=(b,..)
        timeout: in #vector; will be converted to #iter
        cluster: with is too slow !!!

        Alchemy
        ---
        for stage2 (maynot stand for stage1):
        - replace rate: 1>0.5;
        - noise rho: 1e-2>0;
        - replace timeout: 4096>1024,16384;
        - enabled in half training steps > full;
        - cluster r0.5 > r0.1?
        """
        assert self.training  # if not self.training: return
        if not hasattr(self, "replace_cnt"):  # only once
            self.replace_cnt = pt.ones(
                self.num_embed, dtype=pt.int, device=latent.device
            )
            self.replace_rate = pt.tensor(
                rate, dtype=latent.dtype, device=latent.device
            )
        assert 0 <= self.replace_rate <= 1
        if self.replace_rate == 0:
            return

        latent = latent.permute(0, 2, 3, 1).flatten(0, -2)
        m = latent.size(0)
        timeout = math.ceil(timeout * self.num_embed / m)  # from #vector to #iter

        assert 0 <= cluster <= 1
        if self.replace_rate > 0 and cluster > 0:  # cluster update rate
            assert m >= self.num_embed
            if not hasattr(self, "replace_centroid"):
                self.replace_centroid = __class__.kmeans_pt(
                    latent,
                    self.num_embed,
                    self.templat.weight.data,  # .to(latent.dtype),
                    max_iter=100,
                )[1]
            else:
                centroid = __class__.kmeans_pt(
                    latent, self.num_embed, self.replace_centroid, max_iter=1
                )[1]
                self.replace_centroid = (  # equal to ema ???
                    self.replace_centroid * (1 - cluster) + centroid * cluster
                )

        assert self.replace_cnt.min() >= 0
        self.replace_cnt -= 1

        # reset cnt of recently used codes
        active_idx = pt.unique(zidx)
        self.replace_cnt.index_fill_(0, active_idx, timeout)

        # reset value of unused codes
        dead_idx = (self.replace_cnt == 0).argwhere()[:, 0]  # (n,)->(n,1)->(n,)
        num_dead = dead_idx.size(0)
        if num_dead > 0:
            print("#", timeout, self.num_embed, m, dead_idx)
            mult = num_dead // m + 1

            ### policy: random from input
            """latent = latent[pt.randperm(m)]
            if mult > 1:  # no need to repeat and shuffle all as mult always == 1
                latent = latent.tile([mult, 1])
            replac = latent[:num_dead]"""
            ### policy: random least similar to others from input
            """dist = __class__.euclidean_distance(latent, self.templat(active_idx))
            ridx = dist.mean(1).topk(min(num_dead, m), sorted=False)[1]
            if mult > 1:
                ridx = ridx.tile(mult)[:num_dead]
            replac = latent[ridx]"""
            ### policy: most similar centriod to self from input -- VQ-NeRV: A Vector Quantized Neural Representation for Videos
            dist = __class__.euclidean_distance(
                self.templat.weight.data[dead_idx], self.replace_centroid
            )
            row_idx, col_idx = linear_sum_assignment(dist.detach().cpu())
            replac = self.replace_centroid[pt.from_numpy(col_idx).to(latent.device)]

            # add noise
            if rho > 0:  # helpful
                norm = replac.norm(p=2, dim=-1, keepdim=True)
                noise = pt.randn_like(replac)
                replac = replac + rho * norm * noise

            self.templat.weight.data = self.templat.weight.data.clone()
            self.templat.weight.data[dead_idx] = (
                self.templat.weight.data[dead_idx] * (1 - self.replace_rate)
                + replac * self.replace_rate
            )
            self.replace_cnt[dead_idx] += timeout

    @staticmethod
    def kmeans_pt(
        X,
        num_cluster: int,
        center=None,
        tol=1e-4,
        max_iter=100,
        split_size=4096,
        replace=True,
    ):
        """euclidean kmeans in pytorch
        https://github.com/subhadarship/kmeans_pytorch/blob/master/kmeans_pytorch/__init__.py

        X: shape=(m,c)
        tol: minimum shift to run before stop
        max_iter: maximum iterations to stop
        center: (initial) centers for clustering; shape=(n,c)
        assign: clustering assignment to vectors in X; shape=(m,)
        """
        m, c = X.shape
        if center is None:
            idx0 = pt.randperm(m)[:num_cluster]
            center = X[idx0]

        shifts = []
        cnt = 0
        while True:
            dist = __class__.euclidean_distance(
                X, center, split_size=split_size
            )  # mc,nc->mn
            dmin, assign = dist.min(1)  # (m,)
            center_old = center.clone()

            for cid in range(num_cluster):
                idx = assign == cid
                if not idx.any():
                    if replace:
                        idx = pt.randperm(m)[:num_cluster]
                        # print(f"center #{cid} replaced")
                    else:
                        continue
                cluster = X[idx]  # (m2,c)  # index_select
                center[cid] = cluster.mean(0)

            shift = ptnf.pairwise_distance(center, center_old).mean().item()
            print(
                f"[kmeans_pt] {cnt}, {shift:.4f}, {dmin.mean().item():.4f}, {center.max().item():.4f} {center.min().item():.4f} {center.norm(2, 1).mean():.4f}"
            )
            shifts.append(shift)
            shifts = shifts[-10:]
            if shift < tol or len(shifts) > 1 and np.std(shifts) == 0:
                # import pdb; pdb.set_trace()
                break
            cnt = cnt + 1
            if max_iter > 0 and cnt >= max_iter:
                # import pdb; pdb.set_trace()
                break
            # print(cnt)

        return assign, center

    @staticmethod
    def match_encode_with_templat(encode, templat, sample, tau=1, detach="encode"):
        """
        encode: in shape (b,c,h,w)
        templat: in shape (m,c)
        zsoft: in shape (b,m,h,w)
        zidx: in shape (b,h,w)
        """
        if detach == "encode":
            encode = encode.detach()
        elif detach == "templat":
            templat = templat.detach()
        # b, c, h, w = encode.shape
        # dist = __class__.euclidean_distance(  # (b*h*w,c) (m,c) -> (b*h*w,m)
        #    encode.permute(0, 2, 3, 1).flatten(0, -2), templat
        # )
        # dist = dist.view(b, h, w, -1).permute(0, 3, 1, 2)  # (b,m,h,w)
        # simi = -dist.square()  # better than without  # TODO XXX learnable scale ???
        dist = (  # always better than cdist.square, why ???
            encode.square().sum(1, keepdim=True)  # (b,1,h,w)
            + templat.square().sum(1)[None, :, None, None]
            - 2 * pt.einsum("bchw,mc->bmhw", encode, templat)
        )  # 1 > 0.5 > 2, 4
        simi = -dist
        if sample and tau > 0:
            zsoft = ptnf.gumbel_softmax(simi, tau, False, dim=1)
        else:
            zsoft = simi.softmax(1)
        zidx = zsoft.argmax(1)  # (b,m,h,w) -> (b,h,w)
        return zsoft, zidx

    @staticmethod
    def euclidean_distance(source, target, split_size=4096):
        """chunked cdist

        source: shape=(b,m,c) or (m,c)
        target: shape=(b,n,c) or (n,c)
        split_size: in case of oom; can be bigger than m
        dist: shape=(b,m,n) or (m,n)
        """
        assert source.ndim == target.ndim and source.ndim in [2, 3]
        source = source.split(split_size)  # type: list
        dist = []
        for s in source:
            d = pt.cdist(s, target, p=2)  # (m2,n);
            dist.append(d)
        dist = pt.concat(dist)  # (m,n)
        return dist
