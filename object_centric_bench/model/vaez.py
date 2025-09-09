import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf
import math
import time

class VQVAEZ(nn.Module):
    """
    Modified to accept input shape (b, n, c) instead of (b, c, h, w)
    """

    def __init__(self, encode, decode, quant, alpha=0.0, retr=True):
        super().__init__()
        self.encode = encode
        self.decode = decode
        self.quant = quant
        self.register_buffer("alpha", pt.tensor(alpha, dtype=pt.float))
        self.retr = retr

    def forward(self, input):
        """
        input: sequence; shape=(b, n, c)  # Changed input shape comment
        """
        # Changed: removed conv-style permutes since input is now (b, n, c)
        encode = self.encode(input)                   # (b, n, c)
        zidx, quant = self.quant(encode)              # zidx shape=(b, n), quant=(b, n, c)
        residual = quant
        decode = None
        if self.decode:
            if self.alpha > 0:
                # Changed: combine encode and quant along last dim
                residual = encode * self.alpha + quant * (1 - self.alpha)
            ste = __class__.naive_ste(encode, residual)
            decode = self.decode(ste)                  # decode expects (b, n, c)
        if self.retr:
            return encode, zidx, quant, residual, decode
        else:
            return encode, zidx, quant, decode

    @staticmethod
    def naive_ste(encode, quant):
        return encode + (quant - encode).detach()


class QuantiZ(nn.Module):
    """
    Modified to accept input shape (b, n, c) and codebook dim [num_code, c]
    """

    def __init__(self, num_code, code_dim, std=0):
        super().__init__()
        self.num_code = num_code
        self.code_dim = code_dim

        # Changed: codebook embedding dim set to code_dim (c)
        self.codebook = nn.Embedding(num_code, code_dim)  # shape: [num_code, c]
        for p in self.codebook.parameters():
            p.requires_grad = False

        # Removed project layer since in_dim == code_dim, identity mapping

        # Changed: initialize codebook via farthest point sampling in dimension code_dim
        pt.cuda.synchronize()
        t0 = time.time()
        candidat = ptnf.normalize(pt.randn(100000, code_dim).float().cuda())  # (100000, c)
        idx = __class__.farthest_point_sampling(candidat, k=num_code)
        self.codebook.weight.data[...] = candidat[idx].cpu()
        pt.cuda.synchronize()
        print(f"### [{__class__.__name__}] initialization time:", time.time() - t0)

        # normalize distances
        self.mu, self.sigma = __class__.chi_dist_mean_std(code_dim)
        self.register_buffer("std", pt.tensor(std, dtype=pt.float))

    def forward(self, input):
        """
        input: encoded sequence; shape=(b, n, c)  # Changed shape comment
        returns: zidx (b, n), quant (b, n, c)
        """
        zsoft, zidx = self.match(input)
        quant = self.select(zidx)
        return zidx, quant

    def match(self, encode):
        b, n, c = encode.shape  # Changed: unpack n instead of h,w

        # Changed: direct use of embedding weights (no project)
        e = self.codebook.weight                                                    # (m, c)
        z = encode.view(-1, c)                                                       # (b*n, c)
        with pt.no_grad():
            s = -pt.cdist(z, e, p=2)  # pairwise distances -> (b*n, m)

        simi = s.view(b, n, -1)                                                      # (b, n, m)
        simi = (simi - self.mu) / self.sigma                                         # normalize

        if self.training and self.std > 0:
            zsoft = ptnf.gumbel_softmax(simi * self.std, 1, False, dim=-1)           # (b, n, m)
        else:
            zsoft = simi.softmax(-1)
        zidx = zsoft.argmax(-1)                                                      # (b, n)
        return zsoft, zidx

    def select(self, zidx):
        quant = self.codebook(zidx)                                                  # (b, n, c)
        return quant

    @staticmethod
    def chi_dist_mean_std(c):
        mean = math.sqrt(2) * math.exp(math.lgamma((c + 1) / 2) - math.lgamma(c / 2))
        std = math.sqrt((c - mean**2))
        return mean, std

    @staticmethod
    @pt.no_grad()
    def farthest_point_sampling(points, k):
        """Farthest Point Sampling."""
        n, c = points.shape
        device = points.device

        selected = pt.empty(k, dtype=pt.long, device=device)
        distances = pt.full((n,), float("inf"), device=device)

        selected[0] = pt.randint(0, n, (1,)).item()
        distances = pt.cdist(points[selected[0]][None], points)[0]

        @pt.compile
        def loop_step(i, points, distances, selected):
            score = distances.clone()
            score[selected[:i]] = -float("inf")
            next_idx = score.argmax()
            selected[i] = next_idx
            distances2 = pt.cdist(points[next_idx][None], points)[0]
            distances = distances.minimum(distances2)

        for i in range(1, k):
            loop_step(i, points, distances, selected)
        return selected



# class VQVAEZ(nn.Module):
#     """"""

#     def __init__(self, encode, decode, quant, alpha=0.0, retr=True):
#         super().__init__()
#         self.encode = encode
#         self.decode = decode
#         self.quant = quant
#         self.register_buffer("alpha", pt.tensor(alpha, dtype=pt.float))  # 0.5->0->0
#         self.retr = retr  # return residual or not

#     def forward(self, input):
#         """
#         input: image; shape=(b,c,h,w)
#         """
#         encode = self.encode(input)
#         zidx, quant = self.quant(encode.permute(0, 2, 3, 1))  # bhw bhwc
#         quant = quant.permute(0, 3, 1, 2)  # bchw
#         residual = quant
#         decode = None
#         if self.decode:
#             if self.alpha > 0:  # no e.detach not converge if align residual to encode
#                 residual = encode * self.alpha + quant * (1 - self.alpha)
#             ste = __class__.naive_ste(encode, residual)
#             decode = self.decode(ste)
#         if self.retr:
#             return encode, zidx, quant, residual, decode
#         else:
#             return encode, zidx, quant, decode

#     @staticmethod
#     def naive_ste(encode, quant):
#         return encode + (quant - encode).detach()
#         # Rotate STE in "Restructuring Vector Quantization with the Rotation Trick": bad.


# class QuantiZ(nn.Module):


#     def __init__(self, num_code, code_dim, in_dim=1024, std=0):
#         super().__init__()
#         self.num_code = num_code
#         self.code_dim = code_dim

#         # adapted from SimVQ; IBQ is hard to converge
#         self.codebook = nn.Embedding(num_code, in_dim)
#         for p in self.codebook.parameters():
#             p.requires_grad = False
#         self.project = nn.Linear(in_dim, code_dim)
#         # original solution
#         # nn.init.normal_(self.codebook.weight, mean=0, std=code_dim**-0.5)

#         # select roughly equi-distant ones  # little improvement
#         pt.cuda.synchronize()
#         t0 = time.time()
#         candidat = ptnf.normalize(pt.randn(100000, in_dim).float().cuda())
#         idx = __class__.farthest_point_sampling(candidat, k=num_code)
#         self.codebook.weight.data[...] = candidat[idx].cpu()  # ~N(0,1/c.sqrt)
#         pt.cuda.synchronize()
#         print(f"### [{__class__.__name__}]", time.time() - t0)

#         # normalize and re-scale simiarities (negative distances)
#         self.mu, self.sigma = __class__.chi_dist_mean_std(code_dim)
#         self.register_buffer("std", pt.tensor(std, dtype=pt.float))

#     def forward(self, input):
#         """
#         - input: encoded feature; shape=(b,h,w,c)
#         - zidx: indexing tensor; shape=(b,h,w)
#         - quant: quantized feature; shape=(b,h,w,c)
#         """
#         zsoft, zidx = self.match(input)
#         quant = self.select(zidx)
#         return zidx, quant

#     def match(self, encode):
#         b, h, w, c = encode.shape

#         if any(_.requires_grad for _ in self.project.parameters()):
#             self.__e = self.project(self.codebook.weight)  # (m,c)
#         else:
#             if not hasattr(self, "__e"):  # to save computation in evaluation
#                 self.__e = self.project(self.codebook.weight)
#         e = self.__e

#         z = encode.flatten(0, -2)  # (b,h,w,c)->(b*h*w,c)
#         with pt.no_grad():  # cdist > cos-match
#             s = -pt.cdist(z, e, p=2)  # (b*h*w,m)

#         simi = s.unflatten(0, [b, h, w])  # (b,h,w,m)
#         # ``mean`` has no effects on gumbel while ``std`` has
#         simi = (simi - self.mu) / self.sigma  # ~N(0,1*std)

#         if self.training and self.std > 0:
#             zsoft = ptnf.gumbel_softmax(simi * self.std, 1, False, dim=-1)  # (b,h,w,m)
#         else:
#             zsoft = simi.softmax(-1)
#         zidx = zsoft.argmax(-1)  # (b,h,w)
#         return zsoft, zidx

#     def select(self, zidx):
#         quant = self.__e[zidx]  # (b,h,w,c)
#         return quant

#     @staticmethod
#     def chi_dist_mean_std(c):
#         """
#         Euclidian distances between two Gaussian distributions follows a Chi distribution.  # TODO XXX
#         """
#         mean = math.sqrt(2) * math.exp(math.lgamma((c + 1) / 2) - math.lgamma(c / 2))
#         std = math.sqrt((c - mean**2))
#         return mean, std

#     @staticmethod
#     @pt.no_grad()
#     def farthest_point_sampling(points, k):
#         """Farthest Point Sampling."""
#         n, c = points.shape
#         device = points.device

#         selected = pt.empty(k, dtype=pt.long, device=device)
#         distances = pt.full((n,), float("inf"), device=device)

#         selected[0] = pt.randint(0, n, (1,)).item()
#         distances = pt.cdist(points[selected[0]][None], points)[0]

#         @pt.compile
#         def loop_step(i, points, distances, selected):
#             score = distances.clone()
#             score[selected[:i]] = -float("inf")  # exclude already selected points
#             next_idx = score.argmax()  # select the index with the highest score
#             selected[i] = next_idx
#             distances2 = pt.cdist(points[next_idx][None], points)[0]
#             distances = distances.minimum(distances2)  # update selection's distances

#         for i in range(1, k):
#             loop_step(i, points, distances, selected)
#         return selected
    
    """
    Why normalize and re-scale simiarities (negative distances)?
    ```
    import matplotlib.pyplot as plt
    import numpy as np
    import torch as pt

    def unchanged_selection_probability(x, g, t):
        i0 = x.argmax(-1)
        o = (x + g) / t
        i1 = o.argmax(-1)
        return (i0 == i1).float().mean().item()

    x = pt.randn(64 * 16 * 16, 4096).float().cuda()  # ~N(0, 1)
    g = (  # gumbel noise
        -pt.empty_like(x, memory_format=pt.legacy_contiguous_format)
        .exponential_()
        .log()
    )

    fig, axs = plt.subplots(2, 1)

    s = np.exp(np.linspace(np.log(0.1), np.log(100), num=100))  # ``std``
    ps = [unchanged_selection_probability(x * _, g, t=1.0) for _ in s]
    axs[0].plot(s, ps)
    # 1.0000,0.59%
    # 1.0723,0.72%
    # 1.1498,0.90%
    # 1.2328,1.08%
    # 1.3219,1.37%
    # 1.4175,1.75%
    # 1.5199,2.20%
    # 1.6298,2.79%
    # 1.7475,3.57%
    # 1.8738,4.54%
    # 2.0092,5.78%
    # 2.1544,7.36%
    # 2.3101,9.26%
    # 2.4771,11.56%
    # 2.6561,14.26%
    # 2.8480,17.29%
    # 3.0539,20.71%
    # 3.2745,24.33%
    # 3.5112,28.13%
    # 3.7649,32.03%
    # 4.0370,35.96%
    # 4.3288,39.86%
    # 4.6416,43.67%
    # 4.9770,47.25%
    # 5.3367,50.63%
    # 5.7224,53.99%
    # 6.1359,57.14%
    # 6.5793,60.04%
    # 7.0548,62.78%
    # 7.5646,65.33%
    # 8.1113,67.78%
    # 8.6975,70.00%
    # 9.3260,72.10%
    # 10.0000,74.06%
    # 10.7227,75.85%
    # 11.4976,77.51%
    # 12.3285,79.08%
    # 13.2194,80.55%
    # 14.1747,81.86%
    # 15.1991,83.11%
    # 16.2975,84.22%
    # 17.4753,85.34%
    # 18.7382,86.35%
    # 20.0923,87.23%
    # 21.5443,88.08%
    # 23.1013,88.89%
    # 24.7708,89.66%
    # 26.5609,90.35%
    # 28.4804,91.00%
    # 30.5386,91.62%
    # 32.7455,92.19%
    # 35.1119,92.74%
    # 37.6494,93.22%
    # 40.3702,93.66%
    # 43.2876,94.07%
    # 46.4159,94.48%
    # 49.7702,94.87%
    # 53.3670,95.21%
    # 57.2237,95.53%
    # 61.3591,95.84%
    # 65.7933,96.12%
    # 70.5480,96.36%
    # 75.6463,96.61%
    # 81.1131,96.84%
    # 86.9749,97.05%
    # 93.2603,97.26%
    # 100.0000,97.44%

    t = np.exp(np.linspace(np.log(0.001), np.log(10), num=100))  # ``tau``
    pp = [unchanged_selection_probability(x * 10, g, _) for _ in t]
    axs[1].plot(s, pp)  # not change

    plt.show()
    ```
    """


class VQVAEZGrouped(VQVAEZ):

    def __init__(self, encode, decode, quant, project, alpha=0.0, retr=True, eaq=False):
        super().__init__(encode, decode, quant, alpha)
        self.project = project  # type: LinearPinv
        self.retr = retr  # return residual or not
        self.eaq = eaq  # encode as quant, for dfz s2 only

    def forward(self, input):
        encode = self.encode(input)
        encode_ = encode.permute(0, 2, 3, 1)
        if self.project:
            encode_ = self.project(encode_, pinv=True)
        zsoft_, zidx_ = __class__.g_match(self.quant, encode_)
        quant_ = __class__.g_select(self.quant, zidx_)
        residual_ = quant_
        if self.alpha > 0:
            residual_ = encode_ * self.alpha + quant_ * (1 - self.alpha)
        ste_ = __class__.naive_ste(encode_, residual_)
        quant = ste_
        if self.project:
            quant = self.project(quant)

        encode_ = encode_.permute(0, 3, 1, 2)
        # zidx = tuple_to_number(zidx_, [_.num_code for _ in self.quant], -1)
        zidx = zidx_.permute(0, 3, 1, 2)
        quant_ = quant_.permute(0, 3, 1, 2)
        residual_ = residual_.permute(0, 3, 1, 2)
        quant = quant.permute(0, 3, 1, 2)

        decode = None
        if self.decode:
            decode = self.decode(quant)
            return encode_, zidx, quant_, residual_, decode
        if self.eaq:
            quant = encode
        if self.retr:
            return encode, zidx, quant, None, decode
        else:
            return encode, zidx, quant, decode

    @staticmethod
    def g_match(quantz: nn.ModuleList, encode):
        """
        quantz: [QuantiZ,..]
        encode: shape=(b,h,w,c)
        zsoft: shape=(b,h,w,g*cg), cg=c**-g
        zidx: shape=(b,h,w,g)
        """
        zsoft = []
        zidx = []
        start = 0
        for g, quant_g in enumerate(quantz):
            end = start + quant_g.code_dim
            encode_g = encode[:, :, :, start:end]
            start = end
            zsoft_g, zidx_g = quant_g.match(encode_g)
            zsoft.append(zsoft_g)
            zidx.append(zidx_g)
        assert end == encode.size(-1)
        zsoft = pt.concat(zsoft, -1)
        zidx = pt.stack(zidx, -1)
        return zsoft, zidx

    @staticmethod
    def g_select(quantz: nn.ModuleList, zidx):
        """
        quantz: [QuantiZ,..]
        zidx: indexes, shape=(b,h,w,g)
        output: shape=(b,h,w,c)
        """
        output = []
        for g, quant_g in enumerate(quantz):
            idx_g = zidx[:, :, :, g]
            output_g = quant_g.select(idx_g)
            output.append(output_g)
        assert len(quantz) == zidx.size(-1)
        output = pt.cat(output, -1)
        return output


class VQVAEZMultiScale(VQVAEZ):

    def __init__(
        self,
        encode,
        decode,
        quant,
        project,
        num_scale=3,
        alpha=0,
        retr=True,
        eaq=False,
        tau=0,
    ):
        super().__init__(encode, decode, quant, alpha)
        self.project = project  # type: LinearPinv
        self.num_scale = num_scale
        self.retr = retr  # return residual or not
        self.eaq = eaq  # encode as quant, for dfz s2 only
        self.register_buffer("tau", pt.tensor(tau, dtype=pt.float))

    def forward(self, input):
        ### resize
        inputs = [input] + [  # s*(b,c,h,w)
            ptnf.interpolate(input, scale_factor=2**-_, mode="bilinear")
            for _ in range(1, self.num_scale)
        ]
        ### encode
        encodes = [self.encode(_) for _ in inputs]
        ### project-up
        encodes_ = [_.permute(0, 2, 3, 1) for _ in encodes]
        if self.project:
            encodes_ = [self.project(_, pinv=True) for _ in encodes_]
        ### quant: match and select
        zidxs_, encodes_, quants_ = __class__.ms_fuse(  # s*(b,h,w) s*(b,h,w,c)x2
            self.quant, encodes_, self.tau
        )
        ### residual
        residuals_ = quants_
        if self.alpha > 0:
            residuals_ = [
                e * self.alpha + q * (1 - self.alpha) for e, q in zip(encodes_, quants_)
            ]
        ### approx grad
        stes_ = [__class__.naive_ste(e, r) for e, r in zip(encodes_, residuals_)]
        ### project-down
        quants = stes_
        if self.project:
            quants = [self.project(_) for _ in quants]

        encodes_ = [_.permute(0, 3, 1, 2) for _ in encodes_]
        # zidxs = [tuple_to_number(_, [self.quant[0].num_code] * 2, -1) for _ in zidxs_]
        zidxs = [_.permute(0, 3, 1, 2) for _ in zidxs_]
        quants_ = [_.permute(0, 3, 1, 2) for _ in quants_]
        residuals_ = [_.permute(0, 3, 1, 2) for _ in residuals_]
        quants = [_.permute(0, 3, 1, 2) for _ in quants]

        ### decode
        decodes = [None] * self.num_scale
        if self.decode:
            decodes = [self.decode(_) for _ in quants]
            return encodes_, zidxs, quants_, residuals_, decodes
        ### else:  # self.decode is None
        if self.eaq:
            quants = encodes
        if self.retr:
            return encodes[0], zidxs[0], quants[0], None, decodes[0]
        else:
            return encodes[0], zidxs[0], quants[0], decodes[0]

    @staticmethod
    def ms_fuse(quantzs: nn.ModuleList, encodes: list, tau):
        s = len(encodes)
        assert len(quantzs) == 1 + s  # shared*1 + specified*s
        b, h, w, c = encodes[0].shape
        ch = c // 2
        assert c % 2 == 0

        zidx1s = []
        encode1s = []
        quant1s = []

        for i1 in range(s):  # fuz234 > nofuz234_fuzto2
            encode1_ = []
            for j1, encode in enumerate(encodes):  # s*(b,h,w,c)
                e1_ = encode[:, :, :, :].permute(0, 3, 1, 2)
                if j1 == i1:
                    pass  # (b,h,w,c/2)
                elif j1 < i1:  # avg vs max: max seems to favor fg
                    e1_ = ptnf.avg_pool2d(e1_, 2 ** (i1 - j1))
                else:
                    e1_ = ptnf.upsample(
                        e1_, scale_factor=2 ** (j1 - i1), mode="nearest"
                    )
                encode1_.append(e1_.permute(0, 2, 3, 1))

            encode1_ = pt.stack(encode1_)  # (s,b,h,w,c/2)
            zsoft1_, zidx1_ = quantzs[0].match(  # (s*b,h,w,m) (s*b,h,w)
                encode1_.flatten(0, 1)
            )
            zsoft1_ = zsoft1_.unflatten(0, [s, b])
            # zidx1_ = zidx1_.unflatten(0, [s, b])

            zsoft11, zidx11 = zsoft1_.max(-1)  # (s,b,h,w,m) -> (s,b,h,w)
            if tau > 0:
                zsoft11 = ptnf.gumbel_softmax(zsoft11 / tau, dim=0)
            zidx12 = zsoft11.argmax(0)  # (s,b,h,w) -> (b,h,w)
            zidx1 = zidx11.gather(0, zidx12[None, :, :, :])[0]
            encode1 = encode1_.gather(  # (s,b,h,w,c/2) -> (b,h,w,c/2)
                0, zidx12[None, :, :, :, None].expand(-1, -1, -1, -1, c)
            )[0]
            quant1 = quantzs[0].select(zidx1)

            zidx1s.append(zidx1)  # s*(b,h,w)
            encode1s.append(encode1)  # s*(b,h,w,c/2)
            quant1s.append(quant1)  # .to(dtype))  # s*(b,h,w,c/2)

        zidx2s = []
        encode2s = []
        quant2s = []

        for i2 in range(s):
            encode2 = encodes[i2][:, :, :, :]
            zsoft2, zidx2 = quantzs[1 + i2].match(encode2)
            quant2 = quantzs[1 + i2].select(zidx2)
            zidx2s.append(zidx2)
            encode2s.append(encode2)
            quant2s.append(quant2)  # .to(dtype))

        zidxs = [pt.stack([u, v], -1) for u, v in zip(zidx1s, zidx2s)]  # 2s*(b,h,w,g)
        encodes = [(u + v) / 2 for u, v in zip(encode1s, encode2s)]  # 2s*(b,h,w,c)
        quants = [(u + v) / 2 for u, v in zip(quant1s, quant2s)]  # 2s*(b,h,w,c)
        return zidxs, encodes, quants


class LinearPinv(nn.Module):
    """Supported shape: (b,n,c) or (b,h,w,c)."""

    def __init__(self, in_channel, out_channel, bias=True):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        linear = nn.Linear(in_channel, out_channel, bias)
        if bias:
            nn.init.zeros_(linear.bias)
        self.weight = linear.weight  # weight-norm always bad
        self.bias = linear.bias

    def forward(self, input, pinv=False):
        if pinv:
            shape = input.shape[1:-1]
            return (
                pt.linalg.lstsq(  # same as A.pinv() @ B  # (b,c,d) (b,c,h*w)
                    self.weight[None, :, :],
                    (input - self.bias).flatten(1, -2).permute(0, 2, 1),
                )
                .solution.permute(0, 2, 1)
                .unflatten(1, shape)
                .contiguous()
            )
        return ptnf.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return f"{self.in_channel}, {self.out_channel}"
