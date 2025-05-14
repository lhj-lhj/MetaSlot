import math
from einops import rearrange, repeat
from scipy.optimize import linear_sum_assignment
import numpy as np
import timm
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf

class MetaSlot(nn.Module):

    def __init__(
        self, num_iter, embed_dim, ffn_dim, dropout=0, kv_dim=None, trunc_bp=None, codebook_size = 512, clust_prob: float = 0.02, buffer_capacity = 672, vq_std=1.0, vq_type='kmeans'
    ):
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
        self.rnn = nn.GRUCell(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = MLP(embed_dim, [ffn_dim, embed_dim], None, dropout)
        self.vq_type = vq_type
        
        if vq_type=='kmeans':
            self.vq = VQ(codebook_size = codebook_size, embed_dim=embed_dim, buffer_capacity = buffer_capacity)
            
        self.register_buffer("clust_prob", pt.tensor(clust_prob, dtype=pt.float))
        
    def step(self, q, k, v, smask = None):
        b, n, c = q.shape
        x = q
        q = self.norm1q(q)
        q = self.proj_q(q)
        u, a = __class__.inverted_scaled_dot_product_attention(q, k, v, smask)
        y = self.rnn(u.flatten(0, 1), x.flatten(0, 1)).view(b, n, -1)
        z = self.norm2(y)
        q = y + self.ffn(z)
        return q, a
    
    def from_slots_get_initial_slots(self, slots, indices):
        
        if len(indices.size()) == 3:
            indices = indices.squeeze(-1)
        
        smask = pt.ones_like(indices, dtype=pt.bool)

        for b in range(indices.shape[0]):
            seen = {}
            for i in range(indices.shape[1]):
                idx_val = indices[b, i].item()
                if idx_val in seen:
                    slots[b, i] = 0
                    smask[b, i] = False
                else:
                    seen[idx_val] = True
        return slots, smask
        
    def noisy(self, kv, step, n_iters, weight = 0.5):
        alpha_i = weight * (1.0 - step / max(n_iters - 1, 1e-8))
        noise = pt.randn_like(kv) * alpha_i
        kv_noisy = kv + noise
        k = self.proj_k(kv_noisy)
        v = self.proj_v(kv_noisy)
        return k, v
    
    def forward(self, input, query, smask=None, num_iter=None):
        """
        input: in shape (b,h*w,c)
        query: in shape (b,n,c)
        smask: slots' mask, shape=(b,n), dtype=bool
        """
        self_num_iter = num_iter or self.num_iter
        kv = self.norm1kv(input)
        q_d = query.detach()
        
        for i in range(self_num_iter):
            k,v = self.noisy(kv, i, self_num_iter)
            q_d, a = self.step(q_d, k, v, smask)
        
        q_vq, zidx = self.vq.codebook(q_d.detach())
        
        # mask
        self.clust_prob = pt.clamp(self.clust_prob * 1.001, max=1)
        if np.random.random() > self.clust_prob or self.if_mask is False:
            smask = None
            q_d = q_vq
        else:
            q_d, smask = self.from_slots_get_initial_slots(q_vq, zidx)
                
        for i in range(self_num_iter):
            k,v = self.noisy(kv, i, self_num_iter)
            if i + 1 == self_num_iter: # bi-level
                q = q_d + query - query.detach()
                q, a = self.step(q, k, v, smask)
            else:
                q_d, a = self.step(q_d, k, v, smask)

        # update
        slots_vq_2, zidx_slots_2 = self.vq.update_codebook(q.detach(), smask=smask)
        
        return q, a
            
    @staticmethod
    def inverted_scaled_dot_product_attention(q, k, v, smask=None, eps=1e-5):
        scale = q.size(2) ** -0.5  # temperature
        logit = pt.einsum("bqc,bkc->bqk", q * scale, k)
        if smask is not None:
            logit = logit.where(smask[:, :, None], -pt.inf)
        a0 = logit.softmax(1)
        a = a0 / (a0.sum(2, keepdim=True) + eps)  # re-normalize over key
        o = pt.einsum("bqv,bvc->bqc", a, v)
        return o, a0


class MLP(nn.Sequential):
    """"""

    def __init__(self, in_dim, dims, ln: str = None, dropout=0):
        """
        - ln: None for no layernorm, 'pre' for pre-norm, 'post' for post-norm
        """
        assert ln in [None, "pre", "post"]

        num = len(dims)
        layers = []
        ci = in_dim

        if ln == "pre":
            layers.append(nn.LayerNorm(ci))

        for i, c in enumerate(dims):
            if i + 1 < num:
                block = [
                    nn.Linear(ci, c),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout else None,
                ]
            else:
                block = [nn.Linear(ci, c)]

            layers.extend([_ for _ in block if _])
            ci = c

        if ln == "post":
            layers.append(nn.LayerNorm(ci))

        super().__init__(*layers)

def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    eps = pt.finfo(logits.dtype).tiny
    gumbels = -(pt.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau
    y_soft = ptnf.softmax(gumbels, dim)
    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = pt.zeros_like(logits).scatter_(dim, index, 1.0)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft
    
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
        n = self.templat.weight.size(0)
        self.templat.weight.data.uniform_(-1 / n, 1 / n)
        self.step = 0

    def forward(self, input):
        if self.training:
            zsoft, zidx = self.match(input, True)
        else:
            zsoft, zidx = self.match(input, False)
        quant = self.select(zidx)
        return quant, zidx
        
    def select(self, idx):
        return self.templat.weight[idx]

    def match(self, encode, sample: bool, tau=1, detach="encode"):
        return __class__.match_encode_with_templat(
            encode, self.templat.weight, sample, tau, detach
        )

    @pt.no_grad()
    def cluster(self, latent, max_iter=100):
        assert self.training
        if not hasattr(self, "cluster_flag"):
            self.cluster_flag = pt.zeros([], dtype=pt.bool, device=latent.device)
        if self.cluster_flag:
            return
        self.cluster_flag.data[...] = True
        latent = latent.view(-1, self.embed_dim)
        n, c = latent.shape
        if n < self.num_embed:
            raise f"warmup samples should >= codebook size: {n} vs {self.num_embed}"
        print("clustering...")
        assign, centroid = __class__.kmeans_pt(
            latent, self.num_embed, max_iter=max_iter
        )
        self.templat.weight.data[...] = centroid
    
    @pt.no_grad()
    def replace(self, latent, zidx, rate=0.8, rho=1e-2, timeout=2048, cluster=0.95):
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
        assert self.training
        
        if not hasattr(self, "replace_cnt"):  # only once
            self.replace_cnt = pt.ones(
                self.num_embed, dtype=pt.int, device=latent.device
            )
            self.replace_rate = pt.as_tensor(
                rate, dtype=latent.dtype, device=latent.device
            )
        assert 0 <= self.replace_rate <= 1
        if self.replace_rate == 0:
            return
        if len(latent.size()) > 2:
            latent = latent.view(-1, self.embed_dim)
        m = latent.size(0)
        timeout = math.ceil(timeout * self.num_embed / m)
        assert 0 <= cluster <= 1
        if self.replace_rate > 0 and cluster > 0:
            assert m >= self.num_embed
            if not hasattr(self, "replace_centroid"):
                self.replace_centroid = __class__.kmeans_pt(
                    latent,
                    self.num_embed,
                    self.templat.weight.data,
                    max_iter=100,
                )[1]
            else:
                centroid = __class__.kmeans_pt(
                    latent, self.num_embed, self.replace_centroid, max_iter=1
                )[1]
                self.replace_centroid = (
                    self.replace_centroid * (1 - cluster) + centroid * cluster
                )
        assert self.replace_cnt.min() >= 0
        self.replace_cnt -= 1
        active_idx = pt.unique(zidx)
        self.replace_cnt.index_fill_(0, active_idx, timeout)
        dead_idx = (self.replace_cnt == 0).argwhere()
        dead_idx = dead_idx[:, 0]
        num_dead = dead_idx.size(0)
        
        if num_dead > 0:
            print("#", timeout, self.num_embed, m, dead_idx)
            mult = num_dead // m + 1
            
            dist = __class__.euclidean_distance(latent, self.templat(active_idx))
            ridx = dist.mean(1).topk(min(num_dead, m), sorted=False)[1]
            if mult > 1:
                ridx = ridx.tile(mult)[:num_dead]
            replac = latent[ridx]
            
            if rho > 0:
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
        split_size=64,
        replace=False,
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
            )
            dmin, assign = dist.min(1)
            center_old = center.clone()

            for cid in range(num_cluster):
                idx = assign == cid
                if not idx.any():
                    if replace:
                        idx = pt.randperm(m)[:num_cluster]
                    else:
                        continue
                cluster = X[idx]
                center[cid] = cluster.mean(0)

            shift = ptnf.pairwise_distance(center, center_old).mean().item()
            shifts.append(shift)
            shifts = shifts[-10:]
            if shift < tol or len(shifts) > 1 and np.std(shifts) == 0:
                break
            cnt = cnt + 1
            if max_iter > 0 and cnt >= max_iter:
                break

        return assign, center
    
    @staticmethod
    def match_encode_with_templat(encode, templat, sample, tau=1, detach="encode", metric="l2"):
        if detach == "encode":
            encode = encode.detach()
        elif detach == "templat":
            templat = templat.detach()

        if metric == "l2":
            dist = (
                encode.pow(2).sum(-1, keepdim=True)
                - 2 * encode @ templat.t()
                + templat.pow(2).sum(-1, keepdim=True).t()
            )
            logits = -dist
        elif metric == "cosine":
            encode_norm = pt.nn.functional.normalize(encode, dim=-1)
            templat_norm = pt.nn.functional.normalize(templat, dim=-1)
            logits = encode_norm @ templat_norm.t()
        else:
            raise ValueError("Unsupported metric type. Choose 'l2' or 'cosine'.")

        if sample and tau > 0:
            zsoft = pt.nn.functional.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        else:
            zsoft = logits.softmax(dim=-1)
            
        zidx = zsoft.argmax(dim=-1)
        return zsoft, zidx


    @staticmethod
    def euclidean_distance(source, target, split_size=64):
        """chunked cdist

        source: shape=(b,m,c) or (m,c)
        target: shape=(b,n,c) or (n,c)
        split_size: in case of oom; can be bigger than m
        dist: shape=(b,m,n) or (m,n)
        """
        assert source.ndim == target.ndim and source.ndim in [2, 3]
        source = source.split(split_size)
        dist = []
        for s in source:
            d = pt.cdist(s, target, p=2)
            dist.append(d)
        dist = pt.concat(dist)
        return dist

class VQ(nn.Module):

    def __init__(self, codebook_size, embed_dim, alpha=0.0, retr=True, buffer_capacity = None):
        super().__init__()
        self.register_buffer("alpha", pt.tensor(alpha, dtype=pt.float))
        self.retr = retr
        self.codebook = Codebook(num_embed=codebook_size, embed_dim=embed_dim)
        self.embed_dim = embed_dim
        if buffer_capacity is not None:
            self.buffer_capacity = buffer_capacity
            self.register_buffer("latent_buffer", pt.empty(self.buffer_capacity, embed_dim).normal_())
            self.register_buffer("idx_buffer", pt.empty(self.buffer_capacity, dtype=pt.long))
            
            self.register_buffer("buffer_ptr", pt.tensor(0, dtype=pt.long))
            
    def forward(self, encode, is_update=True):
        """
        input: image; shape=(b,w,embedding_dim)
        """
        b,c,embedding_dim = encode.size()
        encode_flat = encode.view(-1, embedding_dim)
        quant, zidx = self.codebook(encode_flat)
        residual = quant
        decode = None
        if self.alpha > 0:
            residual = encode_flat * self.alpha + quant * (1 - self.alpha)
        ste = __class__.naive_ste(encode_flat, residual)
        ste = ste.view_as(encode)
        
        if self.training and is_update:
            encode_flat_d = encode_flat.detach()
            with pt.no_grad():
                if hasattr(self, 'latent_buffer'):
                    self._update_buffer(encode_flat_d, zidx)
                    self.update(self.latent_buffer, zidx)
                else:
                    self.update(encode_flat_d, zidx)
        
        return ste, zidx.view(b,c)
    
    def update_codebook(self, encode, is_update=True, smask=None):
        b, c, embedding_dim = encode.size()
        encode_flat = encode.view(-1, embedding_dim)
        quant, zidx = self.codebook(encode_flat)
        residual = quant

        if self.alpha > 0:
            residual = encode_flat * self.alpha + quant * (1 - self.alpha)
        ste = __class__.naive_ste(encode_flat, residual)
        ste = ste.view_as(encode)
        if self.training and is_update:
            with pt.no_grad():
                if smask is not None:
                    smask_flat = smask.view(-1)
                    valid_idx = smask_flat.nonzero(as_tuple=False).squeeze(1)
                    encode_flat_d = encode_flat[valid_idx].detach()
                    zidx_valid = zidx[valid_idx]
                else:
                    encode_flat_d = encode_flat.detach()
                    zidx_valid = zidx

                if hasattr(self, 'latent_buffer'):
                    self._update_buffer(encode_flat_d, zidx_valid)
                    self.update(self.latent_buffer, zidx_valid)
                else:
                    self.update(encode_flat_d, zidx_valid)

        return ste, zidx.view(b, c)
        
    def _update_buffer(self, new_latents, new_idx):
        """
        new_latents: [N, embed_dim]
        new_idx:     [N]
        """
        N = new_latents.size(0)
        C = self.buffer_capacity
        
        arange = pt.arange(N, device=new_latents.device)
        positions = (self.buffer_ptr.long() + arange) % C

        src_latents = new_latents.to(self.latent_buffer.dtype)
        self.latent_buffer.scatter_(
            dim=0,
            index=positions.unsqueeze(-1).expand(-1, src_latents.size(1)),
            src=src_latents
        )

        src_idx = new_idx.to(self.idx_buffer.dtype)
        self.idx_buffer.scatter_(
            dim=0,
            index=positions,
            src=src_idx
        )

        new_ptr = (self.buffer_ptr + N) % C
        self.buffer_ptr.copy_(new_ptr)

    
    def update(self, latent, idx):
        if len(latent.size()) > 2:
            latent = latent.view(-1, self.embed_dim)
        return self.codebook.replace(latent, idx)
        
    @staticmethod
    def naive_ste(encode, quant):
        return encode + (quant - encode).detach()