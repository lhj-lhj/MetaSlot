import math
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf
from scipy.optimize import linear_sum_assignment
import numpy as np
from .basic import MLP


class MetaSlot(nn.Module):
    """TODO XXX modularization/cgv: correct the wrong implementation!"""

    def __init__(
        self, num_iter, embed_dim, ffn_dim, dropout=0, kv_dim=None, trunc_bp=None, codebook_size = 512, \
            clust_prob: float = 0.02, buffer_capacity = 672, vq_std=1.0, vq_type='kmeans', \
            if_noise = True, if_mask = True, if_proto = True, if_downstream = False, timeout = 4096
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
        self.vq_type = vq_type
        self.if_downstream = if_downstream
        
        if vq_type=='kmeans':
            self.vq = VQ(codebook_size = codebook_size, embed_dim=embed_dim, buffer_capacity = buffer_capacity, timeout=timeout)
            
        self.register_buffer("clust_prob", pt.tensor(clust_prob, dtype=pt.float))
        self.if_mask = if_mask
        self.if_noise = if_noise
        self.if_proto = if_proto
        
    def step(self, q, k, v, smask = None):
        b, n, c = q.shape
        x = q
        q = self.norm1q(q)
        q = self.proj_q(q)
        u, a = __class__.inverted_scaled_dot_product_attention(q, k, v, smask)
        y = self.rnn(u.flatten(0, 1), x.flatten(0, 1)).view(b, n, -1)
        z = self.norm2(y)
        q = y + self.ffn(z)  # droppath on ffn seems harmful
        return q, a
    
    def from_slots_get_initial_slots(self, slots, indices):
        
        if len(indices.size()) == 3:
            indices = indices.squeeze(-1)
        
        # 初始化 smask, True 表示第一次出现，False 表示重复 
        smask = pt.ones_like(indices, dtype=pt.bool)

        # 遍历每个 batch 检测重复索引
        for b in range(indices.shape[0]):
            seen = {}
            for i in range(indices.shape[1]):
                idx_val = indices[b, i].item()
                if idx_val in seen:
                    # 如果重复，将对应的 slot 置为负无穷，并将 smask 对应位置置为 False
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
        if self.if_noise is False:
            k = self.proj_k(kv)
            v = self.proj_v(kv)
            
        q_d = query.detach()
        
        for i in range(self_num_iter):
            if self.if_noise:
                k,v = self.noisy(kv, i, self_num_iter)
            q_d, a = self.step(q_d, k, v, smask)
        
        q_vq, zidx = self.vq.codebook(q_d.detach())
        
        self.clust_prob = pt.clamp(self.clust_prob * 1.001, max=1)
        if np.random.random() > self.clust_prob or self.if_mask is False:
            smask = None
            if self.if_proto:
                q_d = q_vq
        else:
            if self.if_proto:
                q_d, smask = self.from_slots_get_initial_slots(q_vq, zidx)
            else:
                q_d, smask = self.from_slots_get_initial_slots(q_d, zidx)
                
        for i in range(self_num_iter):
            if self.if_noise:
                k,v = self.noisy(kv, i, self_num_iter)
            if i + 1 == self_num_iter:
                q = q_d + query - query.detach()
                q, a = self.step(q, k, v, smask)
            else:
                q_d, a = self.step(q_d, k, v, smask)

        slots_vq_2, zidx_slots_2 = self.vq.update_codebook(q.detach(), smask=smask)
        
        if self.if_downstream:
            fidx = zidx.clone()
            fidx[~smask] = -1
            return q, a, fidx
        else:
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


def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
    """``ptnf.gumbel_softmax`` is much worse than this (77->70@iou), why?"""
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
        n = self.templat.weight.size(0)  # good to vqvae pretrain but bad to dvae
        # 均匀分布初始化码本权重
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
    def cluster(self, latent, max_iter=100):  # always harmful
        """Straightening Out the Straight-Through Estimator: Overcoming Optimization Challenges in Vector Quantized Networks

        latent: shape=(b,c,h,w)
        """
        assert self.training  # if not self.training: return
        if not hasattr(self, "cluster_flag"):  # only once
            # 零维张量：没有任何维度，只存储一个单一的数值
            self.cluster_flag = pt.zeros([], dtype=pt.bool, device=latent.device)
        if self.cluster_flag:
            return
        # 在 PyTorch 中，每个 torch.Tensor（包括模型参数）都有 .data 属性，
        # 它直接指向该张量的原始数据，不经过自动求导（autograd）机制。
        # [...] 是一种切片操作，表示“所有元素”。
        self.cluster_flag.data[...] = True
        # latent = latent.permute(0, 2, 3, 1).flatten(0, -2)  # (b,h,w,c) -> (b*h*w,c)
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
    def replace(self, latent, zidx, rate=0.8, rho=1e-2, timeout=2048, cluster=0.95,     ):
        # cluster = cluster + round(self.step / 300000 * 0.09, 2)
        # if self.step < 200000:
        #     self.step = self.step + 1
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
        if not hasattr(self, "replace_rate"):  # only once
            self.replace_rate = pt.as_tensor(
                rate, dtype=latent.dtype, device=latent.device
            )
        # 初始化 replace_cnt（用于计数每个 code 被激活的“冷却时间”）以及替换率变量
        assert 0 <= self.replace_rate <= 1
        if self.replace_rate == 0:
            return
        if len(latent.size()) > 2:
            latent = latent.view(-1, self.embed_dim)
        m = latent.size(0)
        timeout = math.ceil(timeout * self.num_embed / m)  # from #vector to #iter
        if not hasattr(self, "replace_cnt"):  # only once
            self.replace_cnt = pt.ones(
                self.num_embed, dtype=pt.int, device=latent.device
            )
            # self.replace_cnt = self.replace_cnt * timeout
            self.replace_cnt = self.replace_cnt

        # 根据输入向量数量 m 调整 timeout 参数，从“向量数量”转换为“迭代步数”
        assert 0 <= cluster <= 1
        if self.replace_rate > 0 and cluster > 0:  # cluster update rate
            # m 必须不少于 codebook 中的总数 self.num_embed
            # print("m: ", m, "self.num_embed: ", self.num_embed) # m:  224 self.num_embed:  512
            assert m >= self.num_embed
            # 如果还没有初始化 replace_centroid（记录当前聚类中心）
            if not hasattr(self, "replace_centroid"):
                # 使用 k-means 聚类，将 latent 中的向量聚类为 self.num_embed 类，
                # 初始中心以 self.templat.weight.data 为初始值，最多迭代 100 次；
                # kmeans_pt 返回 (loss, centroid)，这里取 centroid 作为 replace_centroid
                self.replace_centroid = __class__.kmeans_pt(
                    latent,
                    self.num_embed,
                    self.templat.weight.data,  # 这里用当前 codebook 权重作为初始中心
                    max_iter=100,
                )[1]
            else:
                # 如果已经存在 replace_centroid，则执行一次短迭代更新（max_iter=1）
                centroid = __class__.kmeans_pt(
                    latent, self.num_embed, self.replace_centroid, max_iter=1
                )[1]
                # print("cluster: ", cluster)
                # print("centroid: ", centroid)
                # print("self.replace_centroid: ", self.replace_centroid)
                # 使用指数滑动平均 (EMA) 的方式平滑更新聚类中心，
                # cluster 参数决定新旧聚类中心的融合比例
                self.replace_centroid = (
                    self.replace_centroid * (1 - cluster) + centroid * cluster
                )
        assert self.replace_cnt.min() >= 0
        self.replace_cnt -= 1
        # reset cnt of recently used codes
        active_idx = pt.unique(zidx)
        self.replace_cnt.index_fill_(0, active_idx, timeout)
        # print("self.replace_cnt: ", self.replace_cnt)
        # reset value of unused codes
        dead_idx = (self.replace_cnt == 0).argwhere()
        dead_idx = dead_idx[:, 0]
        # print("dead_idx: ", dead_idx)
        # dead_idx = (self.replace_cnt == 0).argwhere()[:, 0]  # (n,)->(n,1)->(n,)
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
            
            dist = __class__.euclidean_distance(latent, self.templat(active_idx))
            ridx = dist.mean(1).topk(min(num_dead, m), sorted=False)[1]
            if mult > 1:
                ridx = ridx.tile(mult)[:num_dead]
            replac = latent[ridx]
            
            ### policy: most similar centriod to self from input -- VQ-NeRV: A Vector Quantized Neural Representation for Videos

            # dist = __class__.euclidean_distance(
            #     self.templat.weight.data[dead_idx], self.replace_centroid
            # )
            # row_idx, col_idx = linear_sum_assignment(dist.detach().cpu())
            # replac = self.replace_centroid[pt.from_numpy(col_idx).to(latent.device)]

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
        
        # 随机选择了 m 个样本中的 num_cluster 个索引，用于初始化聚类中心时随机抽取样本。
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
            # print(
            #     f"[kmeans_pt] {cnt}, {shift:.4f}, {dmin.mean().item():.4f}, {center.max().item():.4f} {center.min().item():.4f} {center.norm(2, 1).mean():.4f}"
            # )
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
    def match_encode_with_templat(encode, templat, sample, tau=1, detach="encode", metric="l2"):
        """
        对每个 encode 向量，从 templat 中找到最匹配的向量索引，并输出软分配概率。
        
        :param encode: Tensor，形状为 (b, n, emb_dim)
        :param templat: Tensor，形状为 (m, emb_dim)
        :param sample: bool，是否使用 gumbel_softmax 采样
        :param tau: 温度参数
        :param detach: str，如果为 "encode" 则 detach encode；如果为 "templat" 则 detach templat
        :param metric: str，"l2" 表示使用欧氏距离，"cosine" 表示使用余弦相似度，默认为 "l2"
        :return:
            - zsoft: Tensor，形状为 (b, n, m)，表示每个 encode 对应 templat 的软分配概率
            - zidx: Tensor，形状为 (b, n)，每个 encode 对应 templat 的索引（取最大概率）
        """
        if detach == "encode":
            encode = encode.detach()
        elif detach == "templat":
            templat = templat.detach()

        if metric == "l2":
            # 计算欧氏距离，先求每个向量的平方和，形状调整后广播到 (b, n, m)
            dist = (
                encode.pow(2).sum(-1, keepdim=True)
                - 2 * encode @ templat.t()
                + templat.pow(2).sum(-1, keepdim=True).t()
            )
            # 距离越小越好，因此取负作为 logits
            logits = -dist
        elif metric == "cosine":
            # 先归一化再计算余弦相似度，结果形状为 (b, n, m)
            encode_norm = pt.nn.functional.normalize(encode, dim=-1)
            templat_norm = pt.nn.functional.normalize(templat, dim=-1)
            logits = encode_norm @ templat_norm.t()
        else:
            raise ValueError("Unsupported metric type. Choose 'l2' or 'cosine'.")

        # 根据 sample 参数选择是否使用 gumbel_softmax（硬采样）
        if sample and tau > 0:
            zsoft = pt.nn.functional.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        else:
            zsoft = logits.softmax(dim=-1)

        # 对每个 encode 选择最佳 templat 索引（在候选维度上取 argmax）
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
        source = source.split(split_size)  # type: list
        dist = []
        for s in source:
            # 对每个分块 s，使用 pt.cdist 计算该块中每个样本与 target 中所有样本之间的欧氏距离
            d = pt.cdist(s, target, p=2)  # (m2,n);
            dist.append(d)
        dist = pt.concat(dist)  # (m,n)
        return dist

class VQ(nn.Module):
    """"""

    def __init__(self, codebook_size, embed_dim, alpha=0.0, retr=True, buffer_capacity = None, timeout = 2048):
        super().__init__()
        self.register_buffer("alpha", pt.tensor(alpha, dtype=pt.float))  # 0.5->0->0
        self.retr = retr  # return residual or not
        self.codebook = Codebook(num_embed=codebook_size, embed_dim=embed_dim)
        self.embed_dim = embed_dim
        if buffer_capacity is not None:
            self.buffer_capacity = buffer_capacity
            self.register_buffer("latent_buffer", pt.empty(self.buffer_capacity, embed_dim).normal_())
            self.register_buffer("idx_buffer", pt.empty(self.buffer_capacity, dtype=pt.long))
            # buffer_ptr 用于标记当前更新到的位置
            
            self.register_buffer("buffer_ptr", pt.tensor(0, dtype=pt.long))
        self.timeout = timeout
            
    def forward(self, encode, is_update=True):
        """
        input: image; shape=(b,w,embedding_dim)
        """
        b,c,embedding_dim = encode.size()
        encode_flat = encode.view(-1, embedding_dim)
        quant, zidx = self.codebook(encode_flat)
        residual = quant
        decode = None
        if self.alpha > 0:  # no e.detach not converge if align residual to encode
            residual = encode_flat * self.alpha + quant * (1 - self.alpha)
        ste = __class__.naive_ste(encode_flat, residual)
        ste = ste.view_as(encode)
        
        if self.training and is_update:
            encode_flat_d = encode_flat.detach()
            with pt.no_grad():
                if hasattr(self, 'latent_buffer'):
                    # 更新 latent_buffer 和 idx_buffer
                    self._update_buffer(encode_flat_d, zidx)
                    self.update(self.latent_buffer, zidx)
                else:
                    self.update(encode_flat_d, zidx)
        
        return ste, zidx.view(b,c)
    
    def update_codebook(self, encode, is_update=True, smask=None):
        """
        input: encode; shape = (b, c, embedding_dim)
        smask: shape = (b, c), True 表示该位置未被 mask，可用于更新
        """
        b, c, embedding_dim = encode.size()
        encode_flat = encode.view(-1, embedding_dim)
        quant, zidx = self.codebook(encode_flat)
        residual = quant

        if self.alpha > 0:
            residual = encode_flat * self.alpha + quant * (1 - self.alpha)
        ste = __class__.naive_ste(encode_flat, residual)
        ste = ste.view_as(encode)
        # print("self.training: ", self.training)
        # print("is_update: ", is_update)
        if self.training and is_update:
            with pt.no_grad():
                if smask is not None:
                    # 展平 smask 并获取未被 mask 掉的位置索引
                    smask_flat = smask.view(-1)
                    valid_idx = smask_flat.nonzero(as_tuple=False).squeeze(1)
                    encode_flat_d = encode_flat[valid_idx].detach()
                    zidx_valid = zidx[valid_idx]
                    # print("encode_flat_d: ", encode_flat_d.size())
                    # print("zidx_valid: ", zidx_valid.size())
                else:
                    encode_flat_d = encode_flat.detach()
                    zidx_valid = zidx

                if hasattr(self, 'latent_buffer'):
                    self._update_buffer(encode_flat_d, zidx_valid)
                    self.update(self.latent_buffer, zidx_valid)
                else:
                    self.update(encode_flat_d, zidx_valid)

        return ste, zidx.view(b, c)


    
    # def _update_buffer(self, new_latents, new_idx):
    #     """
    #     new_latents: [N, embed_dim]，其中 N = b*c
    #     """
    #     N = new_latents.size(0)
    #     ptr = int(self.buffer_ptr.item())
    #     # 如果当前剩余空间足够，直接写入
    #     if ptr + N <= self.buffer_capacity:
    #         self.latent_buffer[ptr:ptr+N] = new_latents
    #         self.idx_buffer[ptr:ptr+N] = new_idx
    #         new_ptr = ptr + N
    #     else:
    #         # 剩余空间不足，需要先填满剩余，再从头覆盖
    #         tail = self.buffer_capacity - ptr
    #         if tail > 0:
    #             self.latent_buffer[ptr:self.buffer_capacity] = new_latents[:tail]
    #         remaining = N - tail
    #         self.latent_buffer[0:remaining] = new_latents[tail:]
    #         new_ptr = remaining

    #     if new_ptr == self.buffer_capacity:
    #         new_ptr = 0

    #     self.buffer_ptr.fill_(new_ptr)
        
    def _update_buffer(self, new_latents, new_idx):
        """
        new_latents: [N, embed_dim]
        new_idx:     [N]
        """
        N = new_latents.size(0)
        C = self.buffer_capacity

        # 1. 生成 positions，并确保它是 LongTensor（scatter 的 index 要求 Long）
        arange = pt.arange(N, device=new_latents.device)
        positions = (self.buffer_ptr.long() + arange) % C  # Long by construction

        # 2. 对齐 dtype：把 new_latents 转成和 buffer 一样
        src_latents = new_latents.to(self.latent_buffer.dtype)
        self.latent_buffer.scatter_(
            dim=0,
            index=positions.unsqueeze(-1).expand(-1, src_latents.size(1)),
            src=src_latents
        )

        # 3. 同理对齐 idx 的 dtype（通常 idx_buffer 是 Long，也是 src 也得是 Long）
        src_idx = new_idx.to(self.idx_buffer.dtype)
        self.idx_buffer.scatter_(
            dim=0,
            index=positions,
            src=src_idx
        )

        # 4. 更新指针
        new_ptr = (self.buffer_ptr + N) % C
        self.buffer_ptr.copy_(new_ptr)

    
    def update(self, latent, idx):
        if len(latent.size()) > 2:
            latent = latent.view(-1, self.embed_dim)
        return self.codebook.replace(latent, idx, timeout=self.timeout)
        
    @staticmethod
    def naive_ste(encode, quant):
        return encode + (quant - encode).detach()
        # Rotate STE in "Restructuring Vector Quantization with the Rotation Trick": bad.

        
class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        initial_decay: float = 0.90,
        final_decay: float = 0.99,
        total_steps: int = 10000,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.codebook_size = codebook_size
        self.initial_decay = initial_decay
        self.final_decay = final_decay
        self.total_steps = total_steps  # 总步数，可根据实际情况调整
        self.decay = initial_decay  # 初始 decay 值
        self.eps = eps
        
        # 初始化代码本
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # EMA 统计量
        self.register_buffer("ema_cluster_size", pt.zeros(codebook_size))
        self.register_buffer("ema_embedding", self.embedding.weight.clone())
        self.register_buffer("global_step", pt.tensor(0, dtype=pt.long))
        
        self.codebook = self.embedding

    def forward(self, x: pt.Tensor):
        """
        x: [B, ..., embedding_dim]
        返回:
        quantized_st: 量化后并做了直通估计的输出，可直接送入后续网络
        commitment_loss: 给编码器的承诺损失
        """
        
        # 更新全局步数和 decay
        if self.training:
            self.global_step += 1
            # 线性插值计算 decay
            new_decay = self.initial_decay + (self.final_decay - self.initial_decay) * (
                self.global_step.float() / self.total_steps
            )
            # 限制最大值为 final_decay
            self.decay = min(new_decay.item(), self.final_decay)
            
        # [B*N, embedding_dim]
        flat_x = x.view(-1, self.embedding_dim)
        
        # 计算与代码本的距离 [B*N, codebook_size]
        distances = (
            flat_x.pow(2).sum(1, keepdim=True)
            - 2 * flat_x @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1, keepdim=True).t()
        )
        # 最近邻索引
        encoding_indices = pt.argmin(distances, dim=1)
        # 得到量化后的向量 [B*N, embedding_dim]
        quantized = self.embedding(encoding_indices)

        # ==============================
        # 1) 计算承诺损失
        # ==============================
        # 对量化向量做 stop-gradient，避免更新到代码本
        # 只让编码器的输出 x 去贴近 e (quantized.detach())
        commitment_loss = ptnf.mse_loss(flat_x, quantized.detach())

        # ==============================
        # 2) 直通估计（Straight-Through）
        # ==============================
        # 在前向过程中，用 quantized 代替 x，
        # 但反向传播时，对 quantized 的梯度设为 x 的梯度 (相当于把梯度"穿透"给编码器)。
        # 这样就能训练编码器，而不会训练代码本。
        quantized_st = flat_x + (quantized - flat_x).detach()
        quantized_st = quantized_st.view_as(x)

        if self.training:
            encodings = ptnf.one_hot(encoding_indices, self.codebook_size).float()
            cluster_size = encodings.sum(0)
            embedding_sum = encodings.t() @ flat_x

            with pt.no_grad():
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size * (1 - self.decay))
                self.ema_embedding.mul_(self.decay).add_(embedding_sum * (1 - self.decay))

                n = self.ema_cluster_size.sum()
                norm = (self.ema_cluster_size + self.eps) / (n + self.codebook_size * self.eps)
                self.embedding.weight.copy_(self.ema_embedding / norm.unsqueeze(1))
                
        return quantized_st, commitment_loss