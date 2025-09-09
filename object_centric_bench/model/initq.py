from copy import deepcopy

from einops import rearrange, repeat
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch as pt
import torch.nn as nn


class InitqDINOSAUR(nn.Module):

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        decode,
        clust_prob,  # 1 always merge
        thresh,  # [0, 2] merge distance threshold
    ):
        super().__init__()
        self.encode_backbone = encode_backbone
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.aggregat = aggregat
        self.decode = decode
        self.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat]
        )
        self.register_buffer("clust_prob", pt.tensor(clust_prob, dtype=pt.float))
        self.register_buffer("thresh", pt.tensor(thresh, dtype=pt.float))

    @staticmethod
    def reset_parameters(modules):
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.GRUCell):
                    if m.bias:
                        nn.init.zeros_(m.bias_ih)
                        nn.init.zeros_(m.bias_hh)

    def forward(self, input, condit=None):
        """
        - input: image, shape=(b,c,h,w)
        - condit: condition, shape=(b,n,c)
        """
        feature = self.encode_backbone(input).detach()  # (b,c,h,w)
        b, c, h, w = feature.shape
        encode = feature.permute(0, 2, 3, 1)  # (b,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b,h*w,c)
        encode = self.encode_project(encode)

        query = self.initializ(b if condit is None else condit)  # (b,n,c)
        slotz, attent = self.aggregat(encode, query)
        attent = rearrange(attent, "b n (h w) -> b n h w", h=h)
        ### <<< 2nd slot attention -- 俩解码头，需要考虑是否截断slotz梯度；还有单解码头，就不能截断slotz
        # query_2, smask = merge_slots(slotz, thresh=0.9985, drop=False)  # (b,n,c) (b,n)
        if self.training and np.random.random() > self.clust_prob:
            query_2 = slotz
            smask = pt.ones(b, query_2.size(1), dtype=pt.bool, device=query_2.device)
        else:
            query_2, smask = merge_slots_agglomerat(slotz, self.thresh)  # (b,n,c) (b,n)
        smask = pt.ones_like(smask)  # TODO XXX wrong masking in aggregat and mlpdec ???
        slotz_2, attent_2 = self.aggregat(encode, query_2, smask, num_iter=1)
        attent_2 = rearrange(attent_2, "b n (h w) -> b n h w", h=h)
        # if self.dec == 1:
        slotz = slotz_2
        attent = attent_2
        ### >>>

        clue = [h, w]
        recon, attent2 = self.decode(clue, slotz, smask)  # (b,h*w,c)
        recon = rearrange(recon, "b (h w) c -> b c h w", h=h)
        attent2 = rearrange(attent2, "b n (h w) -> b n h w", h=h)

        return feature, slotz, attent, attent2, recon
        # segment acc: attent < attent2


def merge_slots_agglomerat(slotz, thresh):
    b, n, c = slotz.shape
    dtype = slotz.dtype
    device = slotz.device

    if isinstance(thresh, pt.Tensor):
        thresh = thresh.item()
    with pt.no_grad():  # cosine distance (b,n,n)
        distz = 1 - pt.cosine_similarity(slotz[:, :, None, :], slotz[:, None, :, :], -1)
    labels0 = np.arange(n)

    def cluster_a_sample_by_distances(dist_matrix):
        # dist_matrix = dist_matrix.reshape(n, n)
        clust = AgglomerativeClustering(
            None,
            metric="precomputed",
            compute_full_tree=True,
            linkage="complete",
            distance_threshold=thresh,
        )
        clust.fit(dist_matrix)  # (n,n)
        labels = clust.labels_  # long  TODO XXX remove order keep order
        if labels.max() <= 1:
            labels = labels0
        return labels  # (n,) long

    clustz = list(map(cluster_a_sample_by_distances, distz.cpu().numpy()))
    # clustz = np.apply_along_axis(  # little speedup
    #     func1d=cluster_a_sample_by_distances,
    #     axis=1,
    #     arr=distz.cpu().flatten(1).numpy(),
    # )
    clustz = pt.from_numpy(np.array(clustz)).to(device)  # (b,n)

    adjacen = clustz[:, :, None] == clustz[:, None, :]  # (b,n,n)
    adjacen_ = adjacen.to(dtype)
    merge = adjacen_ / adjacen_.sum(-1, keepdim=True)  # bnn
    slotz1_ = pt.einsum("bqc,bqk->bkc", slotz, merge)  # bnc
    smask = ~adjacen.triu(diagonal=1).any(dim=1)  # bn
    slotz1 = slotz1_.where(smask[:, :, None], 0)
    # print(slotz1.shape, smask.shape)
    return slotz1, smask


def merge_slots(slotz0, thresh=0.9, drop=False):
    """
    - slotz: shape=(b,n,c), dtype=float
    - smask: shape=(b,n), dtype=bool
    """
    dtype = slotz0.dtype
    affin = pt.cosine_similarity(  # bnn
        slotz0[:, None, :, :], slotz0[:, :, None, :], dim=-1
    )
    binariz = affin > thresh  # bnn
    if not drop:
        binariz_ = binariz.to(dtype)
        merge = binariz_ / binariz_.sum(-1, keepdim=True)  # bnn
        # suppose no cases like a~b and b~c but a!~c
        slotz1_ = pt.einsum("bqc,bqk->bkc", slotz0, merge)  # bnc
    else:
        slotz1_ = slotz0
    smask_ = pt.triu(binariz, diagonal=1)  # bnn
    smask = ~smask_.any(dim=1)  # bn
    slotz1 = slotz1_.where(smask[:, :, None], 0)
    return slotz1, smask


####


from typing import Tuple, Type

import torch.nn.functional as F
from torch import nn, Tensor

from sam2.modeling.sam2_utils import MLP


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        point_embedding: Tensor,  # query
        image_embedding: Tensor,  # key
        image_pe: Tensor = 0,  # TODO XXX PositionEmbeddingRandom.forward(image_size)
    ) -> Tuple[Tensor, Tensor]:
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        # bs, c, h, w = image_embedding.shape
        # image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        # image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # Prepare queries
        queries = point_embedding  # (b,n,c)
        keys = image_embedding  # (b,h*w,c)

        # Apply transformer blocks and final layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
