from copy import deepcopy
import math
import random

from einops import rearrange, repeat
import numpy as np
import torch
import torch as pt
import torch.nn as nn
import torch.nn.functional as F

from .basic import (
    DINO2ViT,
    MLP,
    Interpolate,
    TransformerDecoder,
    TransformerDecoderLayer,
    Linear,
)

from .ocl import SlotAttention, NormalShared, NormalSeparat


class SPOTZero(nn.Module):

    def __init__(self, second_encoder=False):
        super().__init__()
        image_size = 256
        vfm_dim = 384
        num_slots = 7
        embed_dim = 256
        finetune_blocks_after = 100

        class DINO2(nn.Module):

            def __init__(self, resolut=[16, 16], norm9=False):
                super().__init__()
                self.model = torch.hub.load(
                    "facebookresearch/dinov2", "dinov2_vits14_reg"
                )
                self.norm9 = norm9
                self.resolut = resolut

            def forward(self, x):
                x = self.model.prepare_tokens_with_masks(x, None)
                for blk in self.model.blocks:
                    x = blk(x)
                    if self.norm9:
                        x = self.model.norm(x)
                x = x[:, 1 + self.model.num_register_tokens :, :]  # cls reg4 patch
                x = rearrange(x, "b (h w) c -> b c h w", h=self.resolut[0])
                return x

        self.encode_backbone0 = Interpolate(scale_factor=0.875, interp="bicubic")
        self.encode_backbone = DINO2()
        self.encode_backbone2 = None
        if second_encoder:
            self.encode_backbone2 = deepcopy(self.encode_backbone).eval()
        for param_name, param in self.encode_backbone.named_parameters():
            if "blocks" in param_name:
                block_id = int(param_name.split(".")[2])
                if block_id >= finetune_blocks_after:
                    param.requires_grad = True  # update by gradient
                else:
                    param.requires_grad = False  # not update by gradient
            else:
                param.requires_grad = False  # not update by gradient
        if self.encode_backbone2 is not None:
            for param in self.encode_backbone2.parameters():
                param.requires_grad = False  # not update by gradient

        self.encode_project = MLP(  # TODO XXX dims
            in_dim=vfm_dim, dims=[vfm_dim, vfm_dim], ln="pre", dropout=0.0
        )
        # TODO XXX
        # nn.init.kaiming_uniform_(self.encode_project[1].weight, nonlinearity="gelu")
        # nn.init.xavier_uniform_(self.encode_backbone[3].weight)

        self.initializ = NormalShared(num=num_slots, dim=embed_dim)
        # self.initializ = NormalSeparat(num=num_slots, dim=embed_dim)  # TODO XXX
        self.aggregat = SlotAttention(
            num_iter=3,
            embed_dim=embed_dim,
            ffn_dim=embed_dim * 4,
            dropout=0.01,  # dropout in mlp: 0.01>0>0.1
            kv_dim=vfm_dim,
            trunc_bp=None,  # TODO XXX "bi-level", None
        )
        [
            nn.init.xavier_uniform_(_.weight)
            for _ in [
                self.aggregat.proj_q,
                self.aggregat.proj_k,
                self.aggregat.proj_v,
                self.aggregat.ffn[0],
                self.aggregat.ffn[3],
            ]
        ]
        nn.init.xavier_uniform_(self.aggregat.rnn.weight_ih)
        nn.init.orthogonal_(self.aggregat.rnn.weight_hh)
        """self.slot_attn = SlotAttentionEncoder(
            3,
            num_slots,
            vfm_dim,
            embed_dim,
            embed_dim * 4,
            4,
            "none",
            "shared_gaussian",
        )"""

        self.decode = AR9TransformerDecoder(
            resolut=[16, 16],
            vfm_dim=vfm_dim,
            posit_embed=nn.Identity(),
            project1=nn.Sequential(
                nn.Linear(vfm_dim, vfm_dim, bias=False), nn.LayerNorm(vfm_dim)
            ),
            project2=nn.Sequential(
                nn.Linear(embed_dim, vfm_dim, bias=False), nn.LayerNorm(vfm_dim)
            ),
            # backbone=TransformerDecoder(
            #     num_blocks=4,
            #     max_len=16 * 16,
            #     d_model=vfm_dim,
            #     num_heads=6,
            #     dropout=0,
            #     num_cross_heads=6,
            # ),
            backbone=TransformerDecoder(
                decoder_layer=TransformerDecoderLayer(
                    d_model=vfm_dim,
                    nhead=6,
                    dim_feedforward=vfm_dim * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                    bias=False,
                ),
                num_layers=4,
            ),
            readout=nn.Identity(),
            # Linear(in_features=vfm_dim, out_features=vfm_dim, bias=False),
            perm_t="random",
            perm_v="default",
        )
        # nn.init.xavier_uniform_(self.decode.project1[0].weight)
        # nn.init.xavier_uniform_(self.decode.project2[0].weight)
        # for _ in self.decode.backbone.modules():
        #     if isinstance(_, nn.Linear):
        #         nn.init.xavier_uniform_(_.weight)
        # gain = (3 * len(self.decode.backbone.layers)) ** -0.5
        # for _ in self.decode.backbone.layers:
        #     nn.init.xavier_uniform_(_.self_attn.out_proj.weight, gain=gain)
        #     nn.init.xavier_uniform_(_.multihead_attn.out_proj.weight, gain=gain)
        #     # nn.init.kaiming_uniform_(_.linear1.weight, nonlinearity="gelu")
        #     nn.init.xavier_uniform_(_.linear2.weight, gain=gain)
        # # all linear: xavier_uniform_
        # # all mha.proj_o gain
        # # all ffn: first linear use kaiming uniform, second use xavier with gain

        __class__.reset_parameters([self.encode_project, self.aggregat, self.decode])
        # __class__.reset_parameters([self.encode_project, self.decode])

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
                    nn.init.xavier_uniform_(m.weight_ih)
                    nn.init.orthogonal_(m.weight_hh)
                    if m.bias:
                        nn.init.zeros_(m.bias_ih)
                        nn.init.zeros_(m.bias_hh)

    def forward(self, image):
        """
        image: batch_size x img_channels x H x W
        """
        feature = self.encode_backbone(self.encode_backbone0(image))
        b, c, h, w = feature.shape
        with torch.no_grad():
            if self.encode_backbone2 is not None:
                feature2 = self.encode_backbone2(self.encode_backbone0(image))
            else:
                feature2 = feature.clone().detach()

        encode = rearrange(feature, "b c h w -> b (h w) c")
        encode2 = rearrange(feature2, "b c h w -> b (h w) c")
        encode = self.encode_project(encode)

        query = self.initializ(b)
        slotz, attent = self.aggregat(encode, query)
        recon, attent2 = self.decode(encode2, slotz)

        mse = F.mse_loss(recon, encode2)

        attent = rearrange(attent, "b n (h w) -> b n h w", h=h)
        attent2 = rearrange(attent2, "b n (h w) -> b n h w", h=h)

        return mse, attent, attent2, slotz, recon


class AR9TransformerDecoder(nn.Module):
    """SPOT's decoder. Auto-regressive Transformer decoder with 9 permutations."""

    def __init__(
        self,
        resolut,
        vfm_dim,
        posit_embed,  # 1d
        project1,
        project2,
        backbone,
        readout,
        perm_t="random",
        perm_v="default",  # ``all`` is not as beneficial as claimed
    ):
        super().__init__()

        self.perm_t = perm_t
        self.perm_v = perm_v
        perm = pt.as_tensor(__class__.generate_permutations(*resolut))
        self.register_buffer("perm", perm, persistent=False)
        if self.perm_t == "default":
            self.perm_v = "default"
        self.perm_idx = list(range(len(self.perm)))

        self.bos = nn.Parameter(pt.randn(len(self.perm), 1, 1, vfm_dim) * vfm_dim**-0.5)
        self.posit_embed = posit_embed
        self.register_buffer(
            "mask", pt.triu(pt.ones([np.prod(resolut)] * 2, dtype=pt.bool), 1)
        )
        self.project1 = project1
        self.project2 = project2

        assert isinstance(backbone, nn.TransformerDecoder)
        self.norm0 = backbone.layers[0].norm1  # very beneficial
        backbone.layers[0].norm1 = nn.Identity()  # very beneficial
        self.backbone = backbone
        self.readout = readout

        def attent_hook_forward_pre(module, args, kwargs):
            kwargs["need_weights"] = True  # obtain the attention weights

        def attent_hook_forward(module, args, output):
            self._attent = output[1]

        self.backbone.layers[-1].multihead_attn.register_forward_pre_hook(
            attent_hook_forward_pre, with_kwargs=True
        )
        self.backbone.layers[-1].multihead_attn.register_forward_hook(
            attent_hook_forward
        )

        """def hook_fn_forward_attn(module, input):
            self.dec_slots_attns.append(input[0])

        self.dec_slots_attns = []
        self.remove_handle = (
            self.backbone._modules["blocks"][-1]
            ._modules["encoder_decoder_attn"]
            ._modules["attn_dropout"]
            .register_forward_pre_hook(hook_fn_forward_attn)
        )"""

    @staticmethod
    def generate_permutations(h, w):
        perm_default = np.arange(h * w)
        perm_default_2d = perm_default.reshape(h, w)

        hs = tuple(range(h))
        ws = tuple(range(w))
        perm_topleft = [perm_default_2d[r, c] for c in ws for r in hs]
        perm_topright = [perm_default_2d[r, c] for c in ws[::-1] for r in hs]
        perm_righttop = [perm_default_2d[r, c] for r in hs for c in ws[::-1]]
        perm_bottomright = [perm_default_2d[r, c] for c in ws[::-1] for r in hs[::-1]]
        perm_rightbottom = [perm_default_2d[r, c] for r in hs[::-1] for c in ws[::-1]]
        perm_bottomleft = [perm_default_2d[r, c] for c in ws for r in hs[::-1]]
        perm_leftbottom = [perm_default_2d[r, c] for r in hs[::-1] for c in ws]

        perm_spiral = []
        A = np.rot90(perm_default_2d.copy(), k=1)
        while A.size:
            perm_spiral.append(A[0])  # take first row
            A = A[1:].T[::-1]  # cut off first row and rotate counterclockwise
        perm_spiral = np.concatenate(perm_spiral)[::-1]

        return (
            perm_default.tolist(),
            perm_topleft,
            perm_topright,
            perm_righttop,
            perm_bottomright,
            perm_rightbottom,
            perm_bottomleft,
            perm_leftbottom,
            perm_spiral.tolist(),
        )

    def forward(self, input, slotz):
        """
        - input: shape=(b,m=h*w,c)
        - slotz: shape=(b,n,c)
        """
        if self.training:
            if self.perm_t == "default":
                which = [0]
            elif self.perm_t == "random":
                which = [random.choice(self.perm_idx)]
            elif self.perm_t == "all":
                which = self.perm_idx
            else:
                raise ValueError
        else:
            if self.perm_v == "default":
                which = [0]
            elif self.perm_v == "random":
                which = [random.choice(self.perm_idx)]
            elif self.perm_v == "all":
                which = self.perm_idx
            else:
                raise ValueError

        output = []
        attent = []

        for perm_idx in which:
            perm_i = self.perm[perm_idx]
            inv_perm_i = perm_i.argsort()

            bos_i = self.bos[perm_idx].expand(input.shape[0], -1, -1)
            query_i = pt.cat([bos_i, input[:, perm_i, :][:, :-1, :]], dim=1)
            query_i = self.project1(query_i)
            memory_i = self.project2(slotz)
            output_i = self.backbone(query_i, memory_i, tgt_mask=self.mask)

            # attent_i = self.dec_slots_attns[0]  # (b,h,m,n)
            # self.dec_slots_attns = []
            # attent_i = attent_i.mean(1)  # TODO XXX sum  # (b,m,n)
            attent_i = self._attent

            output_i = output_i[:, inv_perm_i, :]
            attent_i = attent_i[:, inv_perm_i, :]

            attent.append(attent_i)
            output.append(output_i)

        output = pt.stack(output).mean(0)  # (b,m,c)
        attent = pt.stack(attent).mean(0).permute(0, 2, 1)  # (b,n,m)
        return output, attent


'''
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads, dropout=0.0, gain=1.0):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads

        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)

    def forward(self, q, k, v, attn_mask=None):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """
        B, T, _ = q.shape
        _, S, _ = k.shape

        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)

        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.output_dropout(output)
        return output


class TransformerDecoderBlock(nn.Module):

    def __init__(
        self,
        max_len,
        d_model,
        num_heads,
        dropout=0.0,
        gain=1.0,
        is_first=False,
        num_cross_heads=None,
    ):
        super().__init__()

        self.is_first = is_first

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)

        mask = torch.triu(torch.ones((max_len, max_len), dtype=torch.bool), diagonal=1)
        self.self_attn_mask = nn.Parameter(mask, requires_grad=False)

        self.encoder_decoder_attn_layer_norm = nn.LayerNorm(d_model)

        if num_cross_heads is None:
            num_cross_heads = num_heads
        self.encoder_decoder_attn = MultiHeadAttention(
            d_model, num_cross_heads, dropout, gain
        )

        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, 4 * d_model, weight_init="kaiming"),
            nn.ReLU(),
            linear(4 * d_model, d_model, gain=gain),
            nn.Dropout(dropout),
        )

    def forward(self, input, encoder_output, causal_mask=True):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        T = input.shape[1]

        self_attn_mask = self.self_attn_mask[:T, :T] if causal_mask else None
        if self.is_first:
            input = self.self_attn_layer_norm(input)
            x = self.self_attn(input, input, input, self_attn_mask)
            input = input + x
        else:
            x = self.self_attn_layer_norm(input)
            x = self.self_attn(x, x, x, self_attn_mask)
            input = input + x

        x = self.encoder_decoder_attn_layer_norm(input)
        x = self.encoder_decoder_attn(x, encoder_output, encoder_output)
        input = input + x

        x = self.ffn_layer_norm(input)
        x = self.ffn(x)
        return input + x


class TransformerDecoder(nn.Module):

    def __init__(
        self, num_blocks, max_len, d_model, num_heads, dropout=0.0, num_cross_heads=None
    ):
        super().__init__()

        if num_blocks > 0:
            gain = (3 * num_blocks) ** (-0.5)
            self.blocks = nn.ModuleList(
                [
                    TransformerDecoderBlock(
                        max_len, d_model, num_heads, dropout, gain, is_first=True
                    )
                ]
                + [
                    TransformerDecoderBlock(
                        max_len,
                        d_model,
                        num_heads,
                        dropout,
                        gain,
                        is_first=False,
                        num_cross_heads=num_cross_heads,
                    )
                    for _ in range(num_blocks - 1)
                ]
            )
        else:
            self.blocks = nn.ModuleList()

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input, encoder_output, causal_mask=True):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """
        for block in self.blocks:
            input = block(input, encoder_output, causal_mask)

        return self.layer_norm(input)
'''


def linear(in_features, out_features, bias=True, weight_init="xavier", gain=1.0):
    m = nn.Linear(in_features, out_features, bias)
    if weight_init == "kaiming":
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    if bias:
        nn.init.zeros_(m.bias)
    return m


def gru_cell(input_size, hidden_size, bias=True):
    m = nn.GRUCell(input_size, hidden_size, bias)
    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)
    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)
    return m
