from copy import deepcopy
from operator import attrgetter
import math
import re

from diffusers.models import AutoencoderKL, AutoencoderTiny
from einops import rearrange
import timm
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf
import numpy as np


####


class ModelWrap(nn.Module):

    def __init__(self, m: nn.Module, imap, omap):
        """
        - imap: dict or list.
            If keys in batch mismatches with keys in model.forward, use dict, ie, {key_in_batch: key_in_forward};
            If not, use list.
        - omap: list
        """
        super().__init__()
        assert isinstance(imap, (dict, list, tuple))
        assert isinstance(omap, (list, tuple))
        self.m = m
        self.imap = imap if isinstance(imap, dict) else {_: _ for _ in imap}
        self.omap = omap

    # @pt.compile
    def forward(self, input) -> dict:
        # print(self.imap.keys()) # dict_keys(['input'])
        # print(self.imap.values())
        input2 = {k: input[v] for k, v in self.imap.items()}
        # print(input2.keys(), input2['input'].shape) # dict_keys(['input']) torch.Size([1, 3, 256, 256])
        output = self.m(**input2)
        # output = self.m(input)
        if not isinstance(output, (list, tuple)):
            output = [output]
        # print(self.omap)
        # print(len(output))
        assert len(self.omap) == len(output)
        output2 = dict(zip(self.omap, output))
        return output2

    def load(self, ckpt_file: str, ckpt_map: list, verbose=True):
        state_dict = pt.load(ckpt_file, map_location="cpu", weights_only=True)
        if ckpt_map is None:
            if verbose:
                print("fully")
            self.load_state_dict(state_dict)  # TODO XXX , False
        elif isinstance(ckpt_map, (list, tuple)):
            for dst, src in ckpt_map:
                dkeys = [_ for _ in self.state_dict() if _.startswith(dst)]
                skeys = [_ for _ in state_dict if _.startswith(src)]
                assert len(dkeys) == len(skeys)  # > 0
                if len(dkeys) == 0:
                    print(
                        f"[{__class__.__name__}.load WARNING] ``{dst}, {src}`` has no matched keys !!!"
                    )
                for dk, sk in zip(dkeys, skeys):
                    if verbose:
                        print(dk, sk)
                    self.state_dict()[dk].data[...] = state_dict[sk]
        else:
            raise "ValueError"
        if verbose:
            print(f"checkpoint ``{ckpt_file}`` loaded")

    def save(self, save_file, weights_only=True, key=r".*"):
        if weights_only:
            save_obj = self.state_dict()
            save_obj = {k: v for k, v in save_obj.items() if re.match(key, k)}
        else:
            save_obj = self
        pt.save(save_obj, save_file)

    def freez(self, freez: list, verbose=True):
        for n, p in self.named_parameters():
            for f in freez:
                if bool(re.match(f, n)):
                    p.requires_grad = False
        if verbose:
            [print(k, v.requires_grad) for k, v in self.named_parameters()]

    def group_params(self, coarse=r"^.*", fine=dict()):
        """Group model parameters by coarse and fine filters.

        - coarse: coarse filter; regex string
        - fine: fine filter for grouping and adding extras; {regex1: dict(lr_mult=0.5, wd_mult=0),..}
        """
        # coarse filtering
        named_params = dict(self.named_parameters())
        named_params = {
            k: v for k, v in named_params.items() if bool(re.match(coarse, k))
        }
        if not fine:
            params = []
            for k, v in named_params.items():
                if v.requires_grad:
                    print(f"{k} - to train, require grad")
                    params.append(v)
                else:
                    print(f"{k} - skipped, not require grad")
            return params

        # fine filtering
        param_groups = {k: dict(params=[]) for k in fine}  # TODO lr
        names = list(named_params.keys())
        for n, p in named_params.items():
            for g, (k, v) in enumerate(fine.items()):
                assert isinstance(v, dict)
                if bool(re.match(k, n)):
                    cursor = names.pop(0)
                    assert cursor == n  # ensure no missing or overlap
                    if p.requires_grad:
                        print(f"{n} - #{g}, {v}")
                        param_groups[k]["params"].append(p)
                        param_groups[k].update(v)
                    else:
                        print(f"{n} - #{g}, skipped, not require grad")

        param_groups = {k: v for k, v in param_groups.items() if len(v["params"])}
        return list(param_groups.values())


class Sequential(nn.Sequential):
    """"""

    def __init__(self, modules: list):
        super().__init__(*modules)

    def forward(self, input):
        for module in self:
            if isinstance(input, (list, tuple)):  # TODO control in init
                input = module(*input)
            else:
                input = module(input)
        return input


ModuleList = nn.ModuleList


####


Embedding = nn.Embedding


Conv2d = nn.Conv2d


PixelShuffle = nn.PixelShuffle


ConvTranspose2d = nn.ConvTranspose2d


AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d


Identity = nn.Identity


ReLU = nn.ReLU


GELU = nn.GELU


SiLU = nn.SiLU


Mish = nn.Mish


class Interpolate(nn.Module):

    def __init__(self, size=None, scale_factor=None, interp="bilinear"):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.interp = interp

    def forward(self, input):
        return ptnf.interpolate(input, self.size, self.scale_factor, self.interp)


class Conv2dPixelShuffle(nn.Sequential):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        upscale=2,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels * upscale**2,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        shuff = nn.PixelShuffle(upscale)
        super().__init__(conv, shuff)


Dropout = nn.Dropout


Linear = nn.Linear


GroupNorm = nn.GroupNorm


LayerNorm = nn.LayerNorm


####


MultiheadAttention = nn.MultiheadAttention


TransformerEncoderLayer = nn.TransformerEncoderLayer


TransformerDecoderLayer = nn.TransformerDecoderLayer


TransformerEncoder = nn.TransformerEncoder


TransformerDecoder = nn.TransformerDecoder


###


class CNN(nn.Sequential):
    """hyperparam setting of ConvTranspose2d:
    https://blog.csdn.net/pl3329750233/article/details/130283512.
    """

    conv_types = {
        0: nn.Conv2d,
        1: lambda *a, **k: nn.ConvTranspose2d(*a, **k, output_padding=1),
        2: lambda *a, **k: Conv2dPixelShuffle(*a, **k, upscale=2),
    }

    def __init__(self, in_dim, dims, kernels, strides, ctypes=0, gn=0, act="SiLU"):
        """
        - ctypes: 0 for normal conv2d, 1 for convtransposed, 2 for convpixelshuffle
        - gn: 0 for no groupnorm, >0 for groupnorm(num_groups=g)
        """
        if isinstance(ctypes, int):
            ctypes = [ctypes] * len(dims)
        assert len(dims) == len(kernels) == len(strides) == len(ctypes)
        num = len(dims)

        layers = []
        ci = in_dim

        for i, (t, c, k, s) in enumerate(zip(ctypes, dims, kernels, strides)):
            p = k // 2 if k % 2 != 0 else 0  # XXX for k=s=4, requires isize%k==0

            if i + 1 < num:
                block = [
                    __class__.conv_types[t](ci, c, k, stride=s, padding=p),
                    nn.GroupNorm(gn, c) if gn else None,
                    nn.__dict__[act](inplace=True),  # SiLU>Mish>ReLU>Hardswish
                ]
            else:
                block = [
                    __class__.conv_types[t](ci, c, k, stride=s, padding=p),
                ]

            layers.extend([_ for _ in block if _])
            ci = c

        super().__init__(*layers)


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


class ResNet(nn.Sequential):
    """https://huggingface.co/timm/resnet18.fb_swsl_ig1b_ft_in1k"""

    def __init__(
        self,
        model_name="resnet18.fb_swsl_ig1b_ft_in1k",
        in_dim=3,
        k0=7,
        strides=[2] * 5,
        dilats=[1] * 5,
        gn=0,
        learn_changed_only=False,
    ):
        assert in_dim == 3
        assert 2 <= len(strides) <= 5
        assert all(_ in [1, 2] for _ in strides[1:]) and strides[0] in [1, 2, 4]
        assert len(strides) <= len(dilats)
        dilats = dilats[: len(strides)]
        assert dilats[:2] == [1, 1]

        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        resnet = deepcopy(model)
        if learn_changed_only:
            for p in resnet.parameters():
                p.requires_grad = False

        if gn > 0:
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    group_norm = nn.GroupNorm(gn, module.num_features)
                    group_norm.weight.data[...] = module.weight
                    group_norm.bias.data[...] = module.bias
                    if "." in name:
                        parent = attrgetter(".".join(name.split(".")[:-1]))(resnet)
                    else:
                        parent = resnet
                    setattr(parent, name.split(".")[-1], group_norm)
                    for p in group_norm.parameters():
                        p.requires_grad = True

        if in_dim != 3 or k0 != 7:  # for dvae, k4d4 > k7d2d2
            assert k0 < 7
            conv1 = nn.Conv2d(in_dim, 64, k0, 2, (k0 - 1) // 2, bias=False)
            conv1.weight.data[...] = ptnf.interpolate(resnet.conv1.weight, [k0, k0])
            resnet.conv1 = conv1
            for p in conv1.parameters():
                p.requires_grad = True

        if strides[0] != 2:
            resnet.conv1.stride = strides[0]
            for p in resnet.conv1.parameters():
                p.requires_grad = True
        layers = [resnet.conv1, resnet.bn1, resnet.act1]

        if strides[1] == 2:
            layers.append(resnet.maxpool)
        layers.append(resnet.layer1)

        for i, (s, d) in enumerate(zip(strides, dilats)):
            if i < 2:  # skip conv1 and maxpool
                continue
            layer = getattr(resnet, f"layer{i}")

            if s == 1:
                layer[0].conv1.stride = 1
                for p in layer[0].conv1.parameters():
                    p.requires_grad = True
                layer[0].downsample[0].stride = 1
                for p in layer[0].downsample[0].parameters():
                    p.requires_grad = True

            if d != 1:
                layer[0].conv1.dilation = (d,) * 2
                layer[0].conv1.padding = (layer[0].conv1.padding[0] + d // 2,) * 2
                for p in layer[0].conv1.parameters():
                    p.requires_grad = True

            layers.append(layer)

        super().__init__(*layers)


class ResNetSlice(nn.Sequential):

    def __init__(
        self,
        start,
        end=None,
        model_name="resnet18.fb_swsl_ig1b_ft_in1k",
        in_dim=3,
        k0=7,
        strides=[2] * 5,
        dilats=[1] * 5,
        gn=0,
        learn_changed_only=False,
    ):
        resnet = ResNet(model_name, in_dim, k0, strides, dilats, gn, learn_changed_only)
        layers = [resnet[_] for _ in range(start, end)]
        super().__init__(*layers)

# import timm
# import torch.nn as nn
# from einops import rearrange

import timm
import torch.nn as nn
from einops import rearrange


import timm
import torch.nn as nn
from einops import rearrange


class DINO2ViT_P16(nn.Module):
    """
    封装 timm 模型 ``vit_small_patch16_224.dino``

    默认把 Patch‑Token 重排成 (B, C, 14, 14) 特征图。
    设 `rearrange_tokens=False` 则返回 (B, N+prefix, C) token 序列。

    Parameters
    ----------
    model_name : str
        timm 权重名称，默认 "vit_small_patch16_224.dino".
    in_size : int
        输入分辨率，默认 224.
    rearrange_tokens : bool
        是否把 token 重排回 2‑D.
    norm_out : bool
        是否保留最后 LayerNorm.
    freeze_backbone : bool
        若 True，则冻结所有参数。
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224.dino",
        in_size: int = 224,
        rearrange: bool = True,
        norm_out: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # ------------------------------------------------------------------ #
        # 1. 构建 timm backbone
        # ------------------------------------------------------------------ #
        model = timm.create_model(model_name, pretrained=True, img_size=in_size)

        # ------------------------------------------------------------------ #
        # 2. 基本信息校验
        # ------------------------------------------------------------------ #
        self.in_size = model.patch_embed.img_size[0]
        assert self.in_size == in_size, f"expected {in_size}, got {self.in_size}"

        self.patch_size = model.patch_embed.patch_size[0]
        assert self.patch_size == 16, f"patch size must be 16, got {self.patch_size}"

        self.out_size = in_size // self.patch_size  # =14
        self.rearrange_tokens = rearrange

        # ------------------------------------------------------------------ #
        # 3. 逐层复制（Module/Parameter 级）
        # ------------------------------------------------------------------ #
        self.cls_token   = model.cls_token
        if hasattr(model, "reg_token"):
            self.reg_token = model.reg_token
        if hasattr(model, "dist_token"):
            self.dist_token = model.dist_token

        self.pos_embed   = model.pos_embed
        self.patch_embed = model.patch_embed
        self.pos_drop    = model.pos_drop
        self.patch_drop  = model.patch_drop
        self.norm_pre    = model.norm_pre
        self.blocks      = model.blocks
        self.norm        = model.norm if norm_out else nn.Identity()

        # ------------------------------------------------------------------ #
        # 4. 让 _pos_embed 能正常工作
        # ------------------------------------------------------------------ #
        self.dynamic_img_size  = getattr(model, "dynamic_img_size", False)
        self.disable_pos_embed = getattr(model, "disable_pos_embed", False)

        # 4.1   interpolate_pos_embed — 部分版本可能缺失
        if hasattr(model, "interpolate_pos_embed"):
            # 绑定到当前实例，保证 self 调用
            self.interpolate_pos_embed = model.interpolate_pos_embed.__get__(self, self.__class__)
        else:
            # 兜底：identity 函数，避免 dynamic_img_size=True 时报错
            def _identity_interpolate(pos_embed, *_, **__):
                return pos_embed
            self.interpolate_pos_embed = _identity_interpolate

        # 4.2   _pos_embed & forward_features 直接借用 timm 实现
        self._pos_embed       = model._pos_embed.__get__(self, self.__class__)
        self.forward_features = model.forward_features.__get__(self, self.__class__)
        self.num_prefix_tokens = getattr(model, "num_prefix_tokens", 1)  # 通常=1

        # ------------------------------------------------------------------ #
        # 5. (可选) 冻结骨干
        # ------------------------------------------------------------------ #
        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad_(False)

    # ---------------------------------------------------------------------- #
    # forward
    # ---------------------------------------------------------------------- #
    def forward(self, x):
        """
        x: (B, 3, 224, 224)
        Returns
        -------
        Tensor
            (B, C, 14, 14)  or  (B, N+prefix, C)
        """
        feat = self.forward_features(x)  # (B, N+prefix, C)

        if self.rearrange_tokens:
            feat = feat[:, self.num_prefix_tokens :, :]        # drop CLS / etc.
            feat = rearrange(feat, "b (h w) c -> b c h w", h=self.out_size)
        return feat



class DINO2ViT(nn.Module):
    """
    https://huggingface.co/collections/timm/timm-backbones-6568c5b32f335c33707407f8
    """

    def __init__(
        self,
        model_name="vit_small_patch14_reg4_dinov2.lvd142m",
        in_size=518,
        rearrange=True,
        norm_out=True,
    ):
        super().__init__()
        # dict(
        #     patch_size=14,
        #     embed_dim=384,
        #     depth=12,
        #     num_heads=6,
        #     init_values=1e-05,
        #     reg_tokens=4,
        #     no_embed_class=True,
        #     pretrained_cfg="lvd142m",
        #     pretrained_cfg_overlay=None,
        #     cache_dir=None,
        # )
        model = timm.create_model(model_name, pretrained=True, img_size=in_size)
        self.in_size = model.patch_embed.img_size[0]
        assert self.in_size == in_size
        self.patch_size = model.patch_embed.patch_size[0]
        
        # print("patch_size:", self.patch_size)
        assert self.patch_size == 14

        self.cls_token = model.cls_token
        self.reg_token = model.reg_token
        self.pos_embed = model.pos_embed
        self.patch_embed = model.patch_embed
        self.pos_drop = model.pos_drop
        self.patch_drop = model.patch_drop
        self.norm_pre = model.norm_pre
        self.blocks = model.blocks
        self.norm = model.norm if norm_out else nn.Identity()

        for k, v in model.__dict__.items():
            if any(
                [
                    k.startswith("__") and k.endswith("__"),
                    k.startswith("_"),
                    isinstance(v, nn.Module),
                    isinstance(v, nn.Parameter),
                    hasattr(self, k),
                ]
            ):
                print(f"[{__class__.__name__}] skip {k}")
                continue
            else:
                print(f"[{__class__.__name__}] copy {k}")
                setattr(self, k, v)
        assert hasattr(self, "num_prefix_tokens")

        __class__._pos_embed = model.__class__._pos_embed
        __class__.forward_features = model.__class__.forward_features

        self.rearrange = rearrange
        self.out_size = in_size // self.patch_size
        assert self.out_size <= 518 // 14

    def forward(self, input):
        """
        input: shape=(b,c,h,w), float
        """
        # with pt.inference_mode(True):  # infer+compile: errors
        # print(input.shape)
        feature = self.forward_features(input)
        if self.rearrange:
            feature = feature[:, self.num_prefix_tokens :, :]  # remove class token
            feature = rearrange(feature, "b (h w) c -> b c h w", h=self.out_size)
        return feature  # .clone()


class SAM2Hiera(nn.Module):
    """Heira backbone + FPN neck in SAM2
    https://github.com/facebookresearch/sam2
    """

    ARCHS = {
        "sam2.1_hiera_tiny": dict(
            cfg_name="configs/sam2.1/sam2.1_hiera_t.yaml",
            ckpt_url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        ),
        "sam2.1_hiera_small": dict(
            cfg_name="configs/sam2.1/sam2.1_hiera_s.yaml",
            ckpt_url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        ),
        "sam2.1_hiera_base_plus": dict(
            cfg_name="configs/sam2.1/sam2.1_hiera_b+.yaml",
            ckpt_url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        ),
        "sam2.1_hiera_large": dict(
            cfg_name="configs/sam2.1/sam2.1_hiera_l.yaml",
            ckpt_url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        ),
    }

    def __init__(self, model_name="sam2.1_hiera_tiny", which=2, nb=0):
        super().__init__()
        from pathlib import Path
        from sam2.build_sam import build_sam2
        import sam2

        cfg_file = Path(sam2.__file__).parent / __class__.ARCHS[model_name]["cfg_name"]
        ckpt_file = Path(pt.hub.get_dir()) / f"{model_name}.pt"
        if not ckpt_file.exists():
            pt.hub.download_url_to_file(
                __class__.ARCHS[model_name]["ckpt_url"], ckpt_file
            )
            print(f"``{model_name}`` is downloaded to local ``{ckpt_file}``")

        sam2 = build_sam2(
            config_file="/" + str(cfg_file), ckpt_path=ckpt_file, device="cpu"
        )
        self.hiera = sam2.image_encoder.trunk
        if nb > 0:  # removing last pool for dx16 causes grid effect
            self.hiera.blocks = self.hiera.blocks[:nb]  # use nb=10 for dx16
        if not nb > 0:
            self.fpn = sam2.image_encoder.neck
        self.which = which  # pick which scale: [dx=4, 8, 16, 32]

    @pt._dynamo.disable()  # compile causes nan grad, in whatever mode
    def forward(self, input):
        with pt.inference_mode():
            if hasattr(self, "fpn"):
                features, posits = self.fpn(self.hiera(input))
                assert len(features) == 4
            else:
                features = self.hiera(input)
        return features[self.which].clone()  # clone for backward  # dx=16


class EncoderVAESD(nn.Sequential):
    """https://huggingface.co/stabilityai/sd-vae-ft-mse"""

    def __init__(self, pre=1, se=[0, 4], mid=1, post=1):
        vaeenc = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", torch_dtype=pt.float16
        ).encoder.float()
        layers = []
        if pre:
            layers.append(vaeenc.conv_in)
        layers.extend(vaeenc.down_blocks[se[0] : se[1]])
        if mid:
            layers.append(vaeenc.mid_block)
        if post:
            layers.extend([vaeenc.conv_norm_out, vaeenc.conv_act, vaeenc.conv_out])
        super().__init__(*layers)


class DecoderVAESD(nn.Sequential):
    """https://huggingface.co/stabilityai/sd-vae-ft-mse"""

    def __init__(self, pre=1, mid=1, se=[0, 4], post=1):
        vaedec = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", torch_dtype=pt.float16
        ).decoder.float()
        layers = []
        if pre:
            layers.append(vaedec.conv_in)
        if mid:
            layers.append(vaedec.mid_block)
        layers.extend(vaedec.up_blocks[se[0] : se[1]])
        if post:
            layers.extend([vaedec.conv_norm_out, vaedec.conv_act, vaedec.conv_out])
        super().__init__(*layers)


class EncoderTAESD(nn.Sequential):
    """
    https://huggingface.co/docs/diffusers/en/api/models/autoencoder_tiny
    https://huggingface.co/madebyollin/taesd
    """

    def __init__(self, se=[0, 15], gn=0):
        self.se = se
        vaeenc = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", torch_dtype=pt.float16
        ).encoder.float()
        assert len(vaeenc.layers) == 15
        if gn > 0:
            self._insert_groupnorm(vaeenc.layers, "14", gn)
        layers = vaeenc.layers[se[0] : se[1]]
        super().__init__(*layers)

    @staticmethod
    def _insert_groupnorm(layers, skip_last_conv, gn=1):
        """does not change ``len(layers)``'s value"""
        for n, m in layers.named_modules():
            if isinstance(m, nn.Conv2d):
                if n == skip_last_conv:
                    continue
                parts = n.split(".")
                p = ".".join(parts[:-1])
                c = parts[-1]
                parent = attrgetter(p)(layers) if p != "" else layers
                child = nn.Sequential(m, nn.GroupNorm(gn, m.out_channels))
                setattr(parent, c, child)
        assert n == skip_last_conv

    def forward(self, input):
        if self.se[0] == 0:
            input = input.add(1).div(2)  # original implementation
        return super().forward(input)


class DecoderTAESD(nn.Sequential):
    """
    https://huggingface.co/docs/diffusers/en/api/models/autoencoder_tiny
    https://huggingface.co/madebyollin/taesd
    """

    def __init__(self, se=[0, 19], gn=0):
        self.se = se
        vaedec = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", torch_dtype=pt.float16
        ).decoder.float()
        assert len(vaedec.layers) == 19
        if gn > 0:
            EncoderTAESD._insert_groupnorm(vaedec.layers, "18", gn)
        layers = vaedec.layers[se[0] : se[1]]
        super().__init__(*layers)

    def forward(self, input):
        output = super().forward(input)
        if self.se[1] == 19:
            output = output.mul(2).sub(1)  # original implementation
        return output
