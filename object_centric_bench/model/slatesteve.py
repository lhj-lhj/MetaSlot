import math

from einops import rearrange
import torch as pt
import torch.nn as nn


class SLATE(nn.Module):

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        mediat,  # dVAE originally
        decode,
    ):
        super().__init__()
        self.encode_backbone = encode_backbone
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.aggregat = aggregat
        self.mediat = mediat
        self.decode = decode
        self.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat]
        )

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
        - input: image in shape (b,c,h,w)
        - condit: condition in shape (b,n,c)
        """
        feature = self.encode_backbone(input)  # (b,c,h,w)
        b, c, h, w = feature.shape
        encode = feature.permute(0, 2, 3, 1)  # (b,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b,h*w,c)
        encode = self.encode_project(encode)

        query = self.initializ(b if condit is None else condit)  # (b,n,c)
        slotz, attent = self.aggregat(encode, query)
        attent = rearrange(attent, "b n (h w) -> b n h w", h=h)

        with pt.inference_mode(True):
            encode1, zidx, quant, decode1 = self.mediat(input)
        b, c, h, w = quant.shape
        zidx = zidx.clone()
        quant = quant.clone()  # (b,c,h,w)

        clue = rearrange(quant, "b c h w -> b (h w) c")
        recon = self.decode(clue, slotz)  # (b,h*w,c)
        recon = rearrange(recon, "b (h w) c -> b c h w", h=h)

        return slotz, attent, zidx, recon


class STEVE(SLATE):

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        mediat,  #  dVAE originally
        decode,
        transit,
    ):
        super().__init__(
            encode_backbone,
            encode_posit_embed,
            encode_project,
            initializ,
            aggregat,
            mediat,
            decode,
        )
        self.transit = transit
        self.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat, self.transit]
        )

    def forward(self, input, condit=None):
        """
        - input: video in shape (b,t,c,h,w)
        - condit: condition in shape (b,t,n,c)
        """
        b, t, c, h, w = input.shape
        input = input.flatten(0, 1)  # (b*t,c,h,w)

        feature = self.encode_backbone(input)  # (b*t,c,h,w)
        bt, c, h, w = feature.shape
        encode = feature.permute(0, 2, 3, 1)  # (b*t,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b*t,h*w,c)
        encode = self.encode_project(encode)

        encode = rearrange(encode, "(b t) hw c -> b t hw c", b=b)

        query = self.initializ(b if condit is None else condit[:, 0, :, :])  # (b,n,c)
        slotz = []
        attent = []
        for i in range(t):
            slotz_i, attent_i = self.aggregat(encode[:, i, :, :], query)
            query = self.transit(slotz_i)
            slotz.append(slotz_i)  # [(b,n,c),..]
            attent.append(attent_i)  # [(b,n,h*w),..]
        slotz = pt.stack(slotz, 1)  # (b,t,n,c)
        attent = pt.stack(attent, 1)  # (b,t,n,h*w)
        attent = rearrange(attent, "b t n (h w) -> b t n h w", h=h)

        with pt.inference_mode(True):
            encode1, zidx, quant, decode1 = self.mediat(input)
        bt, c, h, w = quant.shape
        zidx = zidx.clone().unflatten(0, [b, t])  # (b,t,h,w)
        quant = quant.clone()  # (b*t,c,h,w)

        clue = rearrange(quant, "bt c h w -> bt (h w) c")
        recon = self.decode(clue, slotz.flatten(0, 1))  # (b*t,h*w,c)
        recon = rearrange(recon, "(b t) (h w) c -> b t c h w", b=b, h=h)

        return slotz, attent, zidx, recon


class ARTransformerDecoder(nn.Module):
    """SLATE's decoder."""

    def __init__(self, resolut, embed_dim, posit_embed, backbone, readout):
        super().__init__()
        self.bos = nn.Parameter(pt.randn(1, 1, embed_dim) * embed_dim**-0.5)
        self.posit_embed = posit_embed  # 1d
        self.register_buffer(
            "mask", pt.triu(pt.ones([math.prod(resolut)] * 2, dtype=pt.bool), 1)
        )
        assert isinstance(backbone, nn.TransformerDecoder)
        self.norm0 = backbone.layers[0].norm1  # very beneficial
        backbone.layers[0].norm1 = nn.Identity()  # very beneficial
        self.backbone = backbone
        self.readout = readout

    def forward(self, input, slotz):
        """
        input: target to be destructed, shape=(b,m,c)
        slotz: slots, shape=(b,n,c)
        """
        b, m, c = input.shape
        query = pt.cat([self.bos.expand(b, -1, -1), input[:, :-1, :]], 1)
        query = self.posit_embed(query)
        autoreg = self.backbone(self.norm0(query), memory=slotz, tgt_mask=self.mask)
        recon = self.readout(autoreg)  # (b,m,c)
        return recon
