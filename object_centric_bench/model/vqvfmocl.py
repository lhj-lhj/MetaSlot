from einops import rearrange, repeat
import torch as pt
import torch.nn as nn


class VqVfmOcl(nn.Module):
    """
    Vector-Quantized Vision Foundation Model for Object Centric Learning.
    """

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        mediat,
        decode,
    ):
        super().__init__()
        self.encode_backbone = encode_backbone
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.aggregat = aggregat
        self.mediat = mediat  # pretrain: reconstruct vfm feature < original image
        self.decode = decode
        __class__.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat]
        )

    @staticmethod
    def reset_parameters(modules):
        for module in modules:
            if module is None:
                continue
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
        - input: image, shape=(b,c,h,w), float32
        - condit: condition, shape=(b,n,c), float32
        """
        feature = self.encode_backbone(input).detach()  # (b,c,h,w)
        b, c, h, w = feature.shape

        encode1, zidx, quant, residual, decode1 = self.mediat(feature)  # (b,c,h,w)
        if self.aggregat is None:  # pretrain
            return feature, encode1, zidx, quant, residual, decode1
        quant = quant.detach()

        encode = feature.permute(0, 2, 3, 1)  # (b,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b,h*w,c)
        encode = self.encode_project(encode)

        query = self.initializ(b if condit is None else condit)  # (b,n,c)
        slotz, attent = self.aggregat(encode, query)
        attent = rearrange(attent, "b n (h w) -> b n h w", h=h)

        dkwds = locals()
        dkwds.pop("self")
        decode = self.forward_decode(**dkwds)  # type: tuple
        return feature, zidx, quant, slotz, attent, *decode

    def forward_decode(self, **kwds) -> tuple:
        raise NotImplementedError


class VqVfmOclT(VqVfmOcl):
    """
    Vector-Quantized Vision Foundation Model for Object Centric Learning Temporal.
    """

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        transit,
        mediat,
        decode,
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
        __class__.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat, self.transit]
        )

    def forward(self, input, condit=None):
        """
        - input: video, shape=(b,t,c,h,w), float32
        - condit: condition, shape=(b,t,n,c), float32
        """
        b, t, c, h, w = input.shape
        input = input.flatten(0, 1)  # (b*t,c,h,w)

        feature = self.encode_backbone(input).detach()  # (b*t,c,h,w)
        bt, c, h, w = feature.shape

        encode1, zidx, quant, residual, decode1 = self.mediat(feature)  # (b*t,c,h,w)
        feature, encode1, zidx, quant, residual, decode1 = [
            None if _ is None else _.unflatten(0, [b, t])  # (b,t,c,h,w)
            for _ in [feature, encode1, zidx, quant, residual, decode1]
        ]
        if self.aggregat is None:  # pretrain
            return feature, encode1, zidx, quant, residual, decode1
        quant = quant.detach()

        encode = feature.flatten(0, 1).permute(0, 2, 3, 1)  # (b*t,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b*t,h*w,c)
        encode = self.encode_project(encode)
        encode = encode.unflatten(0, [b, t])

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

        dkwds = locals()
        dkwds.pop("self")
        decode = self.forward_decode(**dkwds)  # type: tuple
        return feature, zidx, quant, slotz, attent, *decode


class VVOTfd(VqVfmOcl):
    """
    VQ-VFM-OCL with TransFormer Decoder.
    """

    def forward_decode(self, quant, slotz, **kwds):
        b, c, h, w = quant.shape
        clue = rearrange(quant, "b c h w -> b (h w) c")
        recon = self.decode(clue, slotz)
        recon = rearrange(recon, "b (h w) c -> b c h w", h=h)
        return (recon,)
        # forward output: feature, zidx, quant, slotz, attent, recon


class VVOTfdT(VqVfmOclT):
    """
    Temporal VQ-VFM-OCL with TransFormer Decoder.
    """

    def forward_decode(self, quant, slotz, **kwds):
        b, t, c, h, w = quant.shape
        clue = rearrange(quant, "b t c h w -> (b t) (h w) c")
        slotz = rearrange(slotz, "b t n c -> (b t) n c")
        recon = self.decode(clue, slotz)
        recon = rearrange(recon, "(b t) (h w) c -> b t c h w", t=t, h=h)
        return (recon,)
        # forward output: feature, zidx, quant, slotz, attent, recon


class VVOMlp(VqVfmOcl):
    """
    VQ-VFM-OCL with MLP Decoder.
    """

    def forward_decode(self, quant, slotz, **kwds):
        b, c, h, w = quant.shape
        clue = [h, w]
        recon, attent2 = self.decode(clue, slotz)
        recon = rearrange(recon, "b (h w) c -> b c h w", h=h)
        attent2 = rearrange(attent2, "b n (h w) -> b n h w", h=h)
        # interleave feature and quant as recon target (in whichever dimensions): bad
        return recon, attent2
        # forward ouput: feature, zidx, quant, slotz, attent, recon, attent2


class VVOMlpT(VqVfmOclT):
    """
    Temporal VQ-VFM-OCL with MLP Decoder.
    """

    def forward_decode(self, quant, slotz, **kwds):
        b, t, c, h, w = quant.shape
        clue = [h, w]
        slotz = slotz.flatten(0, 1)
        recon, attent2 = self.decode(clue, slotz)
        recon = rearrange(recon, "(b t) (h w) c -> b t c h w", b=b, h=h)
        attent2 = rearrange(attent2, "(b t) n (h w) -> b t n h w", b=b, h=h)
        return recon, attent2
        # forward output: feature, zidx, quant, slotz, attent, recon, attent2


class VVODfz(VqVfmOcl):
    """
    VQ-VFM-OCL with Diffusion Decoder.
    """

    def forward_decode(self, quant, slotz, **kwds):
        clue = quant
        recon, noise = self.decode(clue, slotz)
        return recon, noise
        # forward output: feature, zidx, quant, slotz, attent, recon, noise


class VVODfzT(VqVfmOclT):
    """
    Temporal VQ-VFM-OCL with Diffusion Decoder.
    """

    def forward_decode(self, quant, slotz, **kwds):
        b, t, c, h, w = quant.shape
        clue = quant.flatten(0, 1)
        slotz = slotz.flatten(0, 1)
        recon, noise = self.decode(clue, slotz)
        recon = rearrange(recon, "(b t) c h w -> b t c h w", b=b)
        noise = rearrange(noise, "(b t) c h w -> b t c h w", b=b)
        return recon, noise
        # forward output: feature, zidx, quant, slotz, attent, recon, noise


class VVOSmdT(VqVfmOclT):
    """
    Temporal VQ-VFM-OCL with SlotMixer decoder.
    """

    forward_decode = VVOMlpT.forward_decode
