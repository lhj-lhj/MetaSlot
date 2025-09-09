from einops import rearrange, repeat
import torch as pt
import torch.nn as nn


class DINOSAUR(nn.Module):

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        decode,
        if_vq = False,
    ):
        super().__init__()
        self.encode_backbone = encode_backbone
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.aggregat = aggregat
        self.decode = decode
        self.if_vq = if_vq
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
        if self.if_vq:
            slotz, attent, idx = self.aggregat(encode, query)
        else:
            slotz, attent = self.aggregat(encode, query)
        # slotz, attent, indices, vq_loss, commitment_loss = self.aggregat(encode, query)
        # print("slotz: ", slotz.shape)
        # print("attent: ", attent.shape)
            # print(slotz.shape, attent.shape) torch.Size([32, 7, 256]) torch.Size([32, 7, 256])
        attent = rearrange(attent, "b n (h w) -> b n h w", h=h)
        
        clue = [h, w]
        recon, attent2 = self.decode(clue, slotz)  # (b,h*w,c)
        recon = rearrange(recon, "b (h w) c -> b c h w", h=h)
        attent2 = rearrange(attent2, "b n (h w) -> b n h w", h=h)
        if self.if_vq:
            return feature, slotz, attent, attent2, recon, idx
        else:
            return feature, slotz, attent, attent2, recon
        # segment acc: attent < attent2
        # return feature, slotz, attent, attent2, recon, indices, vq_loss, commitment_loss


class DINOSAURT(DINOSAUR):

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        decode,
        transit,
    ):
        super().__init__(
            encode_backbone,
            encode_posit_embed,
            encode_project,
            initializ,
            aggregat,
            decode,
        )
        self.transit = transit
        self.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat, self.transit]
        )

    def forward(self, input, condit=None):
        """
        - input: video, shape=(b,t,c,h,w)
        - condit: condition, shape=(b,t,n,c)
        """
        b, t, c, h, w = input.shape
        input = input.flatten(0, 1)  # (b*t,c,h,w)

        feature = self.encode_backbone(input).detach()  # (b*t,c,h,w)
        bt, c, h, w = feature.shape
        encode = feature.permute(0, 2, 3, 1)  # (b*t,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b*t,h*w,c)
        encode = self.encode_project(encode)

        feature = rearrange(feature, "(b t) c h w -> b t c h w", b=b)
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

        clue = [h, w]
        recon, attent2 = self.decode(clue, slotz.flatten(0, 1))  # (b*t,h*w,c)
        recon = rearrange(recon, "(b t) (h w) c -> b t c h w", b=b, h=h)
        attent2 = rearrange(attent2, "(b t) n (h w) -> b t n h w", b=b, h=h)

        return feature, slotz, attent, attent2, recon


class BroadcastMLPDecoder(nn.Module):  # TODO BroadcastCNNDecoder
    """DINOSAUR's decoder."""

    def __init__(self, posit_embed, backbone):
        super().__init__()
        self.posit_embed = posit_embed
        self.backbone = backbone

    def forward(self, input, slotz, smask=None):
        """
        - input: destructed target, shape=(b,m,c)
        - slotz: slots, shape=(b,n,c)
        - smask: slots' mask, shape=(b,n), dtype=bool
        """
        h, w = input
        b, n, c = slotz.shape

        mixture = repeat(slotz, "b n c -> (b n) hw c", hw=h * w)
        mixture = self.posit_embed(mixture)
        mixture = self.backbone(mixture)

        recon, alpha = mixture[:, :, :-1], mixture[:, :, -1:]
        recon = rearrange(recon, "(b n) hw c -> b n hw c", b=b)
        alpha = rearrange(alpha, "(b n) hw 1 -> b n hw 1", b=b)
        if smask is not None:
            alpha = alpha.where(smask[:, :, None, None], -pt.inf)
        # faster than pt.einsum()
        alpha = alpha.softmax(1)
        recon = (recon * alpha).sum(1)  # (b,hw,c)

        attent2 = alpha[:, :, :, 0]  # (b,n,hw)
        return recon, attent2
