from einops import rearrange, repeat
import torch as pt
import torch.nn as nn


class SAVi(nn.Module):
    """
    (SAVi) Conditional Object-Centric Learning from Video
    SAVi++: Towards End-to-End Object-Centric Learning from Real-World Videos

    Differences between SAVi and SAVi++, as well as our unified implementation:
    - data augmentation: naive | advanced | advanced
    - backbone: naive CNN | ResNet + tfeb*4 | VFM
    - reconstruction target: flow | depth | flow or depth
    """

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        transit,
        decode,
    ):
        super().__init__()
        self.encode_backbone = encode_backbone
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.aggregat = aggregat
        self.transit = transit
        self.decode = decode
        self.reset_parameters(
            [self.encode_posit_embed, self.encode_project, self.aggregat, self.transit]
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
        recon, attent2 = self.decode(clue, slotz.flatten(0, 1))
        recon = rearrange(recon, "(b t) c h w -> b t c h w", b=b)
        attent2 = rearrange(attent2, "(b t) n h w -> b t n h w", b=b)

        return feature, slotz, attent, attent2, recon


class BroadcastCNNDecoder(nn.Module):
    """SAVi's decoder."""

    def __init__(self, posit_embed, backbone):
        super().__init__()
        self.posit_embed = posit_embed
        self.backbone = backbone

    def forward(self, input, slotz):
        """
        - input: destructed target
        - slotz: slots, shape=(b,n,c)
        """
        h, w = input
        b, n, c = slotz.shape

        mixture = repeat(slotz, "b n c -> (b n) h w c", h=h, w=w)
        mixture = self.posit_embed(mixture)  # (b*n,h,w,c)
        mixture = mixture.permute(0, 3, 1, 2)  # (b*n,c,h,w)
        mixture = self.backbone(mixture)  # (b*n,c=3+1,16*h,16*w)

        recon, alpha = mixture[:, :-1, :, :], mixture[:, -1:, :, :]
        recon = rearrange(recon, "(b n) c h w -> b n c h w", b=b)
        alpha = rearrange(alpha, "(b n) 1 h w -> b n 1 h w", b=b)
        # faster than pt.einsum()
        recon = (recon * alpha.softmax(1)).sum(1)  # (b,c,h,w)

        attent2 = alpha[:, :, 0, :, :]  # (b,n,h,w)
        return recon, attent2
