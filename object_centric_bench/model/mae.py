""" TODO XXX 20240605
Code
---
masks = []
for batch in range(batch_size):
    mask = []
    for row in range(num_rows):
        mask_row = mask1.flatten() * mask2.flatten()  # mask for one row
        mask.append(mask_row)
    mask = pt.stack(mask_row)  # (h*w,h*w)
    masks.append(mask)
masks = pt.stack(masks)  (b,h*w,h*w)

Notes
---
- mask1: remove a patch itself
- mask2: remove patchs around the patch above
    - 1st order neighbor, 2nd, 3rd,.. 
        - 0~multiple combination of these;
    - left, right, above, below, left-above, right-above, left-below, right-below
        - 1~multiple combination of these;
    - combination of the above two series
"""

import math

from einops import rearrange
import torch as pt
import torch.nn as nn


class SLATEMAE(nn.Module):

    def __init__(
        self,
        mediat,
        h1w1,
        encode_backbone,
        h2w2,
        encode_posit_embed,
        encode_project,
        initializ,
        correct,
        mask_ratio,
        decode_m,
        decode_posit_embed,
        decode_backbone,
        decode_readout,
    ):
        super().__init__()
        self.mediat = mediat  # type: dVAE
        self.h1w1 = h1w1
        self.h1w1p = h1w1[0] * h1w1[1]
        self.encode_backbone = encode_backbone
        self.h2w2 = h2w2
        self.encode_posit_embed = encode_posit_embed
        self.encode_project = encode_project
        self.initializ = initializ
        self.correct = correct
        self.mask_ratio = mask_ratio
        self.decode_m = decode_m
        self.decode_posit_embed = decode_posit_embed
        self.decode_backbone = decode_backbone
        self.decode_readout = decode_readout
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():  # ``zero init conv/linear/gru bias`` converges faster
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
        image: in shape (b,c,h,w)
        condit: in shape (b,n,c)
        """
        b, c, h, w = input.shape

        encode = self.encode_backbone(input)  # (b,c,h,w)
        encode_ = encode.permute(0, 2, 3, 1)  # (b,h,w,c)
        encode_ = self.encode_posit_embed(encode_)
        encode_ = encode_.flatten(1, 2)  # (b,h*w,c)
        encode_ = self.encode_project(encode_)

        hidden = self.initializ(b if condit is None else condit)  # (b,n,c)
        correct, attent_ = self.correct(encode_, hidden)
        attent = rearrange(attent_, "b n (h w) -> b n h w", h=self.h2w2[0])

        zsoft, zidx, quant, decode1 = self.mediat(input)  # (b,c,h,w)

        imask_ = (
            pt.rand([b, self.h1w1p, 1], dtype=quant.dtype, device=quant.device)
            > self.mask_ratio
        )
        quant_ = rearrange(quant, "b c h w -> b (h w) c")
        token = quant_.where(imask_, self.decode_m.expand_as(quant_))
        token = self.decode_posit_embed(token)
        token = self.decode_backbone(token, correct)
        prob_ = self.decode_readout(token)  # (b,h*w,c)
        prob = rearrange(prob_, "b (h w) c -> b c h w", h=self.h1w1[0])

        segment = attent.argmax(1)  # (b,h,w)
        # return zidx, prob, segment, correct, attent  # TODO XXX
        return quant, prob, segment, correct, attent
