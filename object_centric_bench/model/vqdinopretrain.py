from copy import deepcopy

from einops import rearrange, repeat
from torchvision import transforms
from xformers.ops.fmha.attn_bias import BlockDiagonalMask
import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf
import torch.nn.init as ptni
import torch.nn.utils.parametrizations as ptnup


# IMAGENET_MEAN = (0.485, 0.456, 0.406)
# IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN = pt.tensor([[[123.675]], [[116.28]], [[103.53]]])
IMAGENET_STD = pt.tensor([[[58.395]], [[57.12]], [[57.375]]])


class DINODataAugment:

    def __init__(
        self,
        global_crop_scale,
        local_crop_scale,
        local_crop_num,
        global_crop_size=224,
        local_crop_size=96,
        image_key="image",
        global_key="global_crop",
        local_key="local_crop",
    ):
        self.global_crop_scale = global_crop_scale
        self.local_crop_scale = local_crop_scale
        self.local_crop_num = local_crop_num
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.image_key = image_key
        self.global_key = global_key
        self.local_key = local_key

        self.geometric_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crop_size,
                    scale=global_crop_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        self.geometric_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crop_size,
                    scale=local_crop_scale,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # ``transforms.RandomApply.p``: probability to keep the original
        ColorJitterRandom = lambda p, **k: transforms.RandomApply(
            [transforms.ColorJitter(**k)], p=p
        )
        GaussianBlurRandom = lambda p, **k: transforms.RandomApply(
            [transforms.GaussianBlur(**k)], p=p
        )

        colorjit_grayscale = transforms.Compose(
            [
                ColorJitterRandom(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        # normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        self.photometric_global1 = transforms.Compose(
            [
                colorjit_grayscale,
                transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
                # normalize,
            ]
        )
        self.photometric_global2 = transforms.Compose(
            [
                colorjit_grayscale,
                GaussianBlurRandom(kernel_size=9, sigma=(0.1, 2.0), p=0.9),
                transforms.RandomSolarize(threshold=128, p=0.2),
                # normalize,
            ]
        )
        self.photometric_local = transforms.Compose(
            [
                colorjit_grayscale,
                GaussianBlurRandom(kernel_size=9, sigma=(0.1, 2.0), p=0.5),
                # normalize,
            ]
        )

    def __call__(self, **sample: dict) -> dict:
        """from ``image`` to ``global_crop`` and ``local_crop``"""
        sample2 = sample.copy()
        image = sample2[self.image_key]

        global_crops = [
            self.photometric_global1(self.geometric_global(image)),
            self.photometric_global2(self.geometric_global(image)),
        ]
        # sample2[self.global_key] = global_crops
        sample2[self.global_key] = [
            (_ - IMAGENET_MEAN) / IMAGENET_STD for _ in global_crops
        ]

        local_crops = [
            self.photometric_local(self.geometric_local(image))
            for _ in range(self.local_crop_num)
        ]
        # sample2[self.local_key] = local_crops
        sample2[self.local_key] = [
            (_ - IMAGENET_MEAN) / IMAGENET_STD for _ in local_crops
        ]

        # image: (c,h,w); global_crop: (a1,c,h,w); local_crop: (a2,c,h,w)
        return sample2


class DINOCollateMask:

    def __init__(
        self,
        mask_gen,
        mask_ratios: tuple,
        mask_prob: float,
        n_token,  # (img_size/patch_size)**2=256
        global_key="global_crop",
        local_key="local_crop",
        mask_key="mask",
        weight_key="weight",
    ):
        self.mask_gen = mask_gen
        self.mask_ratio_tuple = mask_ratios
        self.mask_prob = mask_prob
        self.n_token = n_token
        self.global_key = global_key
        self.local_key = local_key
        self.mask_key = mask_key
        self.weight_key = weight_key

    def __call__(self, samples: list) -> dict:
        global_crop = list(zip(*[_[self.global_key] for _ in samples]))
        global_crop = pt.stack([pt.stack(_) for _ in global_crop])  # pt.tensor(...) ???
        local_crop = list(zip(*[_[self.local_key] for _ in samples]))
        local_crop = pt.stack([pt.stack(_) for _ in local_crop])  # pt.tensor(...) ???

        ng, b, c, h, w = global_crop.shape
        n_maskedsample = int(ng * b * self.mask_prob)

        mask = []  # a*b*(n,)
        probs = pt.linspace(*self.mask_ratio_tuple, n_maskedsample + 1)
        for i in range(0, n_maskedsample):
            pmin = probs[i]
            pmax = probs[i + 1]
            masked_area = int(self.n_token * np.random.uniform(pmin, pmax))
            m = pt.BoolTensor(self.mask_gen(masked_area))
            mask.append(m)
        for i in range(n_maskedsample, ng * b):
            m = pt.BoolTensor(self.mask_gen(0))
            mask.append(m)

        np.random.shuffle(mask)
        mask = rearrange(
            mask, "(a b) h w -> a b (h w)", b=b
        )  # (a b) n -> a b n TODO ???
        weight = (1 / mask.sum(2, True).clamp(min=1.0)).expand_as(mask)[mask]  # (?,)
        weight /= ng * b  # then in ibot loss ``loss/(ng*b)`` is not needed

        samples2 = {
            self.global_key: global_crop,  # (a,b,c,h,w)
            self.local_key: local_crop,  # (a,b,c,h,w)
            self.mask_key: mask,  # (a,b,n), bool
            self.weight_key: weight,  # (?,)
        }
        return samples2


class MaskGenerator:

    def __init__(
        self,
        input_size: int,  # 16  # image_size//patch_size
        min_masked_area=4,
        max_masked_ratio=0.5,  # 128  # 0.5*(img_size//patch_size)**2
        min_aspect=0.3,
        max_aspect=None,
    ):
        self.height = self.width = input_size
        self.min_masked_area = min_masked_area
        self.max_masked_area = max_masked_ratio * input_size**2
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspects = (np.log(min_aspect), np.log(max_aspect))

    @pt.no_grad()
    def __call__(self, masked_area=0):
        """
        masked_area: int
        mask: shape=(h,w); dtype=bool
        """
        mask = pt.zeros([self.height, self.width], dtype=pt.bool)  # (h,w)
        count = 0
        while count < masked_area:
            max_masked_area = masked_area - count
            max_masked_area = min(max_masked_area, self.max_masked_area)
            delta = self._mask(mask, max_masked_area)
            if delta == 0:  # always fail, no need to try any more
                break
            else:
                count += delta
        return mask

    def _mask(self, mask, max_masked_area, max_trial=10):
        delta = 0

        for _ in range(max_trial):
            area = np.random.uniform(self.min_masked_area, max_masked_area)
            aspect = np.exp(np.random.uniform(*self.log_aspects))
            h = int(round(np.sqrt(area * aspect)))
            w = int(round(np.sqrt(area / aspect)))

            if h >= self.height or w >= self.width:
                continue
            area = h * w

            t = np.random.randint(0, self.height - h)
            l = np.random.randint(0, self.width - w)
            p = mask[t : t + h, l : l + w]

            delta = area - p.sum()
            if delta <= 0 or delta > max_masked_area:
                assert delta >= 0
                continue

            p[...] = 1
            break

        return delta


###


class VQDINO2Meta(nn.Module):

    def __init__(self, backbone, vq, dinohead, ibothead, s_tau=0.1, t_tau=0.07):
        super().__init__()
        self.s = nn.ModuleDict(
            dict(backbone=backbone, dinohead=dinohead, ibothead=ibothead)
        )
        self.s.register_buffer("tau", pt.tensor(s_tau, dtype=pt.float))
        self.t = deepcopy(self.s).eval()
        self.t.register_buffer("tau", pt.tensor(t_tau, dtype=pt.float))
        self.vq = vq

    def train(self, mode=True):
        self.s.train(mode), self.t.eval()
        return self

    def vq1d(self, patchtoken):
        ab, n, c = patchtoken.shape
        h = w = int(n**0.5)
        assert h * w == n
        patchtoken = rearrange(patchtoken, "ab (h w) c -> ab c h w", h=h)
        encode, zidx, quant, decode = self.vq(patchtoken)
        encode = rearrange(encode, "ab c h w -> ab (h w) c")
        zidx = rearrange(zidx, "ab h w -> ab (h w)")
        quant = rearrange(quant, "ab c h w -> ab (h w) c")
        decode = rearrange(decode, "ab c h w -> ab (h w) c")
        return encode, zidx, quant, decode

    def forward(
        self,
        global_crop,  # (a=2,b,c,h,w), float
        local_crop,  # (a=8,b,c,h,w), float
        mask,  # (a,b,n), bool
    ):
        ng, b, _, h, w = global_crop.shape
        assert ng == 2
        nl, b, _, h, w = local_crop.shape
        # assert nl == 8
        g_crop = rearrange(global_crop, "a b c h w -> (a b) c h w")
        l_crop = rearrange(local_crop, "a b c h w -> (a b) c h w")
        mask = rearrange(mask, "a b n -> (a b) n")

        sg_token = self.s.backbone(g_crop, mask)  # (a*b,1+n,c)
        sg_classtoken = sg_token[:, 0, :]  # (a*b,c)
        sg_patchtoken = sg_token[:, 1:, :]  # (a*b,n,c)
        sl_token = self.s.backbone(l_crop, None)  # (a*b,1+n,c)
        sl_classtoken = sl_token[:, 0, :]  # (a*b,c)
        sl_patchtoken = sl_token[:, 1:, :]  # (a*b,n,c)

        sg_encode, sg_zidx, sg_quant, sg_decode = self.vq1d(sg_patchtoken)
        sg_patchtoken = sg_decode

        _, c = sg_classtoken.shape

        s_attnbias, s_classtoken = BlockDiagonalMask.from_tensor_list(
            [sg_classtoken[None], sl_classtoken[None]]
        )
        s_classtoken_dino = self.s.dinohead(s_classtoken)
        sg_classtoken_dino, sl_classtoken_dino = s_attnbias.split(s_classtoken_dino)
        sg_classtoken_dino = sg_classtoken_dino[0]
        sl_classtoken_dino = sl_classtoken_dino[0]

        sg_patchtoken_masked = sg_patchtoken[mask]
        sg_patchtoken_ibot = self.s.ibothead(sg_patchtoken_masked)  # (?,c)

        with pt.no_grad():  # , is_training=True
            tg_token = self.t.backbone(g_crop)  # (a*b,1+n,c)
            tg_classtoken = tg_token[:, 0, :]  # (a*b,c) a=2
            tg_patchtoken = tg_token[:, 1:, :]  # (a*b,n,c)

            tg_encode, tg_zidx, tg_quant, tg_decode = self.vq1d(tg_patchtoken)
            tg_patchtoken = tg_decode

            tg_classtoken_dino = self.t.dinohead(tg_classtoken)

            tg_patchtoken_masked = tg_patchtoken[mask]
            tg_patchtoken_ibot = self.t.ibothead(tg_patchtoken_masked)  # (?,c)

        # if self.training:  # both train and eval
        sg_classtoken_dino = sg_classtoken_dino / self.s.tau
        sl_classtoken_dino = sl_classtoken_dino / self.s.tau
        sg_patchtoken_ibot = sg_patchtoken_ibot / self.s.tau
        tg_classtoken_dino = tg_classtoken_dino / self.t.tau
        tg_patchtoken_ibot = tg_patchtoken_ibot / self.t.tau

        with pt.no_grad():  # TODO move out to loss wrap or callback.after_forward
            # tg_classtoken_dino = pt.stack(  # a=2, switch
            #     [tg_classtoken_dino[b:], tg_classtoken_dino[:b]]
            # )
            tg_classtoken_dino = sinkhorn_knopp(tg_classtoken_dino)  # center
            tg_patchtoken_ibot = sinkhorn_knopp(tg_patchtoken_ibot)  # center

        sg_classtoken = rearrange(sg_classtoken, "(a b) c -> a b c", b=b)
        sg_classtoken_dino = rearrange(sg_classtoken_dino, "(a b) c -> a b c", b=b)
        sl_classtoken_dino = rearrange(sl_classtoken_dino, "(a b) c -> a b c", b=b)
        tg_classtoken_dino = rearrange(tg_classtoken_dino, "(a b) c -> a b c", b=b)

        return (
            sg_classtoken,
            sg_classtoken_dino,
            sl_classtoken_dino,
            sg_patchtoken_ibot,
            pt.stack([_ for _ in tg_classtoken_dino][::-1]),  # a=2, switch
            tg_patchtoken_ibot,
            sg_encode,  # commit
            sg_quant,  # to
            sg_zidx,
            tg_encode,  # to
            tg_quant,  # align
            tg_zidx,
        )


def sinkhorn_knopp(target, n_iter=3):
    assert target.ndim == 2
    target = target.float()  # TODO remove
    Q = target.exp()
    b, k = Q.shape  # number of prototypes; number of samples
    Q /= Q.sum()  # make the matrix sums to 1
    for _ in range(n_iter):
        Q /= Q.sum(0, True) * k  # normalize col: total weight per prototype must be 1/K
        Q /= Q.sum(1, True) * b  # normalize row: total weight per sample must be 1/B
    Q *= b  # the rows must sum to 1 so that Q is an assignment
    return Q


class DINOHead(nn.Module):

    def __init__(self, in_dim, out_dim=131072, hidden_dim=2048, bottleneck_dim=384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim, bias=True),
        )
        self.apply(self._init_weights)
        self.last = ptnup.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # self.last.weight_g.data.fill_(1)  # deprecated
        self.last.parametrizations.weight.original0.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            ptni.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == pt.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last(x)
        return x


class KoLeoLoss:

    @staticmethod
    @pt.no_grad()
    def nearest_neighbors(x):
        """Batch pairwise nearest neighbors for L2-normalized vectors."""
        b, n, c = x.shape
        dot = pt.einsum("bnc,bmc->bnm", x, x)  # (b,n,n)
        dot.view(b, -1)[:, :: n + 1].fill_(-1)  # Trick to fill diagonal with -1
        idx = dot.argmax(2)  # max inner prod -> min distance
        return idx

    def __call__(self, input, eps=1e-8):
        """
        input: shape=(b,c) or (a,b,c)
        """
        assert input.ndim in [2, 3]
        if input.ndim == 2:
            input = input[None, :, :]
        loss_all = 0
        with pt.amp.autocast(enabled=False, device_type="cuda"):
            input = ptnf.normalize(input, eps=eps, p=2, dim=-1)
            idx = self.nearest_neighbors(input)
            dist = ptnf.pairwise_distance(  # bnd,bnd->bn
                input, input.gather(1, idx[:, :, None].expand_as(input)), 2, eps
            )
            loss = -(dist + eps).log().mean()
        loss_all += loss
        return loss_all


class DINOClassLoss:

    def __call__(self, input, target):
        """Cross-entropy between softmax outputs of the teacher and student networks.
        input: shape=(a1,b,c)
        target: shape=(a2,b,c); should be already normalized by sinkhorn_knopp
        """
        a1, b, c = input.shape  # assert input.ndim == target.ndim == 3
        a2, b, c = target.shape
        input = repeat(input, "(a1 1) b c -> (a1 a2 b) c", a2=a2)  # consume more memory
        target = repeat(target, "(1 a2) b c -> (a1 a2 b) c", a1=a1)
        loss = ptnf.cross_entropy(input, target)
        return loss
        # XXX not understand the coefficients
        # official:
        #   n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        #   n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops
        #   output loss := dino_loss / (n_global_crops_loss_terms + n_local_crops_loss_terms)
        # mine:
        #   output loss := official dino_loss / (ng * nl)
        # without such coefficients, the error is 1e-6 level
        # with such coefficients, the error is 1e-1 level


class iBOTPatchLoss:

    def __call__(self, input, target, weight=None):
        """
        input: shape=(?,c)
        target: shape=(?,c); should be already normalized by sinkhorn_knopp
        weight: shape=(?,)
        """
        assert input.ndim == target.ndim == 2
        input = input.float()  # TODO remove ???
        target = target.float()  # TODO remove ???
        loss = ptnf.cross_entropy(input, target, reduction="none")
        if weight is not None:
            loss = loss * weight
        return loss.sum()


####


from object_centric_bench.datum import *
from object_centric_bench.learn import *
from object_centric_bench.model import *


def main():
    max_num = 10 + 1
    resolut1 = [32, 32]
    resolut2 = [128, 128]
    num_code = 4096
    embed_dim = 256

    total_step = 50000
    val_interval = total_step // 50
    batch_size_t = 32  # // 2
    batch_size_v = batch_size_t
    num_work = 4
    lr = 2e-4  # 4e-3

    ### datum

    global_size = 224
    local_size = 98
    patch_size = 14
    embed_size = global_size // patch_size

    transforms = [
        Filter(keys=["image"]),
        DINODataAugment(
            global_crop_scale=[0.32, 1.0],  # TODO meaning ??? -> resize coco to lmdb
            local_crop_scale=[0.05, 0.32],  # TODO meaning ???
            local_crop_num=8,
            global_crop_size=global_size,  # 448
            local_crop_size=local_size,
            image_key="image",
            global_key="global_crop",
            local_key="local_crop",
        ),
    ]
    dataset_t = MSCOCO(
        data_file="coco/train.lmdb",
        transform=Compose(transforms=transforms),
        base_dir=...,
    )
    dataset_v = MSCOCO(
        data_file="coco/val.lmdb",
        transform=Compose(transforms=transforms),
        base_dir=...,
    )
    collate_fn = DINOCollateMask(
        mask_gen=MaskGenerator(input_size=embed_size, max_masked_ratio=0.5),
        mask_ratios=[0.1, 0.5],
        mask_prob=0.5,
        n_token=embed_size**2,
        global_key="global_crop",
        local_key="local_crop",
        mask_key="mask",
        weight_key="weight",
    )

    ### model

    model = VQDINO2Meta(  # ema backbone.mask_token, dinohead and ibothead only
        backbone=DINO2ViT(
            model_name="vit_small_patch14_dinov2.lvd142m",
            in_size=global_size,
            patch_size=patch_size,
            drop_path_rate=0.3,
            drop_path_uniform=True,
            rearrange=False,
        ),
        vq=...,
        dinohead=DINOHead(
            in_dim=384, out_dim=65536, hidden_dim=2048, bottleneck_dim=256
        ),
        ibothead=DINOHead(
            in_dim=384, out_dim=65536, hidden_dim=2048, bottleneck_dim=256
        ),
    )
    model_imap = dict(global_crop="global_crop", local_crop="local_crop", mask="mask")
    model_omap = [
        "sg_classtoken",
        "sg_classtoken_dino",
        "sl_classtoken_dino",
        "sg_patchtoken_ibot",
        "tg_classtoken_dino",
        "tg_patchtoken_ibot",
        "sg_zero_",
        "sg_quant_",
        "sg_idx",
        "tg_zero_",
        "tg_quant_",
        "tg_idx",
    ]
    ckpt_map = []
    freez = [r"^m\.s\.backbone\.(?!mask_token).*", r"^m\.t\..*"]

    ### learn

    param_groups = dict(
        coarse=r"^m\.(s|vq)\..*",  # only m.s or m.vq
        fine={
            # different ``lr_mult``s for short; const 1 for l/g except patch_embed
            # - only m.s.backbone.patch_embed.* not *bias$
            r"^m\.s\.backbone\.patch_embed\.(?!.*bias$)": dict(lr_mult=0.2, wd_mult=1),
            # - not m.s.backbone.patch_embed.weight; not *bias$, *norm*, *gamma*
            r"^m\.s\.(?!backbone\.patch_embed\.)(?!.*(norm|gamma)).*(?<!bias)$": dict(
                lr_mult=1, wd_mult=1
            ),
            # - only *bias$, *norm*, *gamma*
            r"^m\.s\..*(bias$|norm|gamma).*": dict(lr_mult=1, wd_mult=0),
            # - only m.vq.*
            r"^m\.vq\..*": dict(lr_mult=1, wd_mult=0),
        },
    )
    optimiz = AdamW(params=param_groups, lr=lr, betas=(0.9, 0.999))
    gscale = GradScaler()
    gclip = ClipGradNorm(max_norm=3)

    loss_fn = dict(
        coleo=dict(
            metric=KoLeoLoss(),
            map=dict(input="output.sg_classtoken"),
            weight=0.1,
        ),
        dinog=dict(
            metric=DINOClassLoss(),
            map=dict(
                input="output.sg_classtoken_dino", target="output.tg_classtoken_dino"
            ),
            transform=Rearrange(keys=["input", "target"], pattern="a b c -> 1 (a b) c"),
            weight=1.0 * 2,
        ),
        dinol=dict(
            metric=DINOClassLoss(),
            map=dict(
                input="output.sl_classtoken_dino", target="output.tg_classtoken_dino"
            ),
        ),
        ibotg=dict(
            metric=iBOTPatchLoss(),
            map=dict(
                input="output.sg_patchtoken_ibot",
                target="output.tg_patchtoken_ibot",
                weight="batch.weight",
            ),
        ),
    )
    metric_fn = dict()

    before_step = [
        ToDevice(
            keys=[
                "batch.global_crop",
                "batch.local_crop",
                "batch.mask",
                "batch.weight",
            ],
        ),
        CbLinearCosine(  # learning rate
            assigns=["for _ in optimiz.param_groups: _['lr']=_['lr_mult']*value"],
            nlin=total_step // 10,
            ntotal=total_step,
            vstart=0,
            vbase=lr,
            vfinal=1e-6,
        ),
        CbCosine(  # weight decay
            assigns=[
                "for _ in optimiz.param_groups: _['weight_decay']=_['wd_mult']*value"
            ],
            ntotal=total_step,
            vbase=0.04,
            vfinal=0.2,
        ),
        CbCosine(  # momentum
            assigns=["callback_t[0].after_step[0].momentum=value"],
            ntotal=total_step,
            vbase=0.006,  # 1-0.994
            vfinal=0,  # 1-1
        ),
        # XXX freeze last layer of dino/ibot head for the first epoch
    ]
    after_step = [  # TODO check ema
        DINOEMA(  # ema on backbone.mask_token, dinohead and ibothead only
            source=r"^m\.s\.(backbone\.mask_token|dinohead\.|ibothead\.).*",
            target=r"^m\.t\.(backbone\.mask_token|dinohead\.|ibothead\.).*",
            momentum=0.006,  # from 1-0.994 to 1-1
        ),
    ]
    callback_t = [
        Callback(before_step=before_step, after_step=after_step),
        AverageLog(log_file=...),
    ]
    callback_v = [
        Callback(before_step=before_step[:1]),
        callback_t[1],
        SaveModel(save_dir=..., since_step=total_step * 0.5, key=r".*"),
    ]


if __name__ == "__main__":
    main()
