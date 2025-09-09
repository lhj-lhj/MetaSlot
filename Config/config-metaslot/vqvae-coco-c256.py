resolut0 = [256, 256]
num_code = 4096
embed_dim = 256

total_step = 30000  # 100000 better
val_interval = total_step // 40
batch_size_t = 64
batch_size_v = batch_size_t
num_work = 4
lr = 2e-3

### datum

IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
transform_t = [
    # the following 2 == RandomResizedCrop: better than max sized random crop
    dict(type="RandomCrop", keys=["image"], size=None, scale=[0.75, 1]),
    dict(type="Resize", keys=["image"], size=resolut0, interp="bilinear"),
    dict(type="RandomFlip", keys=["image"], dims=[-1], p=0.5),
    dict(type="Normalize", keys=["image"], mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
transform_v = [
    dict(type="CenterCrop", keys=["image"], size=None),
    dict(type="Resize", keys=["image"], size=resolut0, interp="bilinear"),
    dict(type="Normalize", keys=["image"], mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
dataset_t = dict(
    type="MSCOCO",
    data_file="coco/train.lmdb",
    instance=False,
    extra_keys=[],
    transform=dict(type="Compose", transforms=transform_t),
    base_dir=...,
)
dataset_v = dict(
    type="MSCOCO",
    data_file="coco/val.lmdb",
    instance=False,
    extra_keys=[],
    transform=dict(type="Compose", transforms=transform_v),
    base_dir=...,
)
collate_fn_t = None
collate_fn_v = None

### model

model = dict(
    type="VQVAE",
    encode=dict(
        type="Sequential",
        modules=[
            # ~=EncoderAKL (w/o mid); >>ResNet18, naive CNN
            dict(type="Interpolate", scale_factor=0.5, interp="bicubic"),
            dict(type="EncoderTAESD", se=[0, 14], gn=0),  # more convs in between: bad
            dict(type="GroupNorm", num_groups=1, num_channels=64),
            dict(type="Conv2d", in_channels=64, out_channels=embed_dim, kernel_size=1),
        ],
    ),
    decode=dict(
        type="Sequential",
        modules=[
            dict(type="Conv2d", in_channels=embed_dim, out_channels=64, kernel_size=1),
            dict(type="GroupNorm", num_groups=1, num_channels=64),
            dict(type="DecoderTAESD", se=[2, 19], gn=0),  # >> naive cnn  # in case oom
        ],
    ),
    codebook=dict(type="Codebook", num_embed=num_code, embed_dim=embed_dim),
)
model_imap = dict(input="image")
model_omap = ["encode", "zidx", "quant", "decode"]
ckpt_map = None
freez = [r"m\.encode\.1\.(?:[0-9]|10)\..*"]  # train whole decode is the best

### learn

param_groups = None
optimiz = dict(type="Adam", params=param_groups, lr=lr)
gscale = dict(type="GradScaler")
gclip = dict(type="ClipGradNorm", max_norm=1)

loss_fn = dict(
    recon=dict(
        metric=dict(type="MSELoss"),
        map=dict(input="output.decode", target="batch.image"),
        transform=dict(
            type="Resize",
            keys=["target"],
            size=[_ // 2 for _ in resolut0],
            interp="bicubic",
        ),
    ),
    align=dict(
        metric=dict(type="MSELoss"),
        map=dict(input="output.quant", target="output.encode"),
        transform=dict(type="Detach", keys=["target"]),
    ),
    commit=dict(
        metric=dict(type="MSELoss"),
        map=dict(input="output.encode", target="output.quant"),
        transform=dict(type="Detach", keys=["target"]),
        weight=0.25,
    ),
)
metric_fn_t = metric_fn_v = dict()

before_step = [
    dict(
        type="CbLinearCosine",
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=total_step // 20,
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=0,
    ),
]
callback_t = [
    dict(type="Callback", before_step=before_step),
    dict(type="AverageLog", log_file=...),
]
callback_v = [
    dict(type="Callback", before_step=None),
    callback_t[1],
    dict(type="SaveModel", save_dir=..., since_step=total_step * 0.5),
]
