max_num = 7
resolut0 = [256, 256]
resolut1 = [16, 16]
embed_dim = 256
# 500000
total_step = 100000  # 100000 better
val_interval = total_step // 40
batch_size_t = 32  # 64 better
batch_size_v = batch_size_t
num_work = 4
lr = 2e-4

### datum

IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
transform_t = [
    # the following 2 == RandomResizedCrop: better than max sized random crop
    dict(type="RandomCrop", keys=["image", "segment"], size=None, scale=[0.75, 1]),
    dict(type="Resize", keys=["image"], size=resolut0, interp="bilinear"),
    dict(type="Resize", keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type="RandomFlip", keys=["image", "segment"], dims=[-1], p=0.5),
    dict(type="Normalize", keys=["image"], mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
transform_v = [
    dict(type="CenterCrop", keys=["image", "segment"], size=None),
    dict(type="Resize", keys=["image"], size=resolut0, interp="bilinear"),
    dict(type="Resize", keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type="Normalize", keys=["image"], mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
dataset_t = dict(
    type="MSCOCO",
    data_file="coco/train.lmdb",
    instance=True,
    extra_keys=["segment"],
    transform=dict(type="Compose", transforms=transform_t),
    base_dir=...,
)
dataset_v = dict(
    type="MSCOCO",
    data_file="coco/val.lmdb",
    instance=True,
    extra_keys=["segment"],
    transform=dict(type="Compose", transforms=transform_v),
    base_dir=...,
)
collate_fn_t = None
collate_fn_v = None

### model

model = dict(
    type="DINOSAUR",
    encode_backbone=dict(
        type="Sequential",
        modules=[
            dict(type="Interpolate", scale_factor=0.875, interp="bicubic"),
            dict(
                type="DINO2ViT",
                model_name="vit_base_patch16_224.dino",
                in_size=int(resolut0[0] * 0.875),
                rearrange=True,
                norm_out=True,
            ),
        ],
    ),
    encode_posit_embed=dict(type="Identity"),
    encode_project=dict(
        type="MLP", in_dim=768, dims=[768 * 2, embed_dim], ln="pre", dropout=0.0
    ),
    initializ=dict(type="NormalSeparat", num=max_num, dim=embed_dim),
    aggregat=dict(
        type="MetaSlot",
        num_iter=3,
        embed_dim=embed_dim,
        ffn_dim=embed_dim * 4,
        dropout=0,
        codebook_size=512,
        buffer_capacity=batch_size_t * max_num * 4,
        # if_downstream = True,
    ),
    decode=dict(
        type="BroadcastMLPDecoder",
        posit_embed=dict(
            type="LearntPositionalEmbedding",
            resolut=[resolut1[0] * resolut1[1]],
            embed_dim=embed_dim,
        ),
        backbone=dict(
            type="MLP",
            in_dim=embed_dim,
            dims=[2048] * 3 + [768 + 1],  # 2048>1024
            ln=None,
            dropout=0,
        ),
    ),
)
model_imap = dict(input="image")  # condition < random
model_omap = ["feature", "slotz", "attent", "attent2", "recon"]
ckpt_map = []  # target<-source
freez = [r"m\.encode_backbone\..*"]

### learn

param_groups = None
optimiz = dict(type="Adam", params=param_groups, lr=lr)
gscale = dict(type="GradScaler")
gclip = dict(type="ClipGradNorm", max_norm=1)

loss_fn = dict(
    recon=dict(
        metric=dict(type="MSELoss"),
        map=dict(input="output.recon", target="output.feature"),
        transform=dict(type="Detach", keys=["target"]),
    ),
)
_acc_dict_ = dict(
    # metric=...,
    map=dict(input="output.segment2", target="batch.segment"),
    transform=dict(
        type="Rearrange", keys=["input", "target"], pattern="b h w -> b (h w)"
    ),
)
metric_fn_t = dict(
    mbo=dict(metric=dict(type="mBO", skip=[]), **_acc_dict_),
)
metric_fn_v = dict(
    ari=dict(metric=dict(type="ARI", skip=[]), **_acc_dict_),
    ari_fg=dict(metric=dict(type="ARI", skip=[0]), **_acc_dict_),
    mbo=dict(metric=dict(type="mBO", skip=[]), **_acc_dict_),
    miou=dict(metric=dict(type="mIoU", skip=[]), **_acc_dict_),
)

before_step = [
    dict(
        type="CbLinearCosine",
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=total_step // 20,
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=0, # lr/1e3
    ),
]
after_forward = [
    # convert output.attent to segmentation masks: (b,n,h,w) -> (b,h,w)
    dict(type="Clone", keys=["output.attent2"], keys2=["output.segment2"]),
    dict(
        type="Lambda",
        keys=["output.segment2"],
        func=f"lambda _: ptnf.interpolate(_.detach(), size={resolut0}, mode='bilinear').argmax(1).byte()",
    ),
]
callback_t = [
    dict(type="Callback", before_step=before_step, after_forward=after_forward),
    dict(type="AverageLog", log_file=...),
]
callback_v = [
    dict(type="Callback", before_step=None, after_forward=after_forward),
    callback_t[1],
    dict(type="SaveModel", save_dir=..., since_step=total_step * 0.5),
]
