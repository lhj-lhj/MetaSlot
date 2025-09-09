max_num = 7
resolut0 = [256, 256]
resolut1 = [16, 16]
embed_dim = 256
vfm_dim = 384

total_step = 100000  # 100000 better
val_interval = total_step // 40
batch_size_t = 64 // 2  # 64 better
batch_size_v = batch_size_t
num_work = 4
lr = 4e-4 / 2  # scale with batch size

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
    type="SPOT",
    encode_backbone=dict(
        type="Sequential",
        modules=[
            dict(type="Interpolate", scale_factor=0.875, interp="bicubic"),
            dict(
                type="DINO2ViT",
                model_name="vit_small_patch14_reg4_dinov2.lvd142m",
                in_size=int(resolut0[0] * 0.875),
                rearrange=True,
                norm_out=False,  # >True
            ),
        ],
    ),
    encode_posit_embed=dict(type="Identity"),
    encode_project=dict(  # [vfm_dim * 2, embed_dim]
        type="MLP", in_dim=vfm_dim, dims=[vfm_dim, vfm_dim], ln="pre", dropout=0.0
    ),
    initializ=dict(type="NormalShared", num=max_num, dim=embed_dim),
    aggregat=dict(
        type="MetaSlot",
        num_iter=3,
        embed_dim=embed_dim,
        ffn_dim=embed_dim * 4,
        dropout=0,
        kv_dim=vfm_dim,
        codebook_size=512,
        buffer_capacity=batch_size_t * max_num * 4,
    ),
    decode=dict(
        type="AR9TransformerDecoder",
        resolut=resolut1,
        vfm_dim=vfm_dim,
        posit_embed=dict(type="Identity"),
        project1=dict(  # fc>fc+ln
            type="Sequential",
            modules=[
                dict(
                    type="Linear", in_features=vfm_dim, out_features=vfm_dim, bias=False
                ),
                dict(type="LayerNorm", normalized_shape=vfm_dim),
            ],
        ),
        project2=dict(  # fc+ln>fc
            type="Sequential",
            modules=[
                dict(
                    type="Linear",
                    in_features=embed_dim,
                    out_features=vfm_dim,
                    bias=False,
                ),
                dict(type="LayerNorm", normalized_shape=vfm_dim),
            ],
        ),
        backbone=dict(
            type="TransformerDecoder",
            decoder_layer=dict(
                type="TransformerDecoderLayer",
                d_model=vfm_dim,
                nhead=4,  # 4@384, 6@768
                dim_feedforward=vfm_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                bias=False,
            ),
            num_layers=4,
        ),
        readout=dict(type="Identity"),
        perm_t="random",
        perm_v="default",  # TODO all
    ),
)
model_imap = dict(input="image")  # condition < random
model_omap = ["feature", "slotz", "attent", "attent2", "recon"]
ckpt_map = []  # target<-source
freez = [r"^m\.encode_backbone\..*"]

### learn

param_groups = None
optimiz = dict(type="Adam", params=param_groups, lr=lr)
gscale = dict(type="GradScaler")
gclip = dict(type="ClipGradNorm", max_norm=0.3, norm_type="inf")

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
        vfinal=lr / 1e3,  # 1e3 > 1e4
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
