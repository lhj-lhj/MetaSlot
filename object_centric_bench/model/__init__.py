from ..utils import register_module
from .basic import (
    ModelWrap,
    Sequential,
    ModuleList,
    Embedding,
    Conv2d,
    PixelShuffle,
    ConvTranspose2d,
    Interpolate,
    Linear,
    Dropout,
    AdaptiveAvgPool2d,
    GroupNorm,
    LayerNorm,
    ReLU,
    GELU,
    SiLU,
    Mish,
    MultiheadAttention,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    CNN,
    MLP,
    ResNet,
    ResNetSlice,
    Identity,
    DINO2ViT,
    DINO2ViT_P16,
    SAM2Hiera,
    EncoderVAESD,
    DecoderVAESD,
    EncoderTAESD,
    DecoderTAESD,
)
from .ocl import (
    SlotAttention,
    NormalShared,
    NormalSeparat,
    CartesianPositionalEmbedding2d,
    LearntPositionalEmbedding,
    VQVAE,
    Codebook,
    LearntPositionalEmbedding,
)
from .metaslot import (
    MetaSlot,
)
from .savi import SAVi, BroadcastCNNDecoder
from .slatesteve import SLATE, STEVE, ARTransformerDecoder
from .dinosaur import DINOSAUR, DINOSAURT, BroadcastMLPDecoder
from .videosaur import VideoSAUR, SlotMixerDecoder
from .slotdiffusion import (
    SlotDiffusion,
    ConditionDiffusionDecoder,
    NoiseSchedule,
    UNet2dCondition,
)
from .spot import SPOT, SPOTDistill, AR9TransformerDecoder, ARRandTransformerDecoder
from .spot_zero import SPOTZero  # TODO XXX
from .vaez import VQVAEZ, QuantiZ, VQVAEZGrouped, VQVAEZMultiScale, LinearPinv
from .vqvfmocl import VVOTfd, VVOTfdT, VVOMlp, VVOMlpT, VVODfz, VVODfzT, VVOSmdT
from .slotformer import SlotFormer, Rollouter, RelationNetwork
from .initq import InitqDINOSAUR, TwoWayTransformer

[register_module(_) for _ in locals().values() if isinstance(_, type)]
