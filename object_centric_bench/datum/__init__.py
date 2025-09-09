from ..utils import register_module
# from .dataset import DataLoader, ChainDataset, ConcatDataset, StackDataset
from .dataset import DataLoader, ChainDataset, ConcatDataset
from .dataset_birds import Birds
from .dataset_clevrtex import ClevrTex
from .dataset_coco import MSCOCO
from .dataset_flowers import Flowers
from .dataset_movi import MOVi
from .dataset_pets import Pets
from .dataset_shapestacks import ShapeStacks
from .dataset_voc import PascalVOC
from .dataset_bicos import BiCoS
from .dataset_habitat import Habitat
from .dataset_ytvis import YTVIS
from .dataset_physion import Physion, PhysionSlotz

from .transform import (
    Lambda,
    Clip,
    Filter,
    Mask,
    Normalize,
    Concat,
    Rearrange,
    Repeat,
    Clone,
    PadTo1,
    RandomFlip,
    RandomCrop,
    CenterCrop,
    LogPlus,
    Resize,
    Slice1,
    SliceTo1,
    RandomSliceTo1,
    StridedRandomSlice1,
    ToDevice,
    Detach,
    TupleToNumber,
)

[register_module(_) for _ in locals().values() if isinstance(_, type)]
