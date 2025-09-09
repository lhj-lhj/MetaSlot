import lpips
import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf

from object_centric_bench.utils import DictTool
from object_centric_bench.learn.utils import intersection_over_union, hungarian_matching


class MetricWrap(nn.Module):
    """"""

    def __init__(self, detach=False, **metrics):
        super().__init__()
        self.detach = detach
        self.metrics = metrics

    def forward(self, **pack: dict) -> dict:
        if self.detach:
            with pt.inference_mode(True):
                return self._forward(**pack)
        else:
            return self._forward(**pack)

    def _forward(self, **pack: dict) -> dict:
        metrics = {}
        for key, value in self.metrics.items():
            # assert "map" in value
            kwds = {t: DictTool.getattr(pack, s) for t, s in value["map"].items()}
            if self.detach:
                kwds = {k: v.detach() for k, v in kwds.items()}
            if "transform" in value:
                kwds = value["transform"](**kwds)
            # assert "metric" in value
            metric = value["metric"](**kwds)
            if "weight" in value:
                metric = metric * value["weight"]
            metrics[key] = metric
        return metrics

    def compile(self):
        for k, v in self.metrics.items():
            v["metric"] = pt.compile(v["metric"].__call__)


class CrossEntropyLoss:
    """``nn.CrossEntropyLoss``."""

    def __init__(self, reduce="mean"):
        self.reduce = reduce  # mean, None

    def __call__(self, input, target):
        # assert input.ndim == target.ndim + 1
        return ptnf.cross_entropy(input, target, reduction=self.reduce)


class CrossEntropyLossGrouped(CrossEntropyLoss):
    """"""

    def __init__(self, groups, reduce="mean"):
        super().__init__(reduce)
        self.groups = groups

    def __call__(self, input, target):
        """
        input: shape=(b,g*c,..), dtype=float
        target: shape=(b,g,..), dtype=int64
        """
        assert input.ndim == target.ndim
        start = 0
        loss = []
        for g, gc in enumerate(self.groups):
            end = start + gc
            ce = super().__call__(input[:, start:end], target[:, g])
            start = end
            loss.append(ce)
        assert end == input.size(1)
        return sum(loss)


class L1Loss:
    """``nn.L1Loss``."""

    def __init__(self, reduce="mean"):
        self.reduce = reduce  # mean, None

    def __call__(self, input, target=None):
        if target is None:
            target = pt.zeros_like(input)
        assert input.ndim == target.ndim >= 1
        return ptnf.l1_loss(input, target, reduction=self.reduce)


class MSELoss:
    """``nn.MSELoss``."""

    def __init__(self, reduce="mean"):
        self.reduce = reduce  # mean, None

    def __call__(self, input, target):
        assert input.ndim == target.ndim >= 1
        return ptnf.mse_loss(input, target, reduction=self.reduce)


class LPIPSLoss:
    """"""

    def __init__(self, net="vgg", reduce="mean"):
        self.lpips = lpips.LPIPS(pretrained=True, net=net, eval_mode=True)
        self.reduce = reduce  # mean, None
        for p in self.lpips.parameters():
            p.requires_grad = False
        self.lpips.compile()
        # self.lpips = pt.quantization.quantize_dynamic(self.lpips)  # slow

    def __call__(self, input, target):
        """
        input: shape=(b,c,h,w), dtype=float
        target: shape=(b,c,h,w), dtype=int64
        """
        assert input.ndim == target.ndim == 4
        self.lpips.to(input.device)  # to the same device, won't repeat once done
        # TODO XXX ??? input.float()
        lpips = self.lpips(target, input).mean([1, 2, 3])  # (b,)
        if self.reduce == "mean":
            return lpips.mean()  # ()
        return lpips  # (b,)


class Accuracy:
    """"""

    def __init__(self, reduce="mean"):
        self.reduce = reduce  # mean, None

    def __call__(self, input, target):
        """
        input: shape=(b,..), dtype=int64
        target: shape=(b,..), dtype=int64
        """
        assert input.shape == target.shape
        acc = (input == target).flatten(1).mean(1)  # (b,)
        if self.reduce == "mean":
            return acc.mean()  # ()
        return acc  # (b,)


class ARI:
    """"""

    def __init__(self, skip=[], reduce="mean"):
        self.skip = pt.from_numpy(np.array(skip, "int64"))
        self.reduce = reduce  # mean, None

    def __call__(self, input, target):
        """
        input: shape=(b,n), dtype=int, index segment
        target: shape(b,n), dtype=int, index segment
        """
        idx_pd, idx_gt = ARI.segment_assert(input, target)  # (b,n)
        oh_pd, oh_gt = ARI.index_to_onehot(idx_pd, idx_gt, self.skip)  # (b,n,c) (b,n,d)

        ari = __class__.adjusted_rand_index(oh_pd, oh_gt)  # (b,)
        valid = ARI.find_valid(oh_gt)  # (b,)
        if self.reduce == "mean":
            return ari[valid].mean()  # (b,) -> (b',) -> ()
        return ari, valid

    @staticmethod
    def adjusted_rand_index(oh_pd, oh_gt):
        """
        https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py

        oh_pd: shape=(b,n,c), dtype=bool, onehot segment
        oh_gt: shape=(b,n,d), dtype=bool, onehot segment
        return: shape=(b,), dtype=float32
        """
        # the following boolean op is even slower than floating point op
        # long, will be auto-cast to float  # int overflow
        N = pt.einsum("bnc,bnd->bcd", oh_pd.double(), oh_gt.double()).long()  # (b,c,d)
        # N = (oh_gt[:, :, :, None] & oh_pd[:, :, None, :]).sum(1)  # (b,d,c)  # oom
        # the following fixed point op shows negligible speedup vesus floating point op
        A = N.sum(1)  # (b,d)
        B = N.sum(2)  # (b,c)
        num = A.sum(1)  # (b,)

        ridx = (N * (N - 1)).sum([1, 2])  # (b,)
        aidx = (A * (A - 1)).sum(1)  # (b,)
        bidx = (B * (B - 1)).sum(1)  # (b,)

        expect_ridx = (aidx * bidx) / (num * (num - 1)).clip(1)
        max_ridx = (aidx + bidx) / 2
        denominat = max_ridx - expect_ridx
        ari = (ridx - expect_ridx) / denominat  # (b,)

        # two cases ``denominat == 0``
        # - both pd and gt assign all pixels to a single cluster
        #    max_ridx == expect_ridx == ridx == num * (num - 1)
        # - both pd and gt assign max 1 point to each cluster
        #    max_ridx == expect_ridx == ridx == 0
        # we want the ARI score to be 1.0
        ari[denominat == 0] = 1
        return ari

    @staticmethod
    def segment_assert(idx_pd, idx_gt):
        """
        idx_pd: shape=(b,n), dtype=int, indexed segment
        idx_gt: shape=(b,n), dtype=int, indexed segment
        """
        assert idx_pd.dtype == idx_gt.dtype
        assert idx_pd.shape == idx_gt.shape
        assert idx_pd.ndim == 2
        return idx_pd, idx_gt

    @staticmethod
    def index_to_onehot(idx_pd, idx_gt, skip=None, skip_empty=True):
        """
        idx_pd: shape=(b,n), dtype=uint8, indexed segment
        idx_gt: shape=(b,n), dtype=uint8, indexed segment
        """
        oh_pd = ptnf.one_hot(idx_pd.long()).bool()  # (b,n,c)
        oh_gt = ptnf.one_hot(idx_gt.long()).bool()  # (b,n,d)
        if skip is not None:
            b, n, c = oh_gt.shape
            mask = ~pt.isin(pt.arange(c), skip)
            oh_gt = oh_gt[:, :, mask]
        # if skip_empty:  # save computation  # TODO XXX ???
        #     oh_pd = oh_pd[:, :, (oh_pd != 0).any([0, 1])]
        #     oh_gt = oh_gt[:, :, (oh_gt != 0).any([0, 1])]
        # TODO XXX batch impl
        """def remap_tensor(x: torch.Tensor) -> torch.Tensor:
            # torch.unique returns the sorted unique values and, if requested,
            # an inverse mapping that tells, for each element in x, its index in uniques.
            _, inv = torch.unique(x, return_inverse=True)
            return inv.reshape(x.shape)

        # Suppose batch is a tensor of shape (B, H, W) with dtype=torch.uint8.
        batch = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)

        # Use vmap to apply remap_tensor to each batch element.
        remap_vmap = torch.vmap(remap_tensor)
        batch_remapped = remap_vmap(batch)
        print(batch_remapped)"""
        return oh_pd, oh_gt  # (b,n,c) (b,n,d)

    @staticmethod
    def find_valid(oh_gt):
        assert oh_gt.ndim == 3
        valid = oh_gt.flatten(1).any(1)
        return valid


class mBO:
    """"""

    def __init__(self, skip=[], reduce="mean"):
        self.skip = pt.from_numpy(np.array(skip, "int64"))
        self.reduce = reduce  # mean, None

    def __call__(self, input, target):
        """
        input: shape=(b,n), dtype=int, index segment
        target: shape(b,n), dtype=int, index segment
        """
        idx_pd, idx_gt = ARI.segment_assert(input, target)  # (b,n)
        oh_pd, oh_gt = ARI.index_to_onehot(idx_pd, idx_gt, self.skip)  # (b,n,c) (b,n,d)

        mbo = __class__.mean_best_overlap(oh_pd, oh_gt)  # (b,)
        valid = ARI.find_valid(oh_gt)  # (b,)
        if self.reduce == "mean":
            return mbo[valid].mean()  # (b,) -> (b',) -> ()
        return mbo, valid

    @staticmethod
    def mean_best_overlap(oh_pd, oh_gt):
        """
        https://github.com/martius-lab/videosaur/blob/main/videosaur/metrics.py

        oh_pd: shape=(b,n,c), dtype=bool, onehot segment
        oh_gt: shape=(b,n,d), dtype=bool, onehot segment
        return: shape=(b,), dtype=float32
        """
        iou_all = intersection_over_union(oh_pd, oh_gt)  # (b,c,d)
        iou, idx = iou_all.max(1)  # (b,d)
        num_gt = oh_gt.any(1).sum(1)  # (b,)
        return iou.sum(1) / num_gt  # (b,)


class mBOAnalyz(mBO):

    def __init__(self, skip=[]):
        super().__init__(skip, None)

    def __call__(self, input, target):
        """
        input: shape=(b,n), dtype=int, index segment
        target: shape(b,n), dtype=int, index segment
        """
        idx_pd, idx_gt = ARI.segment_assert(input, target)  # (b,n)
        oh_pd, oh_gt = ARI.index_to_onehot(idx_pd, idx_gt, self.skip)  # (b,n,c) (b,n,d)

        iou_all = intersection_over_union(oh_pd, oh_gt)  # (b,c,d)
        iou, idx = iou_all.max(2)  # (b,c)
        return iou
        iou, idx = iou_all.max(1)  # (b,d)
        # num_gt = oh_gt.any(1).sum(1)  # (b,)
        # return iou.sum(1) / num_gt  # (b,)
        valid = ARI.find_valid(oh_gt)  # (b,)
        return iou, valid


class mIoU:
    """"""

    def __init__(self, skip=[], reduce="mean"):
        self.skip = pt.from_numpy(np.array(skip, "int64"))
        self.reduce = reduce  # mean, None

    def __call__(self, input, target):
        """
        input: shape=(b,n), dtype=int, index segment
        target: shape(b,n), dtype=int, index segment
        """
        idx_pd, idx_gt = ARI.segment_assert(input, target)  # (b,n)
        oh_pd, oh_gt = ARI.index_to_onehot(idx_pd, idx_gt, self.skip)  # (b,n,c) (b,n,d)

        miou = __class__.mean_intersection_over_union(oh_pd, oh_gt)  # (b,)
        valid = ARI.find_valid(oh_gt)  # (b,)
        if self.reduce == "mean":
            return miou[valid].mean()  # (b,) -> (b',) -> ()
        return miou, valid

    @staticmethod
    def mean_intersection_over_union(oh_pd, oh_gt):
        """
        https://github.com/martius-lab/videosaur/blob/main/videosaur/metrics.py

        oh_pd: shape=(b,n,c), dtype=bool, onehot segment
        oh_gt: shape=(b,n,d), dtype=bool, onehot segment
        return: shape=(b,), dtype=float32
        """
        iou_all = intersection_over_union(oh_pd, oh_gt)  # (b,c,d)
        iou, idx = hungarian_matching(iou_all, maximize=True)  # (b,d)
        num_gt = oh_gt.any(1).sum(1)  # (b,)
        return iou.sum(1) / num_gt  # (b,)
