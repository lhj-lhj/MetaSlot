import torch.nn.functional as ptnf

from object_centric_bench.learn.utils import intersection_over_union, hungarian_matching


class AttentMatchLoss:
    """
    SPOT's attention matching loss: Hungarian matching and cross entropy.
    """

    def __init__(self, reduce="mean"):
        self.reduce = reduce

    def __call__(self, input, target):
        """
        - input, target: attention maps, same shape=(b,c,..), float
        """
        input = input.flatten(2)  # (b,c,x)
        target = target.flatten(2)  # (b,d,x)
        b, c, x = input.shape
        b, d, x = target.shape

        # match
        oh_pd = ptnf.one_hot(input.argmax(1), c)  # (b,x,c)
        oh_gt = ptnf.one_hot(target.argmax(1), d)  # (b,x,d)
        iou_all = intersection_over_union(oh_pd, oh_gt)  # (b,c,d)
        iou, rcidx = hungarian_matching(iou_all, maximize=True)  # (b,d) (b,2,d)

        # ground-truth
        # (b,x,d) (b,x,d) -> (b,x,d) -> (b,x)
        idx_gt = oh_gt.gather(2, rcidx[:, 1:, :].expand(-1, x, -1)).argmax(2)

        # loss
        return ptnf.cross_entropy(input, idx_gt, reduction=self.reduce)
