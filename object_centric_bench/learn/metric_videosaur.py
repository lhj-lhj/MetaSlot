import torch as pt
import torch.nn.functional as ptnf


class FeatureTimeSimilarity:

    def __init__(self, time_shift=1, thresh=None, tau=1.0, softmax=True):
        super().__init__()
        self.time_shift = time_shift
        self.thresh = thresh
        self.tau = tau
        self.softmax = softmax

    def __call__(self, feature):
        """
        - feature: shape=(b,t,n,c)
        """
        assert feature.ndim == 4
        b, t, n, c = feature.shape

        feature = ptnf.normalize(feature, 2, -1)
        source = feature[:, : -self.time_shift, :, :]
        target = feature[:, self.time_shift :, :, :]

        similarity = __class__.cross_similarity(
            source.flatten(0, 1),
            target.flatten(0, 1),
            self.thresh,
            self.tau,
            self.softmax,
        ).unflatten(0, [b, t - self.time_shift])

        return similarity

    @staticmethod
    @pt.no_grad()
    def cross_similarity(source, target, thresh=None, tau=1.0, softmax=True):
        """
        - source: shape=(b,m,c)
        - target: shape=(b,n,c)
        """
        product = pt.einsum("bmc,bnc->bmn", source, target)
        b, m, n = product.shape
        if thresh is not None:
            product[product < thresh] = -pt.inf
        product /= tau
        if softmax:
            flag = product.isinf().all(-1, keepdim=True).expand(-1, -1, n)  # -inf
            product[flag] = 0.0
            product = product.softmax(-1)
        return product
