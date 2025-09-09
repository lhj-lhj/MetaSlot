from abc import ABC, abstractmethod
from bisect import bisect
from numbers import Number

import numpy as np


class Schedule(ABC):
    """Schedule Everything as You Want.

    Supported cases:
    - model.alpha.fill_(value)
    - model.alpha = value
    - model.tau.data[...] = value
    - optim.param_groups[0]["lr"] = value
    - loss["recon_d"]["weight"] = value
    """

    @abstractmethod
    def __init__(self, assigns, step_count_key="step_count"):
        assert all("value" in _ for _ in assigns)
        self.assigns = assigns
        self.step_count_key = step_count_key
        self.sched = ...

    def __call__(self, **pack: dict) -> dict:
        step_count = pack[self.step_count_key]
        for k in pack.keys():  # extract all global values
            exec(f"{k} = pack['{k}']")
        for assign in self.assigns:
            value = self[step_count]
            exec(assign)  # ``value`` is executed in ``assign`` string
        return pack

    def __getitem__(self, idx):
        return self.sched[idx]

    def __len__(self):
        return len(self.sched)


class CbLinear(Schedule):

    def __init__(self, assigns, ntotal, vbase, vfinal=0):
        super().__init__(assigns)
        self.sched = [  # torch invokes lr_sched ntotal+1 times
            __class__.linear(_, ntotal, vbase, vfinal) for _ in range(ntotal + 1)
        ]

    @staticmethod
    def linear(n, ntotal, vbase, vfinal):
        return (vfinal - vbase) / ntotal * n + vbase


class CbCosine(Schedule):

    def __init__(self, assigns, ntotal, vbase, vfinal=0):
        super().__init__(assigns)
        self.sched = [
            __class__.cosine(_, ntotal, vbase, vfinal) for _ in range(ntotal + 1)
        ]

    @staticmethod
    def cosine(n, ntotal, vbase, vfinal):
        return 0.5 * (vbase - vfinal) * (1 + np.cos(np.pi * n / ntotal)) + vfinal


class CbLnCosine(CbCosine):
    """
    ```
    import matplotlib.pyplot as plt
    import numpy as np
    import torch as pt

    def unchanged_selection_probability(x, g, t):
        i0 = x.argmax(-1)
        o = (x + g) / t
        i1 = o.argmax(-1)
        return (i0 == i1).float().mean().item()

    x = pt.randn(64 * 16 * 16, 4096).float()  # .cuda()  # ~N(0, 1)
    g = (  # gumbel noise
        -pt.empty_like(x, memory_format=pt.legacy_contiguous_format).exponential_().log()
    )

    fig, axs = plt.subplots(2, 2)

    s = np.exp(np.linspace(np.log(0.1), np.log(100), num=100))  # ``std``
    ps = [unchanged_selection_probability(x * _, g, t=1.0) for _ in s]
    axs[0, 0].scatter(
        np.log(s), ps, marker=".", label="std vs unchanged_selection_probability"
    )  # P@s0.1=0.01%, 1.0:0.6%, 10:74%, 100:97.5%

    t = np.exp(np.linspace(np.log(0.001), np.log(10), num=100))  # ``tau``
    pp = [unchanged_selection_probability(x * 10, g, _) for _ in t]
    axs[0, 1].plot(s, pp, label="tau vs unchanged_selection_probability")  # not change

    def cosine(n, ntotal, vbase, vfinal):
        return 0.5 * (vbase - vfinal) * (1 + np.cos(np.pi * n / ntotal)) + vfinal

    vbase = 1
    vfinal = 10
    # ii = [cosine(_, 100, vbase, vfinal) for _ in range(100)]
    ii = [cosine(_, 100, np.log(vbase), np.log(vfinal)) for _ in range(100)]
    axs[1, 0].plot(ii, label="xs")
    # ys = [unchanged_selection_probability(x * _, g, t=1) for _ in xs]
    ys = [unchanged_selection_probability(x * np.exp(_), g, t=1) for _ in ii]
    axs[1, 1].plot(ys, label="xs vs unchanged_selection_probability")
    [_.legend() for _ in axs.flatten()]
    plt.show()
    ```
    """

    def __init__(self, assigns, ntotal, vbase, vfinal=0):
        super().__init__(assigns, ntotal, vbase, vfinal)
        self.sched = [
            np.exp(__class__.cosine(_, ntotal, np.log(vbase), np.log(vfinal)))
            for _ in range(ntotal + 1)
        ]


class CbCosineLinear(Schedule):

    def __init__(self, assigns, ncos, ntotal, vbase, vmid, vfinal):
        super().__init__(assigns)
        nlin = ntotal - ncos
        self.sched = [
            CbCosine.cosine(_, ncos, vbase, vmid) for _ in range(ncos + 1)
        ] + [CbLinear.linear(_, nlin, vmid, vfinal) for _ in range(1, nlin + 1)]


class CbLinearCosine(Schedule):

    def __init__(self, assigns, nlin, ntotal, vstart, vbase, vfinal=0):
        super().__init__(assigns)
        ncos = ntotal - nlin
        self.sched = [
            CbLinear.linear(_, nlin, vstart, vbase) for _ in range(nlin + 1)
        ] + [CbCosine.cosine(_, ncos, vbase, vfinal) for _ in range(1, ncos + 1)]


class CbSquarewave(Schedule):
    """
    e.g., points=[0,500,1000] and values=[1,0] means that value is 1 before step 500 while value is 0 after step 500
    """

    def __init__(self, assigns, points: list, values: list):
        super().__init__(assigns)
        assert len(values) + 1 == len(points)
        assert all(isinstance(_, Number) for _ in points)  # not nested
        # assert all(isinstance(_, Number) for _ in values)  # not nested
        self.sched = [
            __class__.squarewave(_, points, values) for _ in range(points[-1])
        ] + [values[-1]]

    @staticmethod
    def squarewave(n, points, values):
        return values[bisect(points, n) - 1]
