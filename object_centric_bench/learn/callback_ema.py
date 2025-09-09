from copy import deepcopy
import re

import torch as pt

from .callback import Callback


class DINOEMA(Callback):

    def __init__(
        self,
        source=None,  # from key patterns
        target=None,  # to key patterns
        momentum=0.0001,
        # interval=1,
        # interval_swap=0,
        # warm_up=0,
        model_key="model",
        step_count_key="step_count",
    ):
        self.source = source
        self.target = target
        # assert isinstance(interval, int) and interval > 0
        # self.warm_up = warm_up
        # self.interval = interval
        # if interval_swap:
        #     assert interval_swap % interval == 0
        # self.interval_swap = interval_swap  # 0: no swap
        assert momentum > 0 and momentum < 1
        self.momentum = momentum  # **interval
        self.model_key = model_key
        self.step_count_key = step_count_key

    # @pt.no_grad()
    # def duplica(self, model):
    #     if self.target is None:
    #         self.model_ema = deepcopy(model)

    @pt.no_grad()
    def update(self, model, step_count):
        # if step_count < self.warm_up or step_count % self.interval != 0:
        #     return
        # state_dict_ema = self.model_ema.state_dict()
        # state_dict = model.state_dict()
        # assert list(state_dict_ema.keys()) == list(state_dict.keys())
        # for param_ema, param in zip(state_dict_ema.values(), state_dict.values()):
        #     param_ema.mul_(1 - self.momentum).add_(param.data, alpha=self.momentum)
        if not hasattr(self, "s_kvs"):
            state_dict = model.state_dict()
            self.s_kvs = [_ for _ in state_dict.items() if re.match(self.source, _[0])]
            self.t_kvs = [_ for _ in state_dict.items() if re.match(self.target, _[0])]
            # assert all(sk == tk for (sk, _), (tk, _) in zip(self.s_kvs, self.t_kvs))
        for (_, sv), (_, tv) in zip(self.s_kvs, self.t_kvs):
            tv.mul_(1 - self.momentum).add_(sv, alpha=self.momentum)

    # @pt.no_grad()
    # def swap(self, model, step_count):
    #     if step_count < self.warm_up or step_count % self.interval_swap != 0:
    #         return
    #     state_dict_ema = self.model_ema.state_dict()
    #     state_dict = model.state_dict()
    #     assert list(state_dict_ema.keys()) == list(state_dict.keys())
    #     for param_ema, param in zip(state_dict_ema.values(), state_dict.values()):
    #         temp = param.data.clone()
    #         param.data.copy_(param_ema.data)
    #         param_ema.data.copy_(temp.data)

    # before_train = duplica

    def after_step(self, **pack):
        model = pack[self.model_key]
        step_count = pack[self.step_count_key]
        self.update(model, step_count)
        # self.swap(model, step_count)
        return pack
