from pathlib import Path
import json
import pickle as pkl

import numpy as np
import torch as pt

from .callback import Callback
from ..utils import DictTool


def tonumpy(data):
    """recursively convert data into numpy"""
    if isinstance(data, (list, tuple)):
        data = [tonumpy(_) for _ in data]
    elif isinstance(data, dict):
        data = {k: tonumpy(v) for k, v in data.items()}
    elif isinstance(data, pt.Tensor):
        data = data.detach().cpu().numpy()
    else:
        raise NotImplementedError
    return data


class AverageLog(Callback):
    """"""

    def __init__(
        self,
        log_file=None,
        epoch_key="epoch",
        model_key="model",
        loss_key="loss",
        metric_key="metric",
    ):
        self.log_file = log_file
        self.epoch_key = epoch_key
        self.model_key = model_key
        self.loss_key = loss_key
        self.metric_key = metric_key
        self.idx = None
        self.current_dict = {}

    def index(self, epoch, model):
        self.idx = f"{epoch}" if model.training else f"{epoch}/val"
        self.current_dict.clear()

    def append(self, loss, metric):  # TODO variant batch size TODO
        for key, value in {**loss, **metric}.items():
            value = value.detach().cpu().numpy()
            if key in self.current_dict:
                self.current_dict[key].append(value)
            else:
                self.current_dict[key] = [value]

    def mean(self):  # TODO variant batch size TODO
        avg_dict = {}
        for k, v in self.current_dict.items():
            val = np.array(v).mean(0)  # .round(10) not work ???
            avg_dict[k] = val.tolist()
        if self.log_file:
            __class__.save(self.idx, avg_dict, self.log_file)
        print(self.idx, avg_dict)
        return avg_dict

    @staticmethod
    def save(key, avg_dict, log_file):
        line = json.dumps({key: avg_dict})
        with open(log_file, "a") as f:
            f.write(line + "\n")

    def before_epoch(self, **pack):
        epoch = pack[self.epoch_key]
        model = pack[self.model_key]
        self.index(epoch, model)
        return pack

    def after_step(self, **pack):
        loss = pack[self.loss_key]
        metric = pack[self.metric_key]
        self.append(loss, metric)
        return pack

    def after_epoch(self, **pack):
        self.mean()
        return pack


class CollectLog(AverageLog):  # not tested

    def __init__(
        self,
        dump_file="collectlog.pkl",
        epoch_key="epoch",
        model_key="model",
        loss_key="loss",
        metric_key="metric",
    ):
        super().__init__(
            epoch_key=epoch_key,
            model_key=model_key,
            loss_key=loss_key,
            metric_key=metric_key,
        )
        del self.log_file
        self.dump_file = dump_file

    def append(self, loss, metric):
        for key, value in {**loss, **metric}.items():
            value = __class__.tonumpy(value)
            if key in self.current_dict:
                self.current_dict[key].append(value)
            else:
                self.current_dict[key] = [value]

    def mean(self):  # disable this
        raise NotImplementedError

    def save(self):
        with open(self.dump_file, "wb") as f:
            pkl.dump(self.current_dict, f)

    def after_epoch(self, **pack):
        self.save()
        return pack


class SaveModel(Callback):
    """"""

    def __init__(
        self,
        save_dir=None,
        since_step=0,
        weights_only=True,
        key=r".*",
        epoch_key="epoch",
        step_count_key="step_count",
        model_key="model",
    ):
        self.save_dir = save_dir
        self.since_step = since_step  # self.after_step is taken
        self.weights_only = weights_only
        self.key = key
        self.epoch_key = epoch_key
        self.step_count_key = step_count_key
        self.model_key = model_key

    def __call__(self, epoch, step_count, model):
        if step_count >= self.since_step:
            save_file = Path(self.save_dir) / f"{epoch:04d}.pth"
            model.save(save_file, self.weights_only, self.key)

    def after_epoch(self, **pack):
        epoch = pack[self.epoch_key]
        step_count = pack[self.step_count_key]
        model = pack[self.model_key]
        self.__call__(epoch, step_count, model)
        return pack
