import re

import torch.amp.grad_scaler as ptags
import torch.nn.utils.clip_grad as ptnucg
import torch.optim as pto


SGD = pto.SGD


Adam = pto.Adam


AdamW = pto.AdamW


NAdam = pto.NAdam


RAdam = pto.RAdam


####


GradScaler = ptags.GradScaler


class ClipGradNorm:
    """"""

    def __init__(self, max_norm, norm_type=2):  # norm_type: 2 > inf
        self.max_norm = max_norm
        self.norm_type = norm_type

    def __call__(self, params):
        return ptnucg.clip_grad_norm_(params, self.max_norm, self.norm_type)


class ClipGradValue:
    """"""

    def __init__(self, max_value):
        self.max_value = max_value

    def __call__(self, params):
        return ptnucg.clip_grad_value_(params, self.max_value)


####


def group_params_by_keys(
    named_parameters, param_group_idxs, compile_prefix="_orig_mod."
):
    """
    named_parameters: ``model.named_parameters()``
    param_groups: [{key: str, lr: float},..]
    """
    named_parameters = list(named_parameters)
    param_groups = []
    for pgi in param_group_idxs:
        param_group = dict(params=[], lr=pgi["lr"])
        for key, param in named_parameters:
            if key.startswith(compile_prefix):
                key = key[len(compile_prefix) :]
            if re.match(pgi["key"], key):
                param_group["params"].append(param)
        param_groups.append(param_group)
    assert len(named_parameters) == sum(len(_["params"]) for _ in param_groups)
    return param_groups
