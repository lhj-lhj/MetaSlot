from ..utils import Compose


class Callback:

    def __init__(
        self,
        before_train: list = None,
        before_epoch: list = None,
        before_step: list = None,
        after_forward: list = None,
        after_step: list = None,
        after_epoch: list = None,
        after_train: list = None,
    ):
        if before_train:
            self.before_train = Compose(before_train)
        if before_epoch:
            self.before_epoch = Compose(before_epoch)
        if before_step:
            self.before_step = Compose(before_step)
        if after_forward:
            self.after_forward = Compose(after_forward)
        if after_step:
            self.after_step = Compose(after_step)
        if after_epoch:
            self.after_epoch = Compose(after_epoch)
        if after_train:
            self.after_train = Compose(after_train)

    def before_train(self, **pack):
        return pack

    def before_epoch(self, **pack):
        return pack

    def before_step(self, **pack):
        return pack

    def after_forward(self, **pack):
        return pack

    def after_step(self, **pack):
        return pack

    def after_epoch(self, **pack):
        return pack

    def after_train(self, **pack):
        return pack

    # @staticmethod
    # def compose(funcs):
    #     def func(**kwds):
    #         for _ in funcs:
    #             _(**kwds)  # ``kwds``: modified inplace
    #         return kwds

    #     return func
