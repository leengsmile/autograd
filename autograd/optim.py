
from autograd.module import Module
class Optimizer:
    pass


class SGD(Optimizer):

    def __init__(self, lr: float = 0.01) -> None:
        super().__init__()
        self.lr = lr
    
    def step(self, module: Module) -> None:
        for parameter in module.parameters():
            parameter -= parameter.grad * self.lr
