"""
Simple Tensor
"""

from typing import Callable, List, Union, Optional, NamedTuple
import numpy as np


class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]

Arrayable = Union[float, list, np.ndarray]


def ensure_arrayable(arrayable: Arrayable) -> np.ndarray:

    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


class Tensor:

    def __init__(self, 
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None, ) -> None:

        self.data: np.ndarray = ensure_arrayable(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []

        self.shape = self.data.shape

        self.grad: Optional['Tensor'] = None

        if requires_grad:
            self.zero_grad()


    def __repr__(self,) -> str:

        return f"Tensor(self.data={self.data}, requires_grad={self.requires_grad})"

    def zero_grad(self):
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad: 'Tensor' = None) -> None:
        if grad is None:
            if self.data.shape == (): 
                grad = Tensor(1.)
            else:
                raise RuntimeError("grad should be specified for non-zero tensor.")
        
        self.grad.data = grad.data

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad))


    def sum(self) -> 'Tensor':
        return tensor_sum(self)


def tensor_sum(t: Tensor) -> Tensor:

    data = t.data.sum()
    requires_grad = t.requires_grad
    
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            
            return np.ones_like(t.data) * grad

        depends_on = [Dependency(t, grad_fn)]
    else:
        depends_on = []

    return Tensor(data, requires_grad, depends_on)