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


Tensorable = Union['Tensor', float, np.ndarray]


def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)

class Tensor:

    def __init__(self, 
                 data: Arrayable,
                 requires_grad: bool = False,
                 depends_on: List[Dependency] = None, ) -> None:

        self._data: np.ndarray = ensure_arrayable(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []

        self.shape = self._data.shape

        self.grad: Optional['Tensor'] = None

        if requires_grad:
            self.zero_grad()

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        self.grad = None

    def __repr__(self,) -> str:

        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other: Tensorable) -> 'Tensor':
        return _add(self, ensure_tensor(other))
    
    def __radd__(self, other: Tensorable) -> 'Tensor':
        return _add(ensure_tensor(other), self)

    def __iadd__(self, other: Tensorable) -> 'Tensor':
        self.data = self.data + ensure_tensor(other).data
        return self
    
    def __isub__(self, other: Tensorable) -> 'Tensor':
        self.data = self.data - ensure_tensor(other).data
        return self

    def __imul__(self, other: Tensorable) -> 'Tensor':
        self.data = self.data * ensure_tensor(other).data
        return self

    def __mul__(self, other: Tensorable) -> 'Tensor':
        return _mul(self, ensure_tensor(other))

    def __rmul__(self, other: Tensorable) -> 'Tensor':
        return _mul(ensure_tensor(other), self)

    def __neg__(self,):
        return _neg(self)
    
    def __sub__(self, other: Tensorable) -> 'Tensor':
        return _sub(self, ensure_tensor(other))

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        return _matmul(self, other)
    
    def __getitem__(self, idx) -> 'Tensor':
        return _slice(self, idx)

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad: 'Tensor' = None) -> None:
        assert self.requires_grad, "called backward on non-requires-grad tensor"

        if grad is None:
            if self.data.shape == (): 
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad should be specified for non-zero tensor.")
        
        self.grad.data = self.grad.data + grad.data  # type: ignore
        # self.grad.data += grad.data  # type: ignore

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

def _add(t1: Tensor, t2: Tensor) -> Tensor:

    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []
    if t1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            ndim_added = grad.ndim - t1.data.ndim
            for _ in range(ndim_added):
                grad = grad.sum(axis=0)
            return grad

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            ndim_added = grad.ndim - t2.data.ndim
            for _ in range(ndim_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)


def _mul(t1: Tensor, t2: Tensor) -> Tensor:

    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []
    if t1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data
            ndim_added = grad.ndim - t1.data.ndim
            for _ in range(ndim_added):
                grad = grad.sum(axis=0)
            return grad

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data
            ndim_added = grad.ndim - t2.data.ndim
            for _ in range(ndim_added):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)    


def _neg(t: Tensor) -> Tensor:
    data = -t.data    
    requires_grad = t.requires_grad
    depends_on: List[Dependency] = []
    if requires_grad:
        depends_on.append(Dependency(t, lambda x: -x))
    else:
        depends_on = []
    return Tensor(data, requires_grad, depends_on)


def _sub(t1: Tensor, t2: Tensor) -> Tensor:
    return t1 + -t2


def _matmul(t1: Tensor, t2: Tensor) -> Tensor:

    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on: List[Dependency] = []
    if t1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:            
            return grad @ t2.data.T

        depends_on.append(Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad

        depends_on.append(Dependency(t2, grad_fn2))

    return Tensor(data, requires_grad, depends_on)    


def _slice(t: Tensor, idx):
    data = t.data[idx]
    requires_grad = t.requires_grad

    dependens_on: List[Dependency] = []
    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:

            bigger_grad = np.zeros_like(data)
            bigger_grad[idx] = grad
            return bigger_grad
        dependens_on.append(Dependency(t, grad_fn))
    
    return Tensor(data, requires_grad, dependens_on)