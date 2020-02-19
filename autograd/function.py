import numpy as np
from typing import List
from autograd.tensor import Tensor, Dependency


def tanh(t: Tensor) -> Tensor:

    data = np.tanh(t.data)
    requires_grad = t.requires_grad

    depends_on: List[Dependency] = []
    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - data * data)
        
        depends_on.append(Dependency(t, grad_fn))
    
    return Tensor(data, requires_grad, depends_on)