import inspect

from autograd.parameter import Parameter

class Module:
    def parameters(self):
        for name, value in inspect.getmembers(self):

            if isinstance(value, Parameter):
                yield value
            elif  isinstance(value, Module):
                yield from value.parameters()
        
    def zero_grad(self,):

        for parameter in self.parameters():
            parameter.zero_grad()


