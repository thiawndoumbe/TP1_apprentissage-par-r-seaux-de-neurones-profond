import abc
from typing import List


from scratch_grad.variable import Variable


class Optimizer(abc.ABC):
    def __init__(self, params: List[Variable]):
        self.params = params

    @abc.abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()


class SGD(Optimizer):
    def __init__(self, params: List[Variable], lr: float, weight_decay: float = 0.0, momentum: float = 0.0,
                 dampening: float = 0.0, nesterov: bool = False):
        """
        Stochastic Gradient Descent (https://pytorch.org/docs/main/generated/torch.optim.SGD.html#sgd)
        :param params: variables to optimize
        :param lr: learning rate
        :param weight_decay: weight decay
        :param momentum: momentum factor
        :param dampening: dampening for momentum
        :param nesterov: enables Nesterov momentum
        """
        super().__init__(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        # TODO Ajouter les attributs nécessaires
        self.velocities = {id(param): [0.0] * len(param.data) for param in self.params}



    def step(self):
        # TODO Implémenter la mise à jour des paramètres
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad
            if self.weight_decay > 0:
                grad += self.weight_decay * param.data
            velocity = self.velocities[id(param)]
            if self.momentum > 0:
                velocity *= self.momentum
                velocity += (1 - self.dampening) * grad
                if self.nesterov:
                     update = grad + self.momentum * velocity
                else:
                    update = velocity
            else:
                update = grad

            param.data -= self.lr * update
            self.velocities[id(param)] = velocity


class Adam(Optimizer):
    def __init__(self, params: List[Variable], lr: float, betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, amsgrad: bool = False):
        """
        Adam (https://pytorch.org/docs/main/generated/torch.optim.Adam.html#adam)
        :param params: variables to optimize
        :param lr: learning rate
        :param betas: coefficients used for computing running averages of gradient and its square
        :param eps: term added to the denominator to improve numerical stability
        :param weight_decay: weight decay
        :param amsgrad: whether to use the AMSGrad variant of this algorithm
        """
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        # TODO Ajouter les attributs nécessaires
        self.m = {id(param): [0.0] * len(param.data) for param in self.params}
        self.v = {id(param): [0.0] * len(param.data) for param in self.params}
        self.v_hat = {id(param): [0.0] * len(param.data) for param in self.params} if amsgrad else None
        self.t = 0

    def step(self):
        # TODO Implémenter la mise à jour des paramètres
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad
            if self.weight_decay > 0:
                grad += self.weight_decay * param.data
            
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad * grad)

            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)

            if self.amsgrad:
                self.v_hat[param] = max(self.v_hat[param], v_hat)
                v_hat = self.v_hat[param]

            param.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)
