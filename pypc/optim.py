import numpy as np
import torch


def get_optim(params, optim_id, lr, q_lr=None, batch_scale=True, grad_clip=None, weight_decay=None):
    if optim_id is "Adam":
        return Adam(
            params, lr=lr, q_lr=q_lr, batch_scale=batch_scale, grad_clip=grad_clip, weight_decay=weight_decay
        )
    elif optim_id is "SGD":
        return SGD(
            params, lr=lr, q_lr=q_lr, batch_scale=batch_scale, grad_clip=grad_clip, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"{optim_id} not a valid optimizer ID")


class Optimizer(object):
    def __init__(self, params, batch_scale=True, grad_clip=None, weight_decay=None):
        self._params = params
        self.n_params = len(params)
        self.batch_scale = batch_scale
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay

    def scale_batch(self, param, batch_size):
        if self.batch_scale:
            param.grad["weights"] = (1 / batch_size) * param.grad["weights"]
            if param.use_bias:
                param.grad["bias"] = (1 / batch_size) * param.grad["bias"]

    def clip_grads(self, param):
        if self.grad_clip is not None:
            param.grad["weights"] = torch.clamp(param.grad["weights"], -self.grad_clip, self.grad_clip)
            if param.use_bias:
                param.grad["bias"] = torch.clamp(param.grad["bias"], -self.grad_clip, self.grad_clip)

    def decay_weights(self, param):
        if self.weight_decay is not None:
            param.grad["weights"] = param.grad["weights"] - self.weight_decay * param.weights

    def step(self, *args, **kwargs):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr, q_lr=None, batch_scale=True, grad_clip=None, weight_decay=None):
        super().__init__(params, batch_scale=batch_scale, grad_clip=grad_clip, weight_decay=weight_decay)
        self.lr = lr
        self.q_lr = q_lr

    def step(self, *args, batch_size=None, **kwargs):
        for param in self._params:
            _lr = self.q_lr if param.is_forward else self.lr
            self.scale_batch(param, batch_size)
            self.clip_grads(param)
            self.decay_weights(param)

            param.weights += _lr * param.grad["weights"]
            if param.use_bias:
                param.bias += _lr * param.grad["bias"]
            param._reset_grad()


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr,
        q_lr=None,
        batch_scale=True,
        eps=1e-8,
        beta_1=0.9,
        beta_2=0.999,
        weight_decay=None,
        grad_clip=None,
    ):
        super().__init__(params, batch_scale=batch_scale, grad_clip=grad_clip, weight_decay=weight_decay)
        self.lr = lr
        self.q_lr = q_lr
        self.eps = eps
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        self.c_b = [torch.zeros_like(param.bias) for param in self._params]
        self.c_w = [torch.zeros_like(param.weights) for param in self._params]
        self.v_b = [torch.zeros_like(param.bias) for param in self._params]
        self.v_w = [torch.zeros_like(param.weights) for param in self._params]

    def step(self, curr_epoch=None, curr_batch=None, n_batches=None, batch_size=None):
        with torch.no_grad():
            t = (curr_epoch) * n_batches + curr_batch

            for p, param in enumerate(self._params):
                _lr = self.q_lr if param.is_forward else self.lr
                self.scale_batch(param, batch_size)
                self.clip_grads(param)
                self.decay_weights(param)

                self.c_w[p] = self.beta_1 * self.c_w[p] + (1 - self.beta_1) * param.grad["weights"]
                self.v_w[p] = self.beta_2 * self.v_w[p] + (1 - self.beta_2) * param.grad["weights"] ** 2
                delta_w = np.sqrt(1 - self.beta_2 ** t) * self.c_w[p] / (torch.sqrt(self.v_w[p]) + self.eps)
                param.weights += _lr * delta_w

                if param.use_bias:
                    self.c_b[p] = self.beta_1 * self.c_b[p] + (1 - self.beta_1) * param.grad["bias"]
                    self.v_b[p] = self.beta_2 * self.v_b[p] + (1 - self.beta_2) * param.grad["bias"] ** 2
                    delta_b = (
                        np.sqrt(1 - self.beta_2 ** t) * self.c_b[p] / (torch.sqrt(self.v_b[p]) + self.eps)
                    )
                    param.bias += _lr * delta_b

                param._reset_grad()

