import numpy as np
import torch

from pypc import utils
from pypc.layers import FCLayer


class PCModel(object):
    def __init__(self, nodes, mu_dt, act_fn, use_bias=False, kaiming_init=False):
        self.nodes = nodes
        self.mu_dt = mu_dt

        self.n_nodes = len(nodes)
        self.n_layers = len(nodes) - 1

        self.layers = []
        for l in range(self.n_layers):
            _act_fn = utils.Linear() if (l == self.n_layers - 1) else act_fn

            layer = FCLayer(
                in_size=nodes[l],
                out_size=nodes[l + 1],
                act_fn=_act_fn,
                use_bias=use_bias,
                kaiming_init=kaiming_init,
            )
            self.layers.append(layer)

    def reset(self):
        self.preds = [[] for _ in range(self.n_nodes)]
        self.errs = [[] for _ in range(self.n_nodes)]
        self.mus = [[] for _ in range(self.n_nodes)]

    def reset_mus(self, batch_size, init_std):
        for l in range(self.n_layers):
            self.mus[l] = utils.set_tensor(
                torch.empty(batch_size, self.layers[l].in_size).normal_(mean=0, std=init_std)
            )

    def set_input(self, inp):
        self.mus[0] = inp.clone()

    def set_target(self, target):
        self.mus[-1] = target.clone()

    def forward(self, val):
        for layer in self.layers:
            val = layer.forward(val)
        return val

    def propagate_mu(self):
        for l in range(1, self.n_layers):
            self.mus[l] = self.layers[l - 1].forward(self.mus[l - 1])

    def train_batch_supervised(self, img_batch, label_batch, n_iters, fixed_preds=False):
        self.reset()
        self.set_input(img_batch)
        self.propagate_mu()
        self.set_target(label_batch)
        self.train_updates(n_iters, fixed_preds=fixed_preds)
        self.update_grads()

    def train_batch_generative(self, img_batch, label_batch, n_iters, fixed_preds=False):
        self.reset()
        self.set_input(label_batch)
        self.propagate_mu()
        self.set_target(img_batch)
        self.train_updates(n_iters, fixed_preds=fixed_preds)
        self.update_grads()

    def test_batch_supervised(self, img_batch):
        return self.forward(img_batch)

    def test_batch_generative(self, img_batch, n_iters, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.test_updates(n_iters, fixed_preds=fixed_preds)
        return self.mus[0]

    def train_updates(self, n_iters, fixed_preds=False):
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]

        for itr in range(n_iters):
            for l in range(1, self.n_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]

    def test_updates(self, n_iters, fixed_preds):
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]

        for itr in range(n_iters):
            delta = self.layers[0].backward(self.errs[1])
            self.mus[0] = self.mus[0] + self.mu_dt * delta
            for l in range(1, self.n_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]

    def update_grads(self):
        for l in range(self.n_layers):
            self.layers[l].update_gradient(self.errs[l + 1])

    def get_target_loss(self):
        return torch.sum(self.errs[-1] ** 2).item()

    @property
    def params(self):
        return self.layers

