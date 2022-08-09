import torch

from pypc import utils
from pypc.layers import FCLayer


class PCModel(object):
    def __init__(self, nodes, mu_dt, act_fn, use_bias=False, kaiming_init=False):
        """
        Define the Predictive Coding PyTorch model. All layers fully connected. All nodes using the specified activation
        function except for the output layer which is linear. Bias terms are optional. Kaiming weight initialisation is
        optional.

        :param nodes: List of number of nodes in each layer
        :param mu_dt:
        :param act_fn: Activation function
        :param use_bias: Include bias terms?
        :param kaiming_init: Use Kaiming weight initialisation?
        """
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
        """
        Initialise predictions (preds), errors (errs), and variational means (mus) to empty lists
        """
        self.preds = [[] for _ in range(self.n_nodes)]
        self.errs = [[] for _ in range(self.n_nodes)]
        self.mus = [[] for _ in range(self.n_nodes)]

    def reset_mus(self, batch_size, init_std):
        for l in range(self.n_layers):
            self.mus[l] = utils.set_tensor(
                torch.empty(batch_size, self.layers[l].in_size).normal_(mean=0, std=init_std)
            )

    def set_input(self, inp):
        """
        Set input node mus for batch to input values

        :param inp: Input values, Tensor:(batch_size, nodes[0])
        """
        self.mus[0] = inp.clone()

    def set_target(self, target):
        """
        Set output node mus for batch to target values

        :param target: Target values, Tensor:(batch_size, nodes[-1])
        """
        self.mus[-1] = target.clone()

    def forward(self, val):
        for layer in self.layers:
            val = layer.forward(val)
        return val

    def propagate_mu(self):
        """
        Perform forward pass for batch, update mus for all nodes except inputs and outputs (targets)
        """
        for l in range(1, self.n_layers):
            self.mus[l] = self.layers[l - 1].forward(self.mus[l - 1])

    def train_batch_supervised(self, img_batch, label_batch, n_iters, fixed_preds=False):
        """
        Train the model using the (mini)batch images as inputs and labels as targets

        :param img_batch: Batch of input images, Tensor:(batch_size, nodes[0])
        :param label_batch: Batch of target labels, Tensor:(batch_size, nodes[-1])
        :param n_iters: Number of training iterations
        :param fixed_preds: Fix predictions at initial values?
        """
        self.reset()  # Initialise the prediction, error, and mu data structures
        self.set_input(img_batch)  # Set the model inputs, mus[0], equal to the training *images*
        self.propagate_mu()  # Perform forward pass, update mus for all nodes except inputs and targets
        self.set_target(label_batch)  # Set the model outputs (targets), mus[-1], equal to the training *labels*
        self.train_updates(n_iters, fixed_preds=fixed_preds)  # Iteratively update mus, predictions and errors
        self.update_grads()  # Calculate gradients of weights and biases for all layers

    def train_batch_generative(self, img_batch, label_batch, n_iters, fixed_preds=False):
        """
        Train the model using the (mini)batch labels as inputs and images as targets
        (Identical to train_batch_supervised() but inputs and targets are swapped)

        :param img_batch: Batch of target images, Tensor:(batch_size, nodes[-1])
        :param label_batch: Batch of input labels, Tensor:(batch_size, nodes[0])
        :param n_iters: Number of training iterations
        :param fixed_preds: Fix predictions at initial values?
        """
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
        """
        Iteratively update mus, predictions and errors

        :param n_iters: Number of training iterations
        :param fixed_preds: Fix predictions at initial values?
        """
        # For batch, initialise predictions and errors for all nodes except inputs
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]

        # For each training iteration
        for itr in range(n_iters):
            # For batch, update mus for all nodes except inputs and outputs
            for l in range(1, self.n_layers):
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            # For batch, update errors and (optionally) predictions for all nodes except inputs
            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]

    def test_updates(self, n_iters, fixed_preds):
        """
        Test model

        :param n_iters: Number of training iterations
        :param fixed_preds: Fix predictions at initial values?
        """
        # For batch, initialise predictions and errors for all nodes except inputs
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]

        # For each test iteration
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
        """
        Calculate gradients of weights and biases for all layers

        """
        for l in range(self.n_layers):
            self.layers[l].update_gradient(self.errs[l + 1])

    def get_target_loss(self):
        """
        Calculate loss as the sum of the squares of the target errors
        (Not currently used)

        :return: Loss
        """
        return torch.sum(self.errs[-1] ** 2).item()

    @property
    def params(self):
        """
        Allows controlled and standardised access to model parameters (for passing to an optimizer).
        Currently of limited use but could be expanded.

        :return: Model layers
        """
        return self.layers

