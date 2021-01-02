import math
import torch
import numpy as np
from copy import deepcopy
from torch import nn
import torch.nn.functional as F
from pypc import utils


class Layer(nn.Module):
    def __init__(
        self, in_size, out_size, act_fn, use_bias=False, kaiming_init=False, is_forward=False
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.act_fn = act_fn
        self.use_bias = use_bias
        self.is_forward = is_forward
        self.kaiming_init = kaiming_init

        self.weights = None
        self.bias = None
        self.grad = {"weights": None, "bias": None}

        if kaiming_init:
            self._reset_params_kaiming()
        else:
            self._reset_params()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def reset(self):
        if self.kaiming_init:
            self._reset_params_kaiming()
        else:
            self._reset_params()

    def _reset_grad(self):
        self.grad = {"weights": None, "bias": None}

    def _reset_params(self):
        weights = torch.empty((self.in_size, self.out_size)).normal_(mean=0.0, std=0.05)
        bias = torch.zeros((self.out_size))
        self.weights = utils.set_tensor(weights)
        self.bias = utils.set_tensor(bias)

    def _reset_params_kaiming(self):
        self.weights = utils.set_tensor(torch.empty((self.in_size, self.out_size)))
        self.bias = utils.set_tensor(torch.zeros((self.out_size)))
        if isinstance(self.act_fn, utils.Linear):
            nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        elif isinstance(self.act_fn, utils.Tanh):
            nn.init.kaiming_normal_(self.weights)
        elif isinstance(self.act_fn, utils.ReLU):
            nn.init.kaiming_normal_(self.weights)

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)


class FCLayer(Layer):
    def __init__(
        self, in_size, out_size, act_fn, use_bias=False, kaiming_init=False, is_forward=False
    ):
        super().__init__(in_size, out_size, act_fn, use_bias, kaiming_init, is_forward=is_forward)
        self.use_bias = use_bias
        self.inp = None

    def forward(self, inp):
        self.inp = inp.clone()
        out = self.act_fn(torch.matmul(self.inp, self.weights))
        if self.use_bias:
            out = out + self.bias
        return out

    def backward(self, err):
        fn_deriv = self.act_fn.deriv(torch.matmul(self.inp, self.weights))
        out = torch.matmul(err * fn_deriv, self.weights.T)
        return out

    def update_gradient(self, err):
        fn_deriv = self.act_fn.deriv(torch.matmul(self.inp, self.weights))
        delta = torch.matmul(self.inp.T, err * fn_deriv)
        self.grad["weights"] = delta
        if self.use_bias:
            self.grad["bias"] = torch.sum(err, axis=0)



class ConvLayer(Layer):
  def __init__(self,input_size,num_channels,num_filters,batch_size,kernel_size,learning_rate,act_fn,padding=0,stride=1,device="cpu"):
    self.input_size = input_size
    self.num_channels = num_channels
    self.num_filters = num_filters
    self.batch_size = batch_size
    self.kernel_size = kernel_size
    self.padding = padding
    self.stride = stride
    self.output_size = math.floor((self.input_size + (2 * self.padding) - self.kernel_size)/self.stride) +1
    self.learning_rate = learning_rate
    self.act_fn
    self.device = device
    self.kernel= torch.empty(self.num_filters,self.num_channels,self.kernel_size,self.kernel_size).normal_(mean=0,std=0.05).to(self.device)
    self.unfold = nn.Unfold(kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride).to(self.device)
    self.fold = nn.Fold(output_size=(self.input_size,self.input_size),kernel_size=(self.kernel_size,self.kernel_size),padding=self.padding,stride=self.stride).to(self.device)

  def forward(self,inp):
    self.X_col = self.unfold(inp.clone())
    self.flat_weights = self.kernel.reshape(self.num_filters,-1)
    out = self.flat_weights @ self.X_col
    self.activations = out.reshape(self.batch_size, self.num_filters, self.output_size, self.output_size)
    return self.f(self.activations)

  def update_gradient(self,e):
    fn_deriv = self.act_fn.deriv(self.activations)
    e = e * fn_deriv
    self.dout = e.reshape(self.batch_size,self.num_filters,-1)
    dW = self.dout @ self.X_col.permute(0,2,1)
    dW = torch.sum(dW,dim=0)
    dW = dW.reshape((self.num_filters,self.num_channels,self.kernel_size,self.kernel_size))
    self.grad["weights"] = dW

  def backward(self,e):
    fn_deriv = self.act_fn.deriv(self.activations)
    e = e * fn_deriv
    self.dout = e.reshape(self.batch_size,self.num_filters,-1)
    dX_col = self.flat_weights.T @ self.dout
    dX = self.fold(dX_col)
    return torch.clamp(dX,-50,50)

  def get_true_weight_grad(self):
    return self.kernel.grad

  def set_weight_parameters(self):
    self.kernel = nn.Parameter(self.kernel)

  def save_layer(self,logdir,i):
      np.save(logdir +"/layer_"+str(i)+"_weights.npy",self.kernel.detach().cpu().numpy())

  def load_layer(self,logdir,i):
    kernel = np.load(logdir +"/layer_"+str(i)+"_weights.npy")
    self.kernel = set_tensor(torch.from_numpy(kernel))

class MaxPool(Layer):
  def __init__(self, kernel_size,device='cpu'):
    self.kernel_size = kernel_size
    self.device = device
    self.activations = torch.empty(1)
    self.weights = torch.zeros((1,1)).to(self.device)

  def forward(self,x):
    out, self.idxs = F.max_pool2d(x, self.kernel_size,return_indices=True)
    return out

  def backward(self, y):
    return F.max_unpool2d(y,self.idxs, self.kernel_size)

  def update_gradient(self,e,update_weights=False,sign_reverse=False):
    self.grad["weights"] = torch.zeros_like(self.weights)

  def get_true_weight_grad(self):
    return None

  def set_weight_parameters(self):
    pass

  def save_layer(self,logdir,i):
    pass

  def load_layer(self,logdir,i):
    pass

class AvgPool(Layer):
  def __init__(self, kernel_size,device='cpu'):
    self.kernel_size = kernel_size
    self.device = device
    self.activations = torch.empty(1)
    self.weights = torch.zeros((1,1)).to(self.device)


  def forward(self, x):
    self.B_in,self.C_in,self.H_in,self.W_in = x.shape
    return F.avg_pool2d(x,self.kernel_size)

  def backward(self, y):
    N,C,H,W = y.shape
    print("in backward: ", y.shape)
    return F.interpolate(y,scale_factor=(1,1,self.kernel_size,self.kernel_size))

  def update_gradient(self,e,update_weights=False, sign_reverse=False):
    self.grad["weights"] = torch.zeros_like(self.weights)

  def save_layer(self,logdir,i):
    pass

  def load_layer(self,logdir,i):
    pass

class ProjectionLayer(Layer):
  def __init__(self,input_size, output_size,act_fn,learning_rate,device='cpu'):
    self.input_size = input_size
    self.B, self.C, self.H, self.W = self.input_size
    self.output_size =output_size
    self.learning_rate = learning_rate
    self.act_fn = act_fn
    self.device = device
    self.Hid = self.C * self.H * self.W
    self.weights = torch.empty((self.Hid, self.output_size)).normal_(mean=0.0, std=0.05).to(self.device)

  def forward(self, x):
    self.inp = x.detach().clone()
    out = x.reshape((len(x), -1))
    self.activations = torch.matmul(out,self.weights)
    return self.f(self.activations)

  def backward(self, e):
    fn_deriv = self.act_fn.deriv(self.activations)
    out = torch.matmul(e * fn_deriv, self.weights.T)
    out = out.reshape((len(e), self.C, self.H, self.W))
    return torch.clamp(out,-50,50)

  def update_gradient(self, e,update_weights=False,sign_reverse=False):
    out = self.inp.reshape((len(self.inp), -1))
    fn_deriv = self.act_fn.deriv(self.activations)
    dw = torch.matmul(out.T, e * fn_deriv)
    self.grad["weights"] = dw

  def get_true_weight_grad(self):
    return self.weights.grad

  def set_weight_parameters(self):
    self.weights = nn.Parameter(self.weights)

  def save_layer(self,logdir,i):
    np.save(logdir +"/layer_"+str(i)+"_weights.npy",self.weights.detach().cpu().numpy())

  def load_layer(self,logdir,i):
    weights = np.load(logdir +"/layer_"+str(i)+"_weights.npy")
    self.weights = set_tensor(torch.from_numpy(weights))
