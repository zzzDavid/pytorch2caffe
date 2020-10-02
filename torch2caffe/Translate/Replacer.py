from logging import exception
from .functions import (
    _conv2d,
    _linear,
    _relu,
    _relu6,
    _leaky_relu,
    _max_pool2d,
    _avg_pool2d,
    _global_avg_pool2d,
    _dropout,
    _threshold,
    _prelu,
    _batch_norm,
    _instance_norm,
    _softmax,
    _conv_transpose2d,
    _interpolate,
    _sigmoid,
    _tanh,
    _squeeze,
    _flatten,
    _split,
    _max,
    _cat,
    _mean,
    _view,
    _add,
    _iadd,
    _sub,
    _isub,
    _mul,
    _imul,
    _div,
    _idiv,
)
import torch.nn.functional as F
import torch.nn as nn
import torch
import traceback
from .Translog import TransLog
from torch import Tensor as t
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


class Rp(object):
    def __init__(self, raw, replace, translog, layer_names, **kwargs):
        # replace the raw function
        self.replace = replace
        self.raw = raw
        self.translog = translog
        self.layer_names = layer_names

    def __call__(self, *args, **kwargs):
        # find torch_name
        torch_name = None
        for stack in traceback.walk_stack(None):  # walk current stack
            if "self" in stack[0].f_locals:
                layer = stack[0].f_locals["self"]
                if layer in self.layer_names:
                    torch_name = self.layer_names[layer]
                    break
        # assert (
        #     torch_name is not None
        # ), f"torch function {self.raw.__name__} not found in stack"
        # add the argument `torch_name`
        kwargs.update({"torch_name": torch_name})
        out = self.replace(self.translog, self.raw, *args, **kwargs)
        return out


class Replacer():

    def __init__(self):
        pass


    def replace_functions(self, translog, torch_layer_dict):
        # Replace nn.functional Functions
        self.conv2d = F.conv2d
        F.conv2d = Rp(F.conv2d, _conv2d, translog, torch_layer_dict)

        self.linear = F.linear
        F.linear = Rp(F.linear, _linear, translog, torch_layer_dict)

        self.relu = F.relu
        F.relu = Rp(F.relu, _relu, translog, torch_layer_dict)

        self.relu6 = F.relu6
        F.relu6 = Rp(F.relu6, _relu6, translog, torch_layer_dict)

        self.leaky_relu = F.leaky_relu
        F.leaky_relu = Rp(F.leaky_relu, _leaky_relu, translog, torch_layer_dict)

        self.max_pool2d = F.max_pool2d
        F.max_pool2d = Rp(F.max_pool2d, _max_pool2d, translog, torch_layer_dict)

        self.global_avg_pool2d = F.adaptive_avg_pool2d
        F.adaptive_avg_pool2d = Rp(F.adaptive_avg_pool2d, _global_avg_pool2d, translog, torch_layer_dict)

        self.avg_pool2d = F.avg_pool2d
        F.avg_pool2d = Rp(F.avg_pool2d, _avg_pool2d, translog, torch_layer_dict)

        self.dropout = F.dropout
        F.dropout = Rp(F.dropout, _dropout, translog, torch_layer_dict)

        self.threshold = F.threshold
        F.threshold = Rp(F.threshold, _threshold, translog, torch_layer_dict)

        self.prelu = F.prelu
        F.prelu = Rp(F.prelu, _prelu, translog, torch_layer_dict)

        self.batch_norm = F.batch_norm
        F.batch_norm = Rp(F.batch_norm, _batch_norm, translog, torch_layer_dict)

        self.instance_norm = F.instance_norm
        F.instance_norm = Rp(F.instance_norm, _instance_norm, translog, torch_layer_dict)

        self.softmax = F.softmax
        F.softmax = Rp(F.softmax, _softmax, translog, torch_layer_dict)

        self.conv_transpose2d = F.conv_transpose2d
        F.conv_transpose2d = Rp(
            F.conv_transpose2d, _conv_transpose2d, translog, torch_layer_dict
        )

        self.interpolate = F.interpolate
        F.interpolate = Rp(F.interpolate, _interpolate, translog, torch_layer_dict)

        self.sigmoid = F.sigmoid
        F.sigmoid = Rp(F.sigmoid, _sigmoid, translog, torch_layer_dict)

        self.torch_sigmoid = torch.sigmoid
        torch.sigmoid = Rp(torch.sigmoid, _sigmoid, translog, torch_layer_dict)

        self.tanh = F.tanh
        F.tanh = Rp(F.tanh, _tanh, translog, torch_layer_dict)

        self.squeeze = torch.squeeze
        torch.squeeze = Rp(torch.squeeze, _squeeze, translog, torch_layer_dict)

        self.flatten = torch.flatten
        torch.flatten = Rp(torch.flatten, _flatten, translog, torch_layer_dict)

        self.split = torch.split
        torch.split = Rp(torch.split, _split, translog, torch_layer_dict)

        self.max = torch.max
        torch.max = Rp(torch.max, _max, translog, torch_layer_dict)

        self.cat = torch.cat
        torch.cat = Rp(torch.cat, _cat, translog, torch_layer_dict)

        # Replace Tensor operations
        self.view = t.view
        t.view = _view(t.view, translog)

        self.mean = t.mean
        t.mean = _mean(t.mean, translog)

        self._add= t.__add__
        t.__add__ = _add(t.__add__, translog)

        self._iadd = t.__iadd__
        t.__iadd__ = _iadd(t.__iadd__, translog)

        self._sub = t.__sub__
        t.__sub__ = _sub(t.__sub__, translog)

        self._isub = t.__isub__
        t.__isub__ = _isub(t.__isub__, translog)

        self._mul = t.__mul__
        t.__mul__ = _mul(t.__mul__, translog)

        self._imul = t.__imul__
        t.__imul__ = _imul(t.__imul__, translog)

        self._div = t.__div__
        t.__div__ = _div(t.__div__, translog)
        
        self._idiv = t.__idiv__
        t.__idiv__ = _idiv(t.__idiv__, translog)

        logger.info("torch functions have been replaced")

    def place_back(self):
        F.conv2d = self.conv2d
        F.linear = self.linear
        F.relu = self.relu
        F.relu6 = self.relu6
        F.leaky_relu = self.leaky_relu
        F.max_pool2d = self.max_pool2d
        F.avg_pool2d = self.avg_pool2d
        F.adaptive_avg_pool2d = self.global_avg_pool2d
        F.dropout = self.dropout
        F.threshold = self.threshold
        F.prelu = self.prelu
        F.batch_norm = self.batch_norm
        F.instance_norm = self.instance_norm
        F.soft_max = self.softmax
        F.conv_transpose2d = self.conv_transpose2d 
        F.interpolate = self.interpolate 
        F.sigmoid = self.sigmoid 
        F.sigmoid = self.torch_sigmoid
        F.tanh = self.tanh 
        torch.squeeze = self.squeeze
        torch.flatten = self.flatten
        torch.split = self.split
        torch.max = self.max
        torch.cat = self.cat
        # Replace Tensor operations
        t.view = self.view
        t.mean = self.mean
        t.__add__ = self._add
        t.__iadd__ = self._iadd
        t.__sub__ = self._sub
        t.__isub__ = self._isub
        t.__mul__ = self._mul
        t.__imul__ = self._imul
        t.__div__ = self._div
        t.__idiv__ = self._idiv
