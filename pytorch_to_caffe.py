from logging import exception
from Translate.functions import _conv2d, _linear, _relu, _relu6, _leaky_relu, \
    _max_pool2d, _avg_pool2d, _dropout, _threshold, _prelu, _batch_norm, _instance_norm, \
    _softmax, _conv_transpose2d, _interpolate, _sigmoid, _tanh, _squeeze, _flatten, \
    _split, _max, _cat, _mean, _view, _add, _iadd, _sub, _isub, _mul, _imul, _div, _idiv
import torch.nn.functional as F
import torch.nn as nn
import torch
import traceback
from Translate.Translog import TransLog
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

    def __call__(self,*args,**kwargs):
        # find torch_name
        torch_name = None
        for stack in traceback.walk_stack(None): # walk current stack
            if 'self' in stack[0].f_locals:
                layer = stack[0].f_locals['self']
                if layer in self.layer_names:
                    torch_name = self.layer_names[layer]
                    break
        assert torch_name is not None, f"torch function {self.raw.__name__} not found in stack"         
        # add the argument `torch_name`
        kwargs.update({'torch_name': torch_name})
        out=self.replace(self.translog, self.raw, *args, **kwargs)
        return out




def replace_functions(translog, torch_layer_dict):        
    # Replace nn.functional Functions
    F.conv2d             = Rp(F.conv2d, _conv2d, translog, torch_layer_dict)
    F.linear             = Rp(F.linear, _linear,translog, torch_layer_dict)
    F.relu               = Rp(F.relu,_relu, translog, torch_layer_dict)
    F.relu6              = Rp(F.relu6,_relu6, translog, torch_layer_dict)
    F.leaky_relu         = Rp(F.leaky_relu,_leaky_relu, translog, torch_layer_dict)
    F.max_pool2d         = Rp(F.max_pool2d,_max_pool2d, translog, torch_layer_dict)
    F.avg_pool2d         = Rp(F.avg_pool2d,_avg_pool2d, translog, torch_layer_dict)
    F.dropout            = Rp(F.dropout,_dropout, translog, torch_layer_dict)
    F.threshold          = Rp(F.threshold,_threshold, translog, torch_layer_dict)
    F.prelu              = Rp(F.prelu,_prelu, translog, torch_layer_dict)
    F.batch_norm         = Rp(F.batch_norm,_batch_norm, translog, torch_layer_dict)
    F.instance_norm      = Rp(F.instance_norm,_instance_norm, translog, torch_layer_dict)
    F.softmax            = Rp(F.softmax,_softmax, translog, torch_layer_dict)
    F.conv_transpose2d   = Rp(F.conv_transpose2d,_conv_transpose2d, translog, torch_layer_dict)
    F.interpolate        = Rp(F.interpolate,_interpolate, translog, torch_layer_dict)
    F.sigmoid            = Rp(F.sigmoid,_sigmoid, translog, torch_layer_dict)
    torch.sigmoid        = Rp(torch.sigmoid,_sigmoid, translog, torch_layer_dict)
    F.tanh               = Rp(F.tanh,_tanh, translog, torch_layer_dict)
    torch.squeeze        = Rp(torch.squeeze, _squeeze, translog, torch_layer_dict)
    torch.flatten        = Rp(torch.flatten, _flatten, translog, torch_layer_dict)
    torch.split          = Rp(torch.split,_split, translog, torch_layer_dict)
    torch.max            = Rp(torch.max,_max, translog, torch_layer_dict)
    torch.cat            = Rp(torch.cat,_cat, translog, torch_layer_dict)

    # Replace Tensor operations
    t.__view__ = _view(t.__view__, translog)
    t.__mean__ = _mean(t.__mean__, translog)
    t.__add__ = _add(t.__add__, translog)
    t.__iadd__ = _iadd(t.__iadd__, translog)
    t.__sub__ = _sub(t.__sub__, translog)
    t.__isub__ = _isub(t.__isub__, translog)
    t.__mul__ = _mul(t.__mul__, translog)
    t.__imul__ = _imul(t.__imul__, translog)
    t.__div__ = _div(t.__div__, translog)
    t.__idiv__ = _idiv(t.__idiv__, translog)

    logger.info("torch functions have been replaced")
    


class pytorch2caffe(object):
    def __init__(self, model):
        self.layer_names = dict()
        self.translog = TransLog()
        self.model = model
        for name, layer in model.named_modules():
            self.layer_names[layer] = name
        replace_functions(self.translog, self.layer_names)

    def trans_net(self, input_var, name='TransferedPytorchModel'):
        # import ipdb; ipdb.set_trace()
        self.translog.set_net_name(name)
        self.translog.set_input(input_var)
        x = self.model.forward(input_var)
        self.translog.set_softmaxwithloss(x)

        # torch to caffe names
        logger.debug("printing torch to caffe names: ")
        for key, value in self.translog.torch_to_caffe_names.items():
            logger.debug(f"torch name: {key}  --> caffe name: {value}")


    def save_prototxt(self, save_name):
        logger.info("saving prototxt to: " + save_name + " ...")
        self.translog.cnet.save_prototxt(save_name)

    def save_caffemodel(self, save_name):
        logger.info("saving caffemodel to: " + save_name + " ...")
        self.translog.cnet.save(save_name)
