from Translate.functions import _conv2d, _linear, _relu, _relu6, _leaky_relu, \
    _max_pool2d, _avg_pool2d, _dropout, _threshold, _prelu, _batch_norm, _instance_norm, \
    _softmax, _conv_transpose2d, _interpolate, _sigmoid, _tanh, _squeeze, _flatten, \
    _split, _max, _cat, _mean, _view, _add, _iadd, _sub, _isub, _mul, _imul, _div, _idiv, _v_sigmoid    
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
    def __init__(self, raw, replace, translog, **kwargs):
        # replace the raw function
        self.replace = replace
        self.raw = raw
        self.translog = translog

    def __call__(self,*args,**kwargs):
        # for stack in traceback.walk_stack(None): # walk current stack
        #     if 'self' in stack[0].f_locals:
        #         layer = stack[0].f_locals['self']
        #         if layer in self.layer_names:
        #             log.pytorch_layer_name=self.layer_names[layer]
        #             break
        # add the argument `torch_name`
        # kwargs.update({'torch_name': layer_names[layer]})
        # import ipdb; ipdb.set_trace()
        out=self.replace(self.translog, self.raw, *args, **kwargs)
        return out



class Tensor_reload(t):
    def __init__(self, translog, *args, **kwargs):
        super(Tensor_reload, self).__init__()
        self.translog = translog
    def __iadd__(input, *args):
        import ipdb; ipdb.set_trace()
        if id(input) not in translog.blobs.keys() and id(args[0]) not in translog.blobs.keys():
            return input
        b1, b2 = translog.blobs[id(input)], translog.blobs[id(args[0])]
        x = t.__iadd__(input, *args)
        x = x.clone()
        layer_name = translog.add_layer(name='iadd', torch_name=torch_name)
        top_blobs = translog.add_blobs([x], name='iadd_blob')
        layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                    bottom=[b1, b2], top=[translog.blobs[id(x)]])
        layer.param.eltwise_param.operation = 1  # sum is 1
        translog.cnet.add_layer(layer)
        return x 


def replace_functions(translog):        
    # Replace nn.functional Functions
    F.conv2d             = Rp(F.conv2d, _conv2d, translog)
    F.linear             = Rp(F.linear, _linear,translog)
    F.relu               = Rp(F.relu,_relu, translog)
    F.relu6              = Rp(F.relu6,_relu6, translog)
    F.leaky_relu         = Rp(F.leaky_relu,_leaky_relu, translog)
    F.max_pool2d         = Rp(F.max_pool2d,_max_pool2d, translog)
    F.avg_pool2d         = Rp(F.avg_pool2d,_avg_pool2d, translog)
    F.dropout            = Rp(F.dropout,_dropout, translog)
    F.threshold          = Rp(F.threshold,_threshold, translog)
    F.prelu              = Rp(F.prelu,_prelu, translog)
    F.batch_norm         = Rp(F.batch_norm,_batch_norm, translog)
    F.instance_norm      = Rp(F.instance_norm,_instance_norm, translog)
    F.softmax            = Rp(F.softmax,_softmax, translog)
    F.conv_transpose2d   = Rp(F.conv_transpose2d,_conv_transpose2d, translog)
    F.interpolate        = Rp(F.interpolate,_interpolate, translog)
    F.sigmoid            = Rp(F.sigmoid,_sigmoid, translog)
    torch.sigmoid        = Rp(torch.sigmoid,_sigmoid, translog)
    F.tanh               = Rp(F.tanh,_tanh, translog)
    torch.squeeze        = Rp(torch.squeeze, _squeeze, translog)
    torch.flatten        = Rp(torch.flatten, _flatten, translog)
    torch.split          = Rp(torch.split,_split, translog)
    torch.max            = Rp(torch.max,_max, translog)
    torch.cat            = Rp(torch.cat,_cat, translog)

    # Replace Tensor operations
    from torch import Tensor
    Tensor = Tensor_reload
    # t.view = Rp(t.view, _view, translog)
    # t.sigmoid = Rp(t.sigmoid, _v_sigmoid, translog)
    # t.mean = Rp(t.mean, _mean, translog)
    # t.__add__ = Rp(t.__add__, _add, translog)
    # #t.__iadd__ = Rp(t.__iadd__, _iadd, translog)
    # from Translate.functions import _iadd2
    # t.__iadd__ = Rp(t.__iadd__, _iadd2, translog)._iadd
    # t.__sub__ = Rp(t.__sub__, _sub, translog)
    # t.__isub__ = Rp(t.__isub__, _isub, translog)
    # t.__mul__= Rp(t.__mul__, _mul, translog)
    # t.__imul__ = Rp(t.__imul__, _imul, translog)
    # t.__div__ = Rp(t.__div__, _div, translog)
    # t.__idiv__ = Rp(t.__idiv__, _idiv, translog)

    logger.info("torch functions have been replaced")
    


class pytorch2caffe(object):
    def __init__(self, model):
        self.layer_names = dict()
        self.translog = TransLog()
        self.model = model
        for name, layer in model.named_modules():
            self.layer_names[layer] = name
        replace_functions(self.translog)

    def trans_net(self, input_var, name='TransferedPytorchModel'):
        self.translog.set_net_name(name)
        self.translog.set_input(input_var)

        # import ipdb; ipdb.set_trace()
        self.model.forward(input_var)

    def save_prototxt(self, save_name):
        self.translog.cnet.save_prototxt(save_name)

    def save_caffemodel(self, save_name):
        self.translog.cnet.save(save_name)
