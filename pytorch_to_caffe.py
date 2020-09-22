from Translate.functions import _conv2d, _linear, _relu, _relu6, _leaky_relu, \
    _max_pool2d, _avg_pool2d, _dropout, _threshold, _prelu, _batch_norm, _instance_norm, \
    _softmax, _conv_transpose2d, _interpolate, _sigmoid, _tanh, _squeeze, _flatten, \
    _split, _max, _cat, _mean, _view, _add, _iadd, _sub, _isub, _mul, _imul, _div, _idiv, _v_sigmoid
import torch.nn.functional as F
import torch
import traceback
import logging
from Translate.Translog import TransLog
from torch import Tensor as t
logging.basicConfig(level=logging.DEBUG)


layer_names = dict()
log = TransLog()


class Rp(object):
    def __init__(self,raw,replace,**kwargs):
        # replace the raw function to replace function
        self.obj=replace
        self.raw=raw

    def __call__(self,*args,**kwargs):
        if not NET_INITTED:
            return self.raw(*args,**kwargs)
        for stack in traceback.walk_stack(None): # walk current stack
            if 'self' in stack[0].f_locals:
                layer=stack[0].f_locals['self']
                if layer in layer_names:
                    log.pytorch_layer_name=layer_names[layer]
                    print(layer_names[layer])
                    break
        kwargs.update({'torch_name': layer_names[layer]})
        out=self.obj(self.raw, *args, **kwargs)
        # if isinstance(out,Variable):
        #     out=[out]
        return out


# Replace nn.functional Functions
F.conv2d=Rp(F.conv2d, _conv2d)
F.linear=Rp(F.linear, _linear)
F.relu=Rp(F.relu,_relu)
F.relu6=Rp(F.relu6,_relu6)
F.leaky_relu=Rp(F.leaky_relu,_leaky_relu)
F.max_pool2d=Rp(F.max_pool2d,_max_pool2d)
F.avg_pool2d=Rp(F.avg_pool2d,_avg_pool2d)
F.dropout=Rp(F.dropout,_dropout)
F.threshold=Rp(F.threshold,_threshold)
F.prelu=Rp(F.prelu,_prelu)
F.batch_norm=Rp(F.batch_norm,_batch_norm)
F.instance_norm=Rp(F.instance_norm,_instance_norm)
F.softmax=Rp(F.softmax,_softmax)
F.conv_transpose2d=Rp(F.conv_transpose2d,_conv_transpose2d)
F.interpolate = Rp(F.interpolate,_interpolate)
F.sigmoid = Rp(F.sigmoid,_sigmoid)
torch.sigmoid = Rp(torch.sigmoid,_sigmoid)
F.tanh = Rp(F.tanh,_tanh)
torch.squeeze = Rp(torch.squeeze, _squeeze)
torch.flatten = Rp(torch.flatten, _flatten)
torch.split=Rp(torch.split,_split)
torch.max=Rp(torch.max,_max)
torch.cat=Rp(torch.cat,_cat)

# Replace Tensor operations
t.view = _view
t.sigmoid = _v_sigmoid
t.mean = _mean
t.__add__ = _add
t.__iadd__ = _iadd
t.__sub__ = _sub
t.__isub__ = _isub
t.__mul__=_mul
t.__imul__ = _imul
t.__div__ = _div
t.__idiv__ = _idiv


def trans_net(net,input_var,name='TransferedPytorchModel'):
    log.__init__() # clear layer list and blob list before translation
    log.init([input_var], name) # init input layer
    log.cnet.net.name=name
    log.cnet.net.input.extend([log.blobs(input_var)])
    log.cnet.net.input_dim.extend(input_var.size())
    global NET_INITTED
    NET_INITTED=True
    for name,layer in net.named_modules():
        layer_names[layer]=name
    print("torch ops name:", layer_names)
    net.forward(input_var)
    print('Transform Completed')
    NET_INITTED = False

def save_prototxt(save_name):
    log.cnet.save_prototxt(save_name)

def save_caffemodel(save_name):
    log.cnet.save(save_name)
