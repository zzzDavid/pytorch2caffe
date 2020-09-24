import traceback
from Caffe import caffe_net
from torch.nn.modules.utils import _pair
import torch
import torch.nn.functional as F
from torch import Tensor as t
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger().getChild('Translate::functions')

def _conv2d(translog, raw, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, torch_name=None):
    x = raw(input, weight, bias, stride, padding, dilation, groups)
    layer_name = translog.add_layer(name='conv', torch_name=torch_name)
    top_blob_names = translog.add_blobs([x], name='conv_blob')   
    logger.info(f'---> layer name: {layer_name}, bottom blobs: {[translog.blobs[id(input)]]}, top blobs: {top_blob_names}') 
    layer = caffe_net.Layer_param(name=layer_name, type='Convolution',
                                  bottom=[translog.blobs[id(input)]], top=[translog.blobs[id(x)]])
    layer.conv_param(x.size()[1], weight.size()[2:], stride=_pair(stride),
                     pad=_pair(padding), dilation=_pair(dilation), bias_term=bias is not None, groups=groups)
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
    else:
        layer.param.convolution_param.bias_term = False
        layer.add_data(weight.cpu().data.numpy())
    translog.cnet.add_layer(layer)
    return x


def _conv_transpose2d(translog, raw, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1,
                      torch_name=None):
    x = raw(input, weight, bias, stride, padding, output_padding, groups, dilation)
    name = translog.add_layer(name='conv_transpose', torch_name=torch_name)
    translog.add_blobs([x], name='conv_transpose_blob')
    layer = caffe_net.Layer_param(name=name, type='Deconvolution',
                                  bottom=[translog.blobs[id(input)]], top=[translog.blobs[id(x)]])
    layer.conv_param(x.size()[1], weight.size()[2:], stride=_pair(stride),
                     pad=_pair(padding), dilation=_pair(dilation), bias_term=bias is not None, groups=groups)
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
    else:
        layer.param.convolution_param.bias_term = False
        layer.add_data(weight.cpu().data.numpy())
    translog.cnet.add_layer(layer)
    return x


def _linear(translog, raw, input, weight, bias=None, torch_name=None):
    x = raw(input, weight, bias)
    layer_name = translog.add_layer(name='fc', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='fc_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='InnerProduct',
                                  bottom=[translog.blobs[id(input)]], top=[translog.blobs[id(x)]])
    layer.fc_param(x.size()[-1], has_bias=bias is not None)
    if bias is not None:
        layer.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
    else:
        layer.add_data(weight.cpu().data.numpy())
    translog.cnet.add_layer(layer)
    return x


def _split(translog, raw, tensor, split_size, dim=0, torch_name=None):
    # split in pytorch is slice in caffe
    x = raw(tensor, split_size, dim)
    layer_name = translog.add_layer('split', torch_name=torch_name)
    top_blobs = translog.add_blobs(x, name='split_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Slice',
                                  bottom=[translog.blobs[id(tensor)]], top=[translog.blobs[id(x)]])
    if not isinstance(split_size, (list, tuple)):
        # int, split size
        slice_num = int(np.floor(tensor.size()[dim] / split_size))
        slice_param = caffe_net.pb.SliceParameter(axis=dim, slice_point=[split_size * i for i in range(1, slice_num)])
    else:
        # split sections
        for i in range(1, len(split_size)):
            split_size[i] += split_size[i - 1]
        slice_param = caffe_net.pb.SliceParameter(axis=dim, slice_point=split_size[:-1])
    layer.param.slice_param.CopyFrom(slice_param)
    translog.cnet.add_layer(layer)
    return x


def _pool(translog, type, raw, input, x, kernel_size, stride, padding, torch_name=None):
    # TODO dilation,ceil_mode,return indices
    layer_name = translog.add_layer(name='{}_pool'.format(type), torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='{}_pool_blob'.format(type))
    logger.info(f'---> layer name: {layer_name}, bottom blobs: {[translog.blobs[id(input)]]}, top blobs: {top_blobs}') 
    layer = caffe_net.Layer_param(name=layer_name, type='Pooling',
                                  bottom=[translog.blobs[id(input)]], top=[translog.blobs[id(x)]])
    # TODO w,h different kernel, stride and padding
    # processing ceil mode
    layer.pool_param(kernel_size=kernel_size, stride=kernel_size if stride is None else stride,
                     pad=padding, type=type.upper())
    translog.cnet.add_layer(layer)
    if stride is not None:
        oheight = (input.size()[2] - _pair(kernel_size)[0] + 2 * _pair(padding)[0]) % (_pair(stride)[0])
        owidth = (input.size()[3] - _pair(kernel_size)[1] + 2 * _pair(padding)[1]) % (_pair(stride)[1])
        if oheight != 0 or owidth != 0:
            caffe_out = raw(input, kernel_size, stride, padding, ceil_mode=True)
            print("WARNING: the output shape miss match at {}: "

                  "input {} output---Pytorch:{}---Caffe:{}\n"
                  "This is caused by the different implementation that ceil mode in caffe and the floor mode in pytorch.\n"
                  "You can add the clip layer in caffe prototxt manually if shape mismatch error is caused in caffe. ".format(
                layer_name, input.size(), x.size(), caffe_out.size()))


# def _globalavgpool(translog, raw, input, x)                


def _max_pool2d(translog, raw, input, kernel_size, stride=None, padding=0, dilation=1,
                ceil_mode=False, return_indices=False, torch_name=None):
    x = raw(input, kernel_size, stride, padding, dilation, ceil_mode, return_indices)
    _pool(translog, 'max', raw, input, x, kernel_size, stride, padding, torch_name=torch_name)
    return x


def _avg_pool2d(translog, raw, input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True,
                divisor_override=None, torch_name=None):
    x = raw(input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
    _pool(translog, 'ave', raw, input, x, kernel_size, stride, padding, torch_name=torch_name)
    return x


def _max(translog, raw, *args, torch_name=None):
    x = raw(*args)
    if len(args) == 1:
        # TODO max in one tensor
        assert NotImplementedError
    else:
        bottom_blobs = []
        for arg in args:
            bottom_blobs.append(translog.blobs[id(arg)])
        layer_name = translog.add_layer(name='max', torch_name=torch_name)
        top_blobs = translog.add_blobs([x], name='max_blob')
        layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                      bottom=bottom_blobs, top=[translog.blobs[id(x)]])
        layer.param.eltwise_param.operation = 2
        translog.cnet.add_layer(layer)
    return x


def _cat(translog, raw, inputs, dimension=0, torch_name=None):
    x = raw(inputs, dimension)
    bottom_blobs = []
    for input in inputs:
        bottom_blobs.append(translog.blobs[id(input)])
    layer_name = translog.add_layer(name='cat', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='cat_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Concat',
                                  bottom=bottom_blobs, top=[translog.blobs[id(x)]])
    layer.param.concat_param.axis = dimension
    translog.cnet.add_layer(layer)
    return x


def _dropout(translog, raw, input, p=0.5, training=False, inplace=False, torch_name=None):
    bottom_blobs = [translog.blobs[id(input)]]
    x = raw(input, p, training, False)
    layer_name = translog.add_layer(name='dropout', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name=bottom_blobs[0], with_num=False)
    layer = caffe_net.Layer_param(name=layer_name, type='Dropout',
                                  bottom=bottom_blobs, top=[translog.blobs[id(x)]])
    layer.param.dropout_param.dropout_ratio = p
    layer.param.include.extend([caffe_net.pb.NetStateRule(phase=0)])  # 1 for test, 0 for train
    translog.cnet.add_layer(layer)
    return x


def _threshold(translog, raw, input, threshold, value, inplace=False, torch_name=None):
    # for threshold or relu
    if threshold == 0 and value == 0:
        x = raw(input, threshold, value, inplace)
        bottom_blobs = [translog.blobs[id(input)]]
        name = translog.add_layer(name='relu', torch_name=torch_name)
        translog.add_blobs([x], name='relu_blob')
        layer = caffe_net.Layer_param(name=name, type='ReLU',
                                      bottom=bottom_blobs, top=[translog.blobs[id(x)]])
        translog.cnet.add_layer(layer)
        return x
    if value != 0:
        raise NotImplemented("value !=0 not implemented in caffe")
    x = raw(input, input, threshold, value, inplace)
    bottom_blobs = [translog.blobs[id(input)]]
    layer_name = translog.add_layer(name='threshold', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='threshold_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Threshold',
                                  bottom=bottom_blobs, top=[translog.blobs[id(x)]])
    layer.param.threshold_param.threshold = threshold
    translog.cnet.add_layer(layer)
    return x


def _relu(translog, raw, input, inplace=False, torch_name=None):
    # for threshold or prelu
    x = raw(input, False)
    layer_name = translog.add_layer(name='relu', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='relu_blob')
    logger.info(f'---> layer name: {layer_name}, bottom blobs: {[translog.blobs[id(input)]]}, top blobs: {top_blobs}') 
    layer = caffe_net.Layer_param(name=layer_name, type='ReLU',
                                  bottom=[translog.blobs[id(input)]], top=[translog.blobs[id(x)]])
    translog.cnet.add_layer(layer)
    return x


def _relu6(translog, raw, input, inplace=False, torch_name=None):
    # FIXME: as dpu do not suppport relu6, try use relu
    x = raw(input, False)
    layer_name = translog.add_layer(name='relu', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='relu_blob')
    logger.info(f'---> layer name: {layer_name}, bottom blobs: {[translog.blobs[id(input)]]}, top blobs: {top_blobs}') 
    layer = caffe_net.Layer_param(name=name, type='ReLU',
                                  bottom=[translog.blobs[id(input)]], top=[translog.blobs[id(x)]])
    translog.cnet.add_layer(layer)
    return x


def _prelu(translog, raw, input, weight, torch_name=None):
    # for threshold or prelu
    x = raw(input, weight)
    bottom_blobs = [translog.blobs[id(input)]]
    name = translog.add_layer(name='prelu', torch_name=torch_name)
    translog.add_blobs([x], name='prelu_blob')
    layer = caffe_net.Layer_param(name=name, type='PReLU',
                                  bottom=bottom_blobs, top=[translog.blobs[id(x)]])
    if weight.size()[0] == 1:
        layer.param.prelu_param.channel_shared = True
        layer.add_data(weight.cpu().data.numpy()[0])
    else:
        layer.add_data(weight.cpu().data.numpy())
    translog.cnet.add_layer(layer)
    return x


def _leaky_relu(translog, raw, input, negative_slope=0.01, inplace=False, torch_name=None):
    x = raw(input, negative_slope)
    name = translog.add_layer(name='leaky_relu', torch_name=torch_name)
    translog.add_blobs([x], name='leaky_relu_blob')
    layer = caffe_net.Layer_param(name=name, type='ReLU',
                                  bottom=[translog.blobs[id(input)]], top=[translog.blobs[id(x)]])
    layer.param.relu_param.negative_slope = negative_slope
    translog.cnet.add_layer(layer)
    return x


def _tanh(translog, raw, input, torch_name=None):
    # for tanh activation
    x = raw(input)
    name = translog.add_layer(name='tanh', torch_name=torch_name)
    translog.add_blobs([x], name='tanh_blob')
    layer = caffe_net.Layer_param(name=name, type='TanH',
                                  bottom=[translog.blobs[id(input)]], top=[translog.blobs[id(x)]])
    translog.cnet.add_layer(layer)
    return x


def _softmax(translog, raw, input, dim=None, _stacklevel=3, torch_name=None):
    # for F.softmax
    x = raw(input, dim=dim)
    if dim is None:
        dim = F._get_softmax_dim('softmax', input.dim(), _stacklevel)
    bottom_blobs = [translog.blobs[id(input)]]
    name = translog.add_layer(name='softmax', torch_name=torch_name)
    translog.add_blobs([x], name='softmax_blob')
    layer = caffe_net.Layer_param(name=name, type='Softmax',
                                  bottom=bottom_blobs, top=[translog.blobs[id(x)]])
    layer.param.softmax_param.axis = dim
    translog.cnet.add_layer(layer)
    return x


def _batch_norm(translog, raw, input, running_mean, running_var, weight=None, bias=None,
                training=False, momentum=0.1, eps=1e-5, torch_name=None):
    # because the runing_mean and runing_var will be changed after the _batch_norm operation, we first save the parameters

    x = raw(input, running_mean, running_var, weight, bias,
            training, momentum, eps)
    bottom_blobs = [translog.blobs[id(input)]]
    layer_name1 = translog.add_layer(name='batch_norm', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='batch_norm_blob')
    logger.info(f'---> layer name: {layer_name1}, bottom blobs: {bottom_blobs}, top blobs: {top_blobs}') 
    layer1 = caffe_net.Layer_param(name=layer_name1, type='BatchNorm',
                                   bottom=bottom_blobs, top=[translog.blobs[id(x)]])
    if running_mean is None or running_var is None:
        # not use global_stats, normalization is performed over the current mini-batch
        layer1.batch_norm_param(use_global_stats=0, eps=eps)
    else:
        # layer1.batch_norm_param(use_global_stats=1, eps=eps)
        layer1.batch_norm_param(eps=eps)
        running_mean_clone = running_mean.clone()
        running_var_clone = running_var.clone()
        # if weight is not None and bias is not None:
        #     layer1.add_data(running_mean_clone.cpu().numpy(), running_var_clone.cpu().numpy(), weight.cpu().data.numpy(), bias.cpu().data.numpy())
        # else:
        layer1.add_data(running_mean_clone.cpu().numpy(), running_var_clone.cpu().numpy(), np.array([1.0]))
    translog.cnet.add_layer(layer1)

    if weight is not None and bias is not None:
        layer_name2 = translog.add_layer(name='bn_scale', torch_name=torch_name)
        logger.info(f'---> layer name: {layer_name2}, bottom blobs: {[translog.blobs[id(x)]]}, top blobs: {[translog.blobs[id(x)]]}') 
        layer2 = caffe_net.Layer_param(name=layer_name2, type='Scale',
                                       bottom=[translog.blobs[id(x)]], top=[translog.blobs[id(x)]])
        layer2.param.scale_param.bias_term = True
        layer2.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
        translog.cnet.add_layer(layer2)

    return x


def _instance_norm(translog, raw, input, running_mean=None, running_var=None, weight=None,
                   bias=None, use_input_stats=True, momentum=0.1, eps=1e-5, torch_name=None):
    # TODO: the batch size!=1 view operations
    print("WARNING: The Instance Normalization transfers to Caffe using BatchNorm, so the batch size should be 1")
    if running_var is not None or weight is not None:
        # TODO: the affine=True or track_running_stats=True case
        raise NotImplementedError("not implement the affine=True or track_running_stats=True case InstanceNorm")
    x = torch.batch_norm(
        input, weight, bias, running_mean, running_var,
        use_input_stats, momentum, eps, torch.backends.cudnn.enabled)
    bottom_blobs = [translog.blobs[id(input)]]
    layer_name1 = translog.add_layer(name='instance_norm', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='instance_norm_blob')
    layer1 = caffe_net.Layer_param(name=layer_name1, type='BatchNorm',
                                   bottom=bottom_blobs, top=[translog.blobs[id(x)]])
    if running_mean is None or running_var is None:
        # not use global_stats, normalization is performed over the current mini-batch
        layer1.batch_norm_param(use_global_stats=0, eps=eps)
        running_mean = torch.zeros(input.size()[1])
        running_var = torch.ones(input.size()[1])
    else:
        layer1.batch_norm_param(use_global_stats=1, eps=eps)
    running_mean_clone = running_mean.clone()
    running_var_clone = running_var.clone()
    layer1.add_data(running_mean_clone.cpu().numpy(), running_var_clone.cpu().numpy(), np.array([1.0]))
    translog.cnet.add_layer(layer1)
    if weight is not None and bias is not None:
        layer_name2 = translog.add_layer(name='bn_scale', torch_name=torch_name)
        layer2 = caffe_net.Layer_param(name=layer_name2, type='Scale',
                                       bottom=top_blobs, top=[translog.blobs[id(x)]])
        layer2.param.scale_param.bias_term = True
        layer2.add_data(weight.cpu().data.numpy(), bias.cpu().data.numpy())
        translog.cnet.add_layer(layer2)
    return x


# upsample layer
def _interpolate(translog, raw, input, size=None, scale_factor=None, mode='nearest', align_corners=None, torch_name=None):
    # 定义的参数包括 scale,即输出与输入的尺寸比例,如 2;scale_h、scale_w,
    # 同 scale,分别为 h、w 方向上的尺寸比例;pad_out_h、pad_out_w,仅在 scale 为 2 时
    # 有用,对输出进行额外 padding 在 h、w 方向上的数值;upsample_h、upsample_w,输
    # 出图像尺寸的数值。在 Upsample 的相关代码中,推荐仅仅使用 upsample_h、
    # upsample_w 准确定义 Upsample 层的输出尺寸,其他所有的参数都不推荐继续使用。
    # for nearest _interpolate
    if mode != "nearest" or align_corners != None:
        raise NotImplementedError("not implement F.interpolate totoaly")
    x = raw(input, size, scale_factor, mode)

    layer_name = translog.add_layer(name='upsample', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='upsample_blob'.format(type))
    layer = caffe_net.Layer_param(name=layer_name, type='Upsample',
                                  bottom=[translog.blobs[id(input)]], top=[translog.blobs[id(x)]])

    layer.upsample_param(size=(input.size(2), input.size(3)), scale_factor=scale_factor)
    translog.cnet.add_layer(layer)
    return x


# sigmid layer
def _sigmoid(translog, raw, input, torch_name=None):
    # Applies the element-wise function:
    #
    # Sigmoid(x)= 1/(1+exp(−x)）
    #
    # ​
    x = raw(input)
    name = translog.add_layer(name='sigmoid', torch_name=torch_name)
    translog.add_blobs([x], name='sigmoid_blob')
    layer = caffe_net.Layer_param(name=name, type='Sigmoid',
                                  bottom=[translog.blobs[id(input)]], top=[translog.blobs[id(x)]])
    translog.cnet.add_layer(layer)
    return x


# tanh layer
def _tanh(translog, raw, input, torch_name=None):
    # Applies the element-wise function:
    #
    # torch.nn.Tanh
    #
    # ​
    x = raw(input)
    name = translog.add_layer(name='tanh', torch_name=torch_name)
    translog.add_blobs([x], name='tanh_blob')
    layer = caffe_net.Layer_param(name=name, type='TanH',
                                  bottom=[translog.blobs[id(input)]], top=[translog.blobs[id(x)]])
    translog.cnet.add_layer(layer)
    return x


def _squeeze(translog, raw, inputs, *args, torch_name=None):
    x = raw(inputs, *args)
    layer_name = translog.add_layer(name='squeeze', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='view_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Reshape',
                                  bottom=[translog.blobs[id(inputs)]], top=[translog.blobs[id(x)]])
    dims = [0, -1]
    layer.param.reshape_param.shape.CopyFrom(caffe_net.pb.BlobShape(dim=dims))

    translog.cnet.add_layer(layer)
    return x


def _flatten(translog, raw, inputs, *args, torch_name=None):
    dims = inputs.shape
    x = raw(inputs, *args)
    if len(args) == 0:
        start = 0
    else:
        start = args[0]
    layer_name = translog.add_layer(name='flatten', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='flatten_blob')
    logger.info(f'---> layer name: {layer_name}, bottom blobs: {[translog.blobs[id(inputs)]]}, top blobs: {top_blobs}') 
    layer = caffe_net.Layer_param(name=layer_name, type='Flatten',
                                  bottom=[translog.blobs[id(inputs)]], top=[translog.blobs[id(x)]])
    dims = [dims[i] for i in range(start)] + [-1]
    #layer.param.reshape_param.shape.CopyFrom(caffe_net.pb.BlobShape(dim=dims))

    translog.cnet.add_layer(layer)
    return x


# ----- for Variable operations --------

def _view(translog, raw, input, *args, torch_name=None):
    x = t.view(input, *args)
    layer_name = translog.add_layer(name='view', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='view_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Reshape',
                                  bottom=[translog.blobs[id(input)]], top=[translog.blobs[id(x)]])
    # TODO: reshpae added to nn_tools layer
    dims = list(args)
    dims[0] = 0  # the first dim should be batch_size
    layer.param.reshape_param.shape.CopyFrom(caffe_net.pb.BlobShape(dim=dims))
    translog.cnet.add_layer(layer)
    return x


def _mean(translog, raw, input, *args, torch_name=None, **kwargs):
    x = t.mean(input, *args, **kwargs)
    layer_name = translog.add_layer(name='mean', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='mean_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Reduction',
                                  bottom=[translog.blobs[id(input)]], top=[translog.blobs[id(x)]])
    if len(args) == 1:
        dim = args[0]
    elif 'dim' in kwargs:
        dim = kwargs['dim']
    else:
        raise NotImplementedError('mean operation must specify a dim')
    layer.param.reduction_param.operation = 4
    layer.param.reduction_param.axis = dim
    translog.cnet.add_layer(layer)
    return x


def _add(translog, raw, input, *args, torch_name=None):
    x = t.__add__(input, *args)
    layer_name = translog.add_layer(name='add', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='add_blob')
    if isinstance(args[0], (int, float)):
        # handle add constant bias
        # layer = caffe_net.Layer_param(name=layer_name, type='Bias', bottom=[log.blobs(input)], top=[log.blobs(x)])
        # layer.bias_param(args[0], trainable=False)
        # DPU does not support Bias layer
        layer = caffe_net.Layer_param(name=layer_name, type='Scale',
                                      bottom=[translog.blobs[id(input)]], top=[translog.blobs[id(x)]])
        layer.param.scale_param.bias_term = True
        scale = torch.ones(x.size()[1], dtype=x.dtype)
        bias = torch.ones(x.size()[1], dtype=x.dtype) * args[0]
        layer.add_data(scale.numpy(), bias.numpy())
        translog.cnet.add_layer(layer)
    else:
        # elementwise add
        layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                      bottom=[translog.blobs[id(input)], translog.blobs[id(args[0])]], top=[translog.blobs[id(x)]])
        layer.param.eltwise_param.operation = 1  # sum is 1
        translog.cnet.add_layer(layer)
    return x

def _iadd(raw, translog, torch_name=None):
    
    for stack in traceback.walk_stack(None):
        if 'self' in stack[0].f_locals:
            layer = stack[0].f_locals['self']
            import ipdb; ipdb.set_trace()
    
    def __patched_iadd__(input, other):
        if id(input) not in translog.blobs.keys() and id(other) not in translog.blobs.keys():
            return input
        b1, b2 = translog.blobs[id(input)], translog.blobs[id(other)]
        x = raw(input, other)
        x = x.clone()
        layer_name = translog.add_layer(name='eltwise', torch_name=torch_name)
        top_blobs = translog.add_blobs([x], name='iadd_blob')
        logger.info(f'---> layer name: {layer_name}, bottom blobs: {[b1, b2]}, top blobs: {top_blobs}') 
        layer_param = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                            bottom=[b1, b2], top=top_blobs)
        layer_param.param.eltwise_param.operation = 1 # eltwise sum
        translog.cnet.add_layer(layer_param)
        return x

    return __patched_iadd__    
        




def _sub(translog, raw, input, *args, torch_name=None):
    x = t.__sub__(input, *args)
    layer_name = translog.add_layer(name='sub', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='sub_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[translog.blobs[id(input)], translog.blobs[id(args[0])]], top=[translog.blobs[id(x)]])
    layer.param.eltwise_param.operation = 1  # sum is 1
    layer.param.eltwise_param.coeff.extend([1., -1.])
    translog.cnet.add_layer(layer)
    return x


def _isub(translog, raw, input, *args, torch_name=None):
    x = t.__isub__(input, *args)
    x = x.clone()
    layer_name = translog.add_layer(name='isub', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='sub_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[translog.blobs[id(input)], translog.blobs[id(args[0])]], top=[translog.blobs[id(x)]])
    layer.param.eltwise_param.operation = 1  # sum is 1
    translog.cnet.add_layer(layer)
    return x


# TODO: support scalar operation using power layer (y = (shift + scale * x) ^ power, set shift = 0, power = 1)
def _mul(translog, raw, input, *args, torch_name=None):
    x = t.__mul__(input, *args)  # x is a torch Tensor
    if id(input) not in translog.keys():
        return x
    # element wise mul using scale layer
    if isinstance(args[0], float):
        layer_name = translog.add_layer(name='mul', torch_name=torch_name)
        top_blobs = translog.add_blobs([x], name='mul_blob')
        layer = caffe_net.Layer_param(name=layer_name, type='Scale',
                                      bottom=[translog.blobs[id(input)]], top=[translog.blobs[id(x)]])
        layer.param.scale_param.bias_term = False
        scale = torch.ones(x.size()[1], dtype=x.dtype)
        layer.add_data(scale.numpy())
        translog.cnet.add_layer(layer)
        return x
    assert args[0].shape[0] == input.shape[0] and args[0].shape[1] == input.shape[1]
    if not (args[0].shape[2] == input.shape[2] and args[0].shape[3] == input.shape[3]):
        print(
            "WARNING: DPU cannot handle this implictly-broadcast elementwise multiplication efficiently! {} with {}".format(
                args[0].shape, input.shape))
        # Handle implicitly broadcast in pytorch, reshape -> scale;
        # Actually this is not support by DPU (2019.10.16)
        # add reshape layer
        assert args[0].shape[2] == 1 and args[0].shape[3] == 1
        layer_name = translog.add_layer(name="reshape", torch_name=torch_name)
        y = args[0].view(args[0].shape[0], -1)
        layer_name = translog.add_layer(name='mul', torch_name=torch_name)
        top_blobs = translog.add_blobs([x], name='mul_blob')
        layer = caffe_net.Layer_param(name=layer_name, type='Scale',
                                      bottom=[translog.blobs[id(input)], translog.blobs[id(y)]], top=[translog.blobs[id(x)]])
        layer.param.scale_param.bias_term = False
        layer.param.scale_param.axis = 0
        translog.cnet.add_layer(layer)
    else:
        # acutally, dpu only support elementwise...
        layer_name = translog.add_layer(name='mul', torch_name=torch_name)
        top_blobs = translog.add_blobs([x], name='mul_blob')
        layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                      bottom=[translog.blobs[id(input)], translog.blobs[id(args[0])]], top=[translog.blobs[id(x)]])
        layer.param.eltwise_param.operation = 0  # product is 1
        translog.cnet.add_layer(layer)
    return x


def _imul(translog, raw, input, *args, torch_name=None):
    x = t.__imul__(input, *args)
    x = x.clone()
    layer_name = translog.add_layer(name='mul', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='mul_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[translog.blobs[id(input)], translog.blobs[id(args[0])]], top=[translog.blobs[id(x)]])
    layer.param.eltwise_param.operation = 0  # product is 1
    layer.param.eltwise_param.coeff.extend([1., -1.])
    translog.cnet.add_layer(layer)
    return x


# TODO: support division: determine which method is called, now we know that torch.Tensor.__div__ and torch.Tensor.__idiv__ are not called.
def _div(translog, raw, input, *args, torch_name=None):
    x = t.__div__(input, *args)
    # element wise mul using scale layer
    if isinstance(args[0], float):
        layer_name = translog.add_layer(name='div', torch_name=torch_name)
        top_blobs = translog.add_blobs([x], name='div_blob')
        layer = caffe_net.Layer_param(name=layer_name, type='Scale',
                                      bottom=[translog.blobs[id(input)], translog.blobs[id(args[0])]], top=[translog.blobs[id(x)]])
        layer.param.scale_param.bias_term = False
        layer.param.scale_param.axis = 0
        translog.cnet.add_layer(layer)
        return x

    assert args[0].shape[0] == input.shape[0] and args[0].shape[1] == input.shape[1]
    if not (args[0].shape[2] == input.shape[2] and args[0].shape[3] == input.shape[3]):
        print(
            "WARNING: DPU cannot handle this implictly-broadcast elementwise multiplication efficiently! {} with {}".format(
                args[0].shape, input.shape))
        # Handle implicitly broadcast in pytorch, reshape -> scale;
        # Actually this is not support by DPU (2019.10.16)
        # add reshape layer
        assert args[0].shape[2] == 1 and args[0].shape[3] == 1
        layer_name = translog.add_layer(name="reshape", torch_name=torch_name)
        y = args[0].view(args[0].shape[0], -1)
        layer_name = translog.add_layer(name='div', torch_name=torch_name)
        top_blobs = translog.add_blobs([x], name='div_blob')
        layer = caffe_net.Layer_param(name=layer_name, type='Scale',
                                      bottom=[translog.blobs[id(input)], translog.blobs[id(y)]], top=[translog.blobs[id(x)]])
        layer.param.scale_param.bias_term = False
        layer.param.scale_param.axis = 0
    else:
        # acutally, dpu only support elementwise...
        layer_name = translog.add_layer(name='div', torch_name=torch_name)
        top_blobs = translog.add_blobs([x], name='div_blob')
        layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                      bottom=[translog.blobs[id(input)], translog.blobs[id(args[0])]], top=[translog.blobs[id(x)]])
        layer.param.eltwise_param.operation = 0  # product is 1
    translog.cnet.add_layer(layer)
    return x


def _idiv(translog, raw, input, *args, torch_name=None):
    x = t.__idiv__(input, *args)
    x = x.clone()
    layer_name = translog.add_layer(name='div', torch_name=torch_name)
    top_blobs = translog.add_blobs([x], name='div_blob')
    layer = caffe_net.Layer_param(name=layer_name, type='Eltwise',
                                  bottom=[translog.blobs[id(input)], translog.blobs[id(args[0])]], top=[translog.blobs[id(x)]])
    layer.param.eltwise_param.operation = 0  # product is 1
    layer.param.eltwise_param.coeff.extend([1., -1.])
    translog.cnet.add_layer(layer)
    return x

def _v_sigmoid(translog, raw, tensor):
    return torch.sigmoid(tensor)