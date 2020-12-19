from resnet import resnet50, resnet18
from pytorch2caffe.torch2caffe import converter
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn

if __name__ == "__main__":
    model = resnet50(pretrained=False)
    # model = test_model()
    new_height = 224
    new_width = 224
    input = torch.ones([1, 3, new_height, new_width])
    source = "/home/zhangniansong/data/imagenet_calib/calibration.txt"
    root = "/home/zhangniansong/data/imagenet_calib/img/"
    batch_size = 1
    out_proto = 'resnet50.prototxt'
    out_caffemodel = 'resnet50.caffemodel'

    torch2caffe = converter(model)
    torch2caffe.set_input(input, source, root, batch_size, new_height, new_width)
    torch2caffe.trans_net("resnet50")
    torch2caffe.save_caffemodel(out_caffemodel)
    torch2caffe.save_prototxt(out_proto)
