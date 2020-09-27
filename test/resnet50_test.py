from resnet import resnet50, resnet18
import sys, os
from pathlib import Path
path = Path(os.getcwd())
sys.path.append(str(path.parent))
from torch2caffe.pytorch_to_caffe import pytorch2caffe
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn

if __name__ == "__main__":
    model = resnet50(pretrained=False)
    # model = test_model()
    input = Variable(torch.ones([1, 3, 416, 416]))
    out_proto = 'resnet50.prototxt'
    out_caffemodel = 'resnet50.caffemodel'

    torch2caffe = pytorch2caffe(model)
    torch2caffe.set_input(input)
    torch2caffe.trans_net("resnet50")
    torch2caffe.save_caffemodel(out_caffemodel)
    torch2caffe.save_prototxt(out_proto)
