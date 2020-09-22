from resnet import resnet50
import sys, os
from pathlib import Path
path = Path(os.getcwd())
sys.path.append(str(path.parent))
import pytorch_to_caffe as torch2caffe
import numpy as np
from torch.autograd import Variable
import torch

if __name__ == "__main__":
    model = resnet50(pretrained=False)
    input = Variable(torch.ones([1, 3, 416, 416]))
    torch2caffe.trans_net(model, input, "resnet50")
    out_proto = 'resnet50.prototxt'
    out_caffemodel = 'resnet50.caffemodel'
    torch2caffe.save_caffemodel(out_caffemodel)
    torch2caffe.save_prototxt(out_proto)