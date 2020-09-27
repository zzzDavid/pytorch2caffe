"""
This tool is to remove redundant ImageData layer after Vitis quantization
This is a bug of Vitis caffe quantization tool. It cannot correctly replace the ImageData layer
with input layer.
"""
from ..Caffe import caffe_net
import os

def remove_ImageDataLayer(caffe_model_path):
    if not os.path.exists(caffe_model_path):
        raise Exception("caffemodel file does not exist")
    cnet = caffe_net.Caffemodel(caffe_model_path)
    name = caffe_model_path.replace('caffemodel', '')
    for layer in cnet.layers():
        if layer.type == "ImageData":
            cnet.remove_layer_by_name(layer.name)
    cnet.save_prototxt(name + ".prototxt")        
    os.remove(caffe_model_path)
    cnet.save(caffe_model_path)