"""
This tool is to remove redundant ImageData layer after Vitis quantization
This is a bug of Vitis caffe quantization tool. It cannot correctly replace the ImageData layer
with input layer.
"""
from ..Caffe import caffe_net
import os

def remove_ImageDataLayer(protoxt_path):
    if not os.path.exists(protoxt_path):
        raise Exception("prototxt file does not exist")
    cnet = caffe_net.Prototxt(protoxt_path)
    for i, layer in enumerate(cnet.layers()):
        if layer.type == "ImageData":            
            # import ipdb; ipdb.set_trace()
            first_layer_name = cnet.layers()[i+1].name
            h = layer.image_data_param.new_height
            w = layer.image_data_param.new_width
            name = layer.name
            del cnet.net.layer[i]
    
            # if there is no input layer, we add one
            # if cnet.layers()[0].type != "Input":
            #     shape = [1, 3, h, w]
            #     layer_param = caffe_net.Layer_param(
            #         name=name,
            #         type="Input",
            #         top=['data']
            #     )
            #     layer_param.input_param(shape)
            #     cnet.add_layer(layer_param, before=first_layer_name)

    os.remove(protoxt_path)
    cnet.save_prototxt(protoxt_path)        
