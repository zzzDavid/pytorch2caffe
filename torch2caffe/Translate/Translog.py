from ..Caffe import caffe_net
import torch
from collections import OrderedDict

## setup logger
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class TransLog(object):
    """
    self.blobs: 
        - type: dictionary
        - keys: (int) id of torch Tensor object
        - values: (str) name of the blob
    """

    def __init__(self):
        self.layers = dict()
        self.blobs = OrderedDict()
        self.layer_count = dict()
        self.blob_count = dict()
        self._blobs_data = []
        self.cnet = caffe_net.Caffemodel("")
        self.debug = True
        self._logger = logger.getChild(self.__class__.__name__)
        self.torch_to_caffe_names = dict()
        self.name = "default_name"

    def set_net_name(self, name):
        self.cnet.net.name = name
        self.name = name

    def set_input(
        self,
        input_var,
        source="",
        root_folder="",
        batch_size=1,
        channel = 3,
        new_height=224,
        new_width=224,
    ):
        """
        input_var: torch.Tensor object
        """
        # set up layer parameters
        top_blobs = self.add_blobs([input_var], name="data", with_num=False)
        label_blob = self.add_blobs(
            [torch.ones(1)], name="label", with_num=False
        )  # just a placeholder
        layer_name = self.add_layer(name="input")

        if source == "" or root_folder == "" : 
            # if we do not specify input images  
            layer_param = caffe_net.Layer_param(
                name=layer_name, type="Input", top=top_blobs
            )
            layer_param.input_param([batch_size, channel, new_height, new_width])
            self.cnet.add_layer(layer_param)
            logger.info(f"---> layer name: {layer_name}, top blobs: {top_blobs}")
            return

        # if we specify input images
        top_blobs += label_blob
        layer_param = caffe_net.Layer_param(
            name=layer_name, type="ImageData", top=top_blobs
        )
        # import ipdb; ipdb.set_trace()
        layer_param.image_data_param(source, root_folder, batch_size, new_height, new_width)
        self.cnet.add_layer(layer_param)
        logger.info(f"---> layer name: {layer_name}, top blobs: {top_blobs}")


    def set_softmaxwithloss(self, input_var):
        # import ipdb; ipdb.set_trace()
        assert (
            "label" in self.blobs.values()
        ), "Ya'll need the label blob in the input layer to construct a loss layer"
        layer_name = self.add_layer(name="loss")
        top_blobs = self.add_blobs([torch.ones(1)], name="loss", with_num=False)
        bottom_blob = self.blobs[id(input_var)]
        bottom_blobs = ["label", bottom_blob]
        logger.info(
            f"---> layer name: {layer_name}, bottom_blobs: {bottom_blobs}, top blobs: {top_blobs}"
        )
        layer_param = caffe_net.Layer_param(
            name=layer_name, type="SoftmaxWithLoss", bottom=bottom_blobs, top=top_blobs
        )
        self.cnet.add_layer(layer_param)

    def add_layer(self, name="layer", torch_name=None):
        if name in self.layers:
            return self.layers[name]
        if name not in self.layer_count.keys():
            self.layer_count[name] = 0
        self.layer_count[name] += 1
        name = "{}_{}_{}".format(self.name, name, self.layer_count[name])
        self.layers[name] = name  # ? what is it doing
        # if self.debug:
        #     print("{} was added to layers".format(self.layers[name]))
        if torch_name is not None:
            self.torch_to_caffe_names[torch_name] = self.layers[name]
        return self.layers[name]

    def add_blobs(self, blobs, name="blob", with_num=True):
        rst = []
        for i, blob in enumerate(blobs):
            self._blobs_data.append(blob)  # to block the memory address be rewrited
            blob_id = int(id(blob))
            if name not in self.blob_count.keys():
                self.blob_count[name] = 0
            self.blob_count[name] += 1
            if with_num:
                rst.append("{}{}".format(name, self.blob_count[name]))
            else:
                rst.append("{}".format(name))
            # self._logger.info(f'blob: {rst[-1]} was added to the list')
            self.blobs[blob_id] = rst[-1]
        return rst

    def post_check(self):

        self.replace_with_global_avg_pooling()        
        # for DPU
        self.remove_global_avg_pooling()
        

    def replace_with_global_avg_pooling(self):        
        # check Reduction layer
        for i, layer in enumerate(self.cnet.layers()):
            if not layer.type == 'Reduction': continue
            layers = self.cnet.layers()
            if i + 1 > len(layers) - 1: continue
            next_layer =  layers[i+1]
            if next_layer.type == 'Reduction':
                # remove the two reduction layers
                # and add an average global pooling layer
                bottoms = layer.bottom
                tops = next_layer.top
                layer_name = "global_avg_pool"
                layer_param = caffe_net.Layer_param(
                    name = layer_name,
                    type="Pooling",
                    bottom = bottoms,
                    top = tops
                )
                layer_param.pool_param(
                    type='AVE',
                    global_pooling=True,
                    kernel_size=None,
                    stride=None
                )
                if i + 2 <= len(layers) - 1:
                    del self.cnet.layers()[i+2].bottom[:]
                    self.cnet.layers()[i+2].bottom.extend(tops)

                self.cnet.remove_layer_by_name(layer.name)
                self.cnet.remove_layer_by_name(next_layer.name)
                self.cnet.add_layer(layer_param, before=layers[i+2].name)
                
                logger.info("Reduction layer pair is removed.")

    def replace_with_flatten(self):        
        # check Reduction layer
        for i, layer in enumerate(self.cnet.layers()):
            if not layer.type == 'Reduction': continue
            layers = self.cnet.layers()
            if i + 1 > len(layers) - 1: continue
            next_layer =  layers[i+1]
            if next_layer.type == 'Reduction':
                # remove the two reduction layers
                # and add an average global pooling layer
                bottoms = layer.bottom
                tops = next_layer.top
                layer_name = "flatten"
                layer_param = caffe_net.Layer_param(
                    name = layer_name,
                    type="Flatten",
                    bottom = bottoms,
                    top = tops
                )
                if i + 2 <= len(layers) - 1:
                    del self.cnet.layers()[i+2].bottom[:]
                    self.cnet.layers()[i+2].bottom.extend(tops)

                self.cnet.remove_layer_by_name(layer.name)
                self.cnet.remove_layer_by_name(next_layer.name)
                self.cnet.add_layer(layer_param, before=layers[i+2].name)
                
                logger.info("Reduction layer pair is removed.")                

    def remove_global_avg_pooling(self):
        """
        DPU does not support global average pooling, so we remove
        global average pooling along with FC
        """
        remove_list = list()
        k = -1
        for i, layer in enumerate(self.cnet.layers()):
            if layer.type == 'Pooling' and layer.pooling_param.global_pooling: 
                k = i    
            if k > 0 and i >= k:
                # import ipdb; ipdb.set_trace()
                remove_list.append(layer.name)
        for layer_name in remove_list:
            self.cnet.remove_layer_by_name(layer_name)        