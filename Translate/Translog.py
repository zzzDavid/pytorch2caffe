from Caffe import caffe_net

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
        self.blobs = dict()
        self.layer_count = dict()
        self.blob_count = dict()
        self._blobs_data =[]
        self.cnet = caffe_net.Caffemodel('')
        self.debug =True
        self._logger = logger.getChild(self.__class__.__name__)
        self.torch_to_caffe_names = dict()
        self.name = "default_name"

    def set_net_name(self, name):
        self.cnet.net.name = name
        self.name = name    

    def set_input(self, input_var):
        """
        input_var: torch.Tensor object
        """
        top_blobs = self.add_blobs([input_var], name='data', with_num=False)
        layer_name = self.add_layer(name='input')
        logger.info(f'---> layer name: {layer_name}, top blobs: {top_blobs}') 
        layer_param = caffe_net.Layer_param(name=layer_name, type='Input',
                                            top=top_blobs)
        # import ipdb; ipdb.set_trace()
        dims = list()
        dims.extend([1, 3, 224, 224])
        layer_param.input_param(dims)
        self.cnet.add_layer(layer_param)

    def add_layer(self, name='layer', torch_name=None):
        if name in self.layers:
            return self.layers[name]
        if name not in self.layer_count.keys():
            self.layer_count[name] =0
        self.layer_count[name] += 1
        name = '{}_{}_{}'.format(self.name, name, self.layer_count[name])
        self.layers[name] = name # ? what is it doing
        # if self.debug:
        #     print("{} was added to layers".format(self.layers[name]))
        self.torch_to_caffe_names[torch_name] = self.layers[name]
        return self.layers[name]

    def add_blobs(self, blobs, name='blob', with_num=True):
        rst = []
        for i, blob in enumerate(blobs):
            self._blobs_data.append(blob)  # to block the memory address be rewrited
            blob_id = int(id(blob))
            if name not in self.blob_count.keys():
                self.blob_count[name] = 0
            self.blob_count[name] += 1
            if with_num:
                rst.append('{}{}'.format(name, self.blob_count[name]))
            else:
                rst.append('{}'.format(name))
            # self._logger.info(f'blob: {rst[-1]} was added to the list')
            self.blobs[blob_id] = rst[-1]
        return rst

    def blobs(self, var):
        _var = id(var)
        try:
            if self.debug:
                print("{}:{} getting".format(_var, self.blobs[_var]))
            return self.blobs[_var]
        except:
            self._logger.exception("CANNOT FIND blob {}".format(_var))
            return 'None'