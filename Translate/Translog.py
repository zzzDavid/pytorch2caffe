from Caffe import caffe_net

## setup logger
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



class TransLog(object):

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

    def add_layer(self, name='layer', torch_name=None):
        if name in self.layers:
            return self.layers[name]
        if name not in self.layer_count.keys():
            self.layer_count[name] =0
        self.layer_count[name] += 1
        name = '{}_{}_{}'.format(self._id, name, self.layer_count[name])
        self.layers[name] = name # ? what is it doing
        if self.debug:
            print("{} was added to layers".format(self.layers[name]))
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
            if self.debug:
                self._logger.info("{}:{} was added to blobs".format(blob_id, rst[-1]))
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