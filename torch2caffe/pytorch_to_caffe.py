from logging import exception
from .Translate.Translog import TransLog
import logging
from .Translate.Replacer import Replacer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()




class pytorch2caffe(object):
    def __init__(self, model):
        self.layer_names = dict()
        self.translog = TransLog()
        self.model = model
        for name, layer in model.named_modules():
            self.layer_names[layer] = name

        self.replacer = Replacer()    
        self.replacer.replace_functions(self.translog, self.layer_names) 

    def set_input(
        self,
        input_var,
        source="",
        root_folder="",
        batch_size=1,
        new_height=224,
        new_width=224,
    ):
        self.translog.set_input(
            input_var, source, root_folder, batch_size, new_height, new_width
        )
        self.input_var = input_var

    def trans_net(self, name="TransferedPytorchModel"):
        # import ipdb; ipdb.set_trace()
        self.translog.set_net_name(name)
        x = self.model.forward(self.input_var)
        # self.translog.set_softmaxwithloss(x)

        # post check
        self.translog.post_check()

        # torch to caffe names
        logger.debug("printing torch to caffe names: ")
        for key, value in self.translog.torch_to_caffe_names.items():
            logger.debug(f"torch name: {key}  --> caffe name: {value}")
            
        self.replacer.place_back()

    def save_prototxt(self, save_name):
        logger.info("saving prototxt to: " + save_name + " ...")
        self.translog.cnet.save_prototxt(save_name)

    def save_caffemodel(self, save_name):
        logger.info("saving caffemodel to: " + save_name + " ...")
        self.translog.cnet.save(save_name)

    def save_torch2caffe_names_json(self, save_name):
        logger.info("saving torch2caffe names to: " + save_name + " ...")
        import json

        with open(save_name, "w") as f:
            json.dump(f, self.trans_net.torch_to_caffe_names, sort_keys=True, indent=2)
