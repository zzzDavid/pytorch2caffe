# Pytorch2Caffe

This tool is for converting PyTorch CNN models to Caffe network configuration files (prototxt) and parameter files (caffemodel).


## API

```python
import sys
sys.path.append('/path/to/pytorch2caffe/directory')

from torch2caffe.pycorch_to_caffe import pytorch2caffe

def your_function(model):
    # set up input layer information
    input = Variable(torch.ones([1, 3, 416, 416]))
    source = '/path/to/image_list.txt'
    root_foler = 'path/to/image_folder'
    batch_size = 1
    new_height = 416
    new_width = 416

    # initialize a pytorch2caffe object
    torch2caffe = pytorch2caffe(model)

    # set input
    torch2caffe.set_input(input, source, root_folder, batch_size, new_height, new_width)

    # translate 
    torch2caffe.trans_net('resnet50')

    # save results
    torch2caffe.save_prototxt('./resnet50.prototxt')
    torch2caffe.save_caffemodel('./resnet50.caffemodel')
    torch2caffe.save_torch2caffe_names_json('./torch2caffe_names.json')

```

## Done

- [x]  Added None blob checking

- [x] Fixed input layer issues: Vitis requires input blob is named `data`, and it must be an input layer, not input field.

- [x] Add input data source and loss layer

- [x] Tensor operations are supported

## TODO


- [ ] Supported layer checking

- [ ] Global average pooling support: `torch.nn.AdaptiveAvgPooling2d(1,1)`

- [ ] Remove `Reduction` layer, and check `x.mean(2).mean(3)` which is supposed to be a global average pooling layer in caffe

## Note

1. Issues with name correspondence

We have a `dict()` in class `Translog` to record the torch function names and their corresponding caffe layer names. But we don't have that information for tensor operations. 

For example, `+=` can convert into `Eltwise` in Caffe, but it does not have torch name since it is not a function.
