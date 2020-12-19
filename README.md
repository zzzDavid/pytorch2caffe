# Pytorch2Caffe

This tool is for converting PyTorch CNN models to Caffe network configuration files (prototxt) and parameter files (caffemodel).

## Install

To install pytorch2caffe, go to its folder, then:
```
$ pip install .
```
If you would like to install in develop mode:
```
$ pip install -e .
```

## API

```python

from pytorch2caffe.converter import pytorch2caffe_converter

def your_function(model):
    # set up input layer information
    input = Variable(torch.ones([1, 3, 416, 416]))
    source = '/path/to/image_list.txt'
    root_foler = 'path/to/image_folder'
    batch_size = 1
    new_height = 416
    new_width = 416

    # initialize a pytorch2caffe converter object
    torch2caffe = pytorch2caffe_converter(model)

    # set input
    torch2caffe.set_input(input, source, root_folder, batch_size, new_height, new_width)

    # translate 
    torch2caffe.trans_net('resnet50')

    # save results
    torch2caffe.save_prototxt('./resnet50.prototxt')
    torch2caffe.save_caffemodel('./resnet50.caffemodel')
    # optional
    torch2caffe.save_torch2caffe_names_json('./torch2caffe_names.json')

```

## Done

- [x]  Added None blob checking

- [x] Fixed input layer issues: Vitis requires input blob is named `data`, and it must be an input layer, not input field.

- [x] Add input data source and loss layer

- [x] Tensor operations are supported

- [x] Remove `Reduction` layer, and check `x.mean(3).mean(2)` which is supposed to be a global average pooling layer in caffe

## TODO


- [ ] Supported layer checking

- [ ] Global average pooling support: `torch.nn.AdaptiveAvgPooling2d(1,1)`



## Note

1. Issues with name correspondence

We have a `dict()` in class `Translog` to record the torch function names and their corresponding caffe layer names. But we don't have that information for tensor operations. 

For example, `+=` can convert into `Eltwise` in Caffe, but it does not have torch name since it is not a function.

2. ~DPU IP does not support Global Average Pooling~

~Global average pooling is moved to CPU, resulting in multiple kernels.~

It's actually supported. A working example:
```prototxt
layer {
  name: "global_avg_pool"
  type: "Pooling"
  bottom: "batch_norm_blob49"
  top: "mean_blob2"
  pooling_param {
    pool: AVE
    global_pooling: true
  }
}
```

3. The input output channel restriction for depthwise convolution

The Xilinx DPU Compiler actually has restrictions for depthwise conv's i/o channel number. If the number exceeds the maximum value, the compiler would trigger error:
```
[VAI_C-BACKEND][Check Failed: kernel_param * input_channel_group <= weights_buf_depth][/usr/local/anaconda3/conda-bld/vaic_1584632209107/work/dnnc/submodules/asicv2com/src/Operator/OperatorDptConv.cpp:31][DATA_OUTRANGE][Data value is out of range!] 
```
We don't know the actual value of `weights_buf_depth` because the C code is not open-sourced.
So I found out the maximum value with binary search:
- for 7x7 depth conv, the maixmum value is 656,
- for 5x5 depth conv, the maximum value is 1296,
- for 3x3 depth conv, the maximum value is 2000+, we don't have to worry about it.
