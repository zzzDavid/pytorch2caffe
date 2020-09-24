# Pytorch2Caffe

This tool is for converting PyTorch CNN models to Caffe network configuration files (prototxt) and parameter files (caffemodel).


## API

```python
trans_net()
save_prototxt()
save_caffemodel()
```

## Done

- [x]  Added None blob checking

- [x] Fixed input layer issues: Vitis requires input blob is named `data`, and it must be a input layer, not input field.

- [x] Add input data source and loss layer

- [x] Tensor operations are supported

## TODO


- [ ] Supported layer checking

- [ ] Global average pooling support: `torch.nn.AdaptiveAvgPooling2d(1,1)`


## Note

1. Issues with name correspondence

We have a `dict()` in class `Translog` to record the torch function names and their corresponding caffe layer names. But we don't have that information for tensor operations. 

For example, `+=` can convert into `Eltwise` in Caffe, but it does not have torch name since it is not a function.