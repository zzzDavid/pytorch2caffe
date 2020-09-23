# Pytorch2Caffe

This tool is for converting PyTorch CNN models to Caffe network configuration files (prototxt) and parameter files (caffemodel).


## API

```python
trans_net()
save_prototxt()
save_caffemodel()
```

## Done

-[*] Added None blob checking
-[*] Fixed input layer issues: Vitis requires input blob is named `data`, and it must be a input layer, not input field.


## TODO

-[*] Tensor operation is not supported yet. Is there any method to reload the operator and add more arguments in its argument list? 