# Pytorch2Caffe

This tool is for converting PyTorch CNN models to Caffe network configuration files (prototxt) and parameter files (caffemodel).


## API

```python
trans_net()
save_prototxt()
save_caffemodel()
```

由于此项目的代码太过菠萝菠萝哒，现在正在重写。需要解决的问题：
1. None blob, unknown blob check
2. Input layer/Loss layer
暂时想到这些