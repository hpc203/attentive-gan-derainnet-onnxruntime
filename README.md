训练源码在https://github.com/MaybeShewill-CV/attentive-gan-derainnet
，它是CVPR2018的文章，虽然时间有些久远，但是我没写过图像去雨的模型部署，因此
编写了一次。起初使用opencv-dnn做推理部署，加载onnx文件没有报错，但是在forward函数
那里出错了，因而使用onnxruntime做推理引擎了。

onnx文件在百度云盘，
链接：https://pan.baidu.com/s/1l3MNbfSHB6mNit49P5-aqg 
提取码：oqoc
