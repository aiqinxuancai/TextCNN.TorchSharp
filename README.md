# TextCNN.TorchSharp

TorchSharp implements TextCNN, a translation with reference to the code https://github.com/649453932/Chinese-Text-Classification-Pytorch. After proofreading, the training, testing, and prediction results are consistent with the pytorch implementation.

Seeing that officials seem to be preparing to connect TorchSharp to ML.NET, I considered trying it out in advance.

This project will be a port of TextCNN from pytorch, although TorchSharp is in accordance with pytorch naming rules, but the actual use, found that there are still some missing and inconsistently defined API, so to debug python while writing code to ensure that the word list, data input and output and vector conversion data correct.

The neural network-related libraries in python are very well developed, and libraries other than pytorch are often used in the implementation to process vectors, which must then be reimplemented by referring to the implementation of this library. This is also a more painful problem in the translation process.

ML.NET still has a long way to go, I hope Microsoft can invest more power in deep learning, the current line chart shows that ML.NET access to TorchSharp until the end of 2022, my God, this progress is really too slow.

----

TorchSharp实现TextCNN，参照代码 https://github.com/649453932/Chinese-Text-Classification-Pytorch 进行的翻译，经过校对，训练、测试、预测的结果与pytorch实现一致。

看到官方似乎正在准备把TorchSharp接入到ML.NET，所以考虑提前试用一下。

此项目将TextCNN从pytorch的移植，虽然TorchSharp是按照pytorch的命名规则，但实际使用中，发现还是有一些缺失和定义不一致的API，所以要边调试python边写代码，来保证词表、数据输入输出和向量转换数据正确。

python的神经网络相关库非常完善，在实现中经常用到pytorch以外的库对向量进行处理，这时候就必须参考这个库的实现来重新实现。这也是在翻译过程中比较痛苦的问题。

ML.NET的路还很远，希望微软能多投入点力量在深度学习方面，目前的线路图来看，ML.NET接入TorchSharp要到2022年底，我的天，这进度真的太慢了。

## 快速开始

相关的数据可使用 https://github.com/649453932/Chinese-Text-Classification-Pytorch 库中的，注意对比代码，文件名可能有略微不同。
