# TextCNN.TorchSharp

TorchSharp implements TextCNN, a translation with reference to the code https://github.com/649453932/Chinese-Text-Classification-Pytorch. After proofreading, the training, testing, and prediction results are consistent with the pytorch implementation.

NET for some time, suffering from the lack of deep learning modules, and see that the official seems to be preparing to access to TorchSharp, so consider a preview of the text classification for a port from pytorch, although TorchSharp said it is implemented in accordance with the naming rules of pytorch, but the actual use, found that there are still some missing and So we have to write code while debugging python to make sure the word list, data input and output, and vector conversion data are correct.

The road of ML.NET is still far, I hope Microsoft can invest more power in deep learning, the current roadmap, ML.NET access to TorchSharp to the end of 2022, my God, this progress is really too slow.

----

TorchSharp实现TextCNN，参照代码 https://github.com/649453932/Chinese-Text-Classification-Pytorch 进行的翻译，经过校对，训练、测试、预测的结果与pytorch实现一致。

使用ML.NET一段时间，苦于没有深度学习模块，看到官方似乎正在准备接入到TorchSharp，所以考虑提前预习一下，将文本分类进行一次从pytorch的移植，虽然TorchSharp说是按照pytorch的命名规则进行实现，但实际使用中，发现还是有一些缺失和定义不一致的API，所以要边调试python边写代码，来保证词表、数据输入输出和向量转换数据正确。

ML.NET的路还很远，希望微软能多投入点力量在深度学习方面，目前的线路图来看，ML.NET接入TorchSharp要到2022年底，我的天，这进度真的太慢了。

