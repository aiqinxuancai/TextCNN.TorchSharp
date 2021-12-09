using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace TextCNN.TorchSharp.Model
{
    public abstract class ConfigBase
    {
        public string dataset = string.Empty;
        public string embedding = string.Empty;
        public string model_name = string.Empty;


        public string train_path = string.Empty;            //# 训练集
        public string dev_path = string.Empty;              //# 验证集
        public string test_path = string.Empty;             //# 测试集

        public string[] class_list = new string[0];         //# 类别名单
        public int num_classes;

        public string vocab_path = string.Empty;            //# 词表
        public string save_path = string.Empty;             //# 模型训练结果
        public string log_path = string.Empty;

        public int n_vocab;                                 //# 词表大小 运行时复制

        public int batch_size;                              //# mini-batch大小
        public int pad_size;                                //# 每句话处理成的长度(短填长切)


        public Device device = torch.CPU;

        public int require_improvement = 1000;              //# 若超过1000batch效果还没提升，则提前结束训练
        public int num_epochs = 20;                         //# epoch数

        public double learning_rate = 1e-3;                 //# 学习率
    }
}
