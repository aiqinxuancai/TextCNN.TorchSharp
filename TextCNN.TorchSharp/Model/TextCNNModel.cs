using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TextCNN.TorchSharp.Model;
using TorchSharp;
using TorchSharp.Modules;

using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TextCNN.TorchSharp
{
    public class TextCNNConfig : ConfigBase
    {
        public Tensor embedding_pretrained = null; //torch.tensor(np.load(dataset + "/data/" + embedding)["embeddings"].astype(typeof(float)).ToArray<float>());//"float32"
                                                                                //# 预训练词向量
        public float dropout = 0.5f;                                            //# 随机失活

        public long embed = 300;                                                //# 字向量维度
        public int[] filter_sizes = { 2, 3, 4 };                                //# 卷积核尺寸
        public int num_filters = 256;                                           //# 卷积核数量(channels数)

        public TextCNNConfig()
        {
            dataset = ".";
            embedding = "embedding_SougouNews.npz";

            model_name = "TextCNN";
            train_path = dataset + "/data/train.txt";                                //# 训练集
            dev_path = dataset + "/data/dev.txt";                                    //# 验证集
            test_path = dataset + "/data/test.txt";                                  //# 测试集

            class_list = File.ReadAllLines(dataset + "/data/class.txt");             //# 类别名单
            num_classes = File.ReadAllLines(dataset + "/data/class.txt").Length;     //# 类别数

            vocab_path = dataset + "/data/vocab.json";                               //# 词表
            save_path = dataset + "/saved_dict/" + model_name + ".model";            //# 模型训练结果
            log_path = dataset + "/log/" + model_name;

            n_vocab = 0;

            require_improvement = 1000;                                             //# 若超过1000batch效果还没提升，则提前结束训练
            num_epochs = 20;                                                        //# epoch数

            batch_size = 128;                                                       //# mini-batch大小
            pad_size = 64;                                                          //# 每句话处理成的长度(短填长切)

            learning_rate = 1e-3;

            device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;            //# 设备
        }

    }


    /*
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
     */
    public class TextCNNModel : Module
    {
        private Embedding embedding;
        private ModuleList convs;
        private Linear fc;
        private Dropout dropout;
        

        public TextCNNModel(TextCNNConfig config) : base("TextClassification")
        {
            embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx: config.n_vocab - 1);
            var modules = new List<Module>();
            foreach (var item in config.filter_sizes) {
                //modules.Add(nn.Conv2d(1, config.num_filters, new long[] { item, config.embed })); //TODO
                modules.Add(nn.Conv2d(1, config.num_filters, (item, config.embed), (1, 1))); 
            }

            convs = nn.ModuleList(modules.ToArray());
            dropout = nn.Dropout(config.dropout);
            fc = nn.Linear(config.num_filters * config.filter_sizes.Length, config.num_classes);

            RegisterComponents();
        }

        public Tensor conv_and_pool(Tensor x, Module conv)
        {
            x = relu(conv.forward(x)).squeeze(3);
            x = max_pool1d(x, x.size(2)).squeeze(2);
            return x;
        }

        public override Tensor forward(Tensor t)
        {
            t = embedding.forward(t);
            t = t.unsqueeze(1);
            List<Tensor> inputs = new List<Tensor>();
            foreach (var conv in convs)
            {
                inputs.Add(conv_and_pool(t, conv));
            }
            t = torch.cat(inputs, 1);
            t = dropout.forward(t);
            t = fc.forward(t);
            return t;
        }

        public override Tensor forward(Tensor input, Tensor offsets)
        {
            /*
                def forward(self, x):
                    out = self.embedding(x[0])
                    out = out.unsqueeze(1)
                    out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
                    out = self.dropout(out)
                    out = self.fc(out)
                    return out
             */
            var t = embedding.forward(input);
            t = t.unsqueeze(1);
            List<Tensor> inputs = new List<Tensor>();
            foreach (var conv in convs)
            {
                inputs.Add(conv_and_pool(t, conv));
            }
            t = torch.cat(inputs, 1);
            t = dropout.forward(t);
            t = fc.forward(t);
            return t;
        }

        public new TextCNNModel to(Device device)
        {
            base.to(device);
            return this;
        }
    }
}
