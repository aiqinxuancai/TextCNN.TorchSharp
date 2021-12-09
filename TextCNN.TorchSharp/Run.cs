using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TextCNN.TorchSharp.Utils;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;

namespace TextCNN.TorchSharp
{
    public class Run
    {
        internal static void Main(string[] args)
        {
            var config = new TextCNNConfig();
            np.random.seed(1);
            torch.random.manual_seed(1); 
            torch.cuda.manual_seed_all(1);
            //TODO torch.backends.cudnn.deterministic = true;  //# 保证每次结果一样

            var (vocab, train_data, dev_data, test_data) = CharUtils.build_dataset(config);
            var train_iter = CharUtils.build_iterator(train_data, config);
            var dev_iter = CharUtils.build_iterator(dev_data, config);
            var test_iter = CharUtils.build_iterator(test_data, config);
            //time_dif = get_time_dif(start_time);

            config.n_vocab = vocab.Count();
            var model = new TextCNNModel(config).to(config.device);
            TrainEval.init_network(model);

            TrainEval.train(config, model, train_iter, dev_iter, test_iter);

            //TrainEval.test(config, model, test_iter);

            //TrainEval.pred(config, model, vocab, "我你你你你傻傻");
 
            Console.WriteLine(train_iter);
        }
    }
}