using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using TorchSharp.Data;
using TorchSharp.torchvision;
using System.Diagnostics;
using TextCNN.TorchSharp.Utils;
using TextCNN.TorchSharp.Model;

namespace TextCNN.TorchSharp
{
    public class TrainEval
    {
        public static void init_network(TextCNNModel model, string method= "xavier", string exclude= "embedding", int seed= 123)
        {
            foreach (var item in model.named_parameters())
            {
                if (!item.name.Contains(exclude))
                {
                    if (item.name.Contains("weight"))
                    {
                        if (method == "xavier")
                        { 
                            nn.init.xavier_normal_(item.parameter);
                        }
                        else if (method == "kaiming")
                        {
                            nn.init.kaiming_normal_(item.parameter);
                        }
                        else
                        {
                            nn.init.normal_(item.parameter);
                        }
                    }
                    else if (item.name.Contains("bias"))
                    {
                        nn.init.constant_(item.parameter, 0);
                    } 
                    else
                    {
                        //none
                    }
                }
            }
        }


        public static void train(ConfigBase config, Module model, DatasetIterater train_iter, DatasetIterater dev_iter, DatasetIterater test_iter)
        {
            //start_time = time.time();
            model.Train();
            var optimizer = torch.optim.Adam(model.parameters(), learningRate: config.learning_rate);


            //# 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
            //# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

            var total_batch = 0;  //# 记录进行到多少batch
            var dev_best_loss = float.PositiveInfinity;
            var last_improve = 0;  //# 记录上次验证集loss下降的batch数
            var flag = false;  //# 记录是否很久没有效果提升
                               //var writer = SummaryWriter(log_dir = config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()));

            for (int epoch = 0; epoch < config.num_epochs; epoch++)
            {
                Console.WriteLine($"Epoch [{epoch + 1}/{ config.num_epochs}]");
                var iter_count = 0;
                foreach (var item in train_iter)
                {
                    iter_count++;
                    var tensorItem = (((Tensor, Tensor), Tensor))item;
                    var labels = tensorItem.Item2;
                    var outputs = model.forward(tensorItem.Item1.Item1, tensorItem.Item1.Item2);
                    model.zero_grad();
                    var loss = cross_entropy_loss()(outputs, labels);
                    loss.backward();
                    optimizer.step();

                    if (total_batch % 100 == 0)
                    {
                        var trueArray = labels.cpu();
                        var predic = torch.max(outputs, 1).indexes.cpu();
                        Debug.WriteLine(trueArray.ToString(true));
                        Debug.WriteLine(predic.ToString(true));
                        var train_acc = Metrics.accuracy_score_tensor(trueArray, predic);
                        var (dev_acc, dev_loss, _) = evaluate(config, model, dev_iter);
                        var improve = "";
                        if (dev_loss < dev_best_loss)
                        {
                            dev_best_loss = (float)dev_loss;
                            //TODO torch.save(model.state_dict(), config.save_path);
                            improve = "*";
                            last_improve = total_batch;
                        } 
                        else
                        {
                            improve = "";
                        }
                        var msg = $"Iter: {total_batch},  Train Loss: {(loss.item<float>()):N5},  Train Acc: {(train_acc * 100):N3}%,  Val Loss: {dev_loss:N5},  Val Acc: {(dev_acc * 100):N3}%,  Time: {5} {6}";
                        //print(msg.format(total_batch, loss.item<double>, train_acc, dev_loss, dev_acc, time_dif, improve))
                        Console.WriteLine(msg);
                        Debug.WriteLine(dev_acc);
                        Debug.WriteLine(dev_loss);
                        model.Train();
                    }
                    total_batch += 1;

                    if (total_batch - last_improve > config.require_improvement)
                    {


                        //# 验证集loss超过1000batch没下降，结束训练
                        Console.WriteLine("No optimization for a long time, auto-stopping...");
                        flag = true;
                        break;
                    }
                }

                Console.WriteLine(iter_count);

                if (flag)
                {
                    break;
                }

            }
            model.save(config.save_path);
            test(config, model, test_iter);
            
        }

        public static (double, double, string) evaluate(ConfigBase config, Module model, DatasetIterater data_iter, bool test = false)
        {
            model.Eval();
            Tensor loss_total = tensor(0);
            var predict_all = new List<long>();
            var labels_all = new List<long>();

            //double total_acc = 0.0;
            //long total_count = 0;

            long data_iter_count = 0;

            using (torch.no_grad())
            {
                foreach(((Tensor, Tensor), Tensor) item in data_iter)
                {
                    var texts = item.Item1;
                    var labels = item.Item2;
                    
                    var outputs = model.forward(texts.Item1, texts.Item2);
                    var loss = cross_entropy_loss()(outputs, labels);
                    loss_total += loss;
                    var labelsArray = labels.cpu().data<long>().ToArray();
   
                    var (_, pre) = torch.max(outputs, 1);
                    var predic = pre.cpu().data<long>().ToArray();
                    var degre = (int)(pre[0]);

                    labels_all.AddRange(labelsArray);
                    predict_all.AddRange(predic);

                    //自行计算？
                    //total_acc += (outputs.argmax(1) == labels).sum().to(torch.CPU).item<long>();
                    //total_count += labels.size(0);
                    data_iter_count++;

                }

            } 

            var acc = Metrics.accuracy_score(labels_all, predict_all);
            if (test)
            {
                var report = Metrics.classification_report(labels_all, predict_all, config.class_list);
                //var confusion = metrics.confusion_matrix(labels_all, predict_all); //先不实现
                return (acc, (loss_total / (double)data_iter_count).ToDouble(), report);
            }

            return (acc, (loss_total / (double)data_iter_count).ToDouble(), "");
        }


        public static void test(ConfigBase config, Module model, DatasetIterater test_iter)
        {
            model = (TextCNNModel)model.load(config.save_path);
            model.Eval(); 
            var (test_acc, test_loss, test_report) = evaluate(config, model, test_iter, true);
            Console.WriteLine(test_report);
            Console.WriteLine("测试完毕");
        }

        /// <summary>
        /// 预测
        /// </summary>
        /// <param name="config"></param>
        /// <param name="model"></param>
        /// <param name="vocab"></param>
        /// <param name="predString">要预测的文本</param>
        public static void pred(ConfigBase config, Module model, Dictionary<string, int> vocab, string predString)
        {
            model = (TextCNNModel)model.load(config.save_path);
            model.Eval();

            Func<string, string[]> tokenizer = (a) => {
                List<string> vs = new List<string>();
                foreach (var word in a)
                {
                    vs.Add(word.ToString());
                }
                return vs.ToArray();
            };
            var t = CharUtils.StringToTensor(predString, tokenizer, vocab, config.pad_size);
            t = torch.tensor(t.data<long>().ToArray(), new long[] { 1, 64 }).to(config.device);
            t = model.forward(t);
            var (_, pre) = torch.max(t, 1);
            var predic = pre.cpu().data<long>().ToArray();

            Console.WriteLine($"Pred label:{predic[0]}");

            Console.WriteLine("预测完毕");
        }

    }



}
