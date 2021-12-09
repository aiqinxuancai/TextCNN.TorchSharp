using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections;

using static TorchSharp.torch;
using TorchSharp;
using NumSharp;
using TextCNN.TorchSharp.Model;
using Newtonsoft.Json;

namespace TextCNN.TorchSharp.Utils
{ 

    public class CharUtils
    {
        public const int MAX_VOCAB_SIZE = 10000;  //# 词表长度限制
        public const string UNK = "<UNK>";
        public const string PAD = "<PAD>";


        /// <summary>
        /// 生成词表
        /// </summary>
        /// <param name="file_path"></param>
        /// <param name="tokenizer">拆分算法</param>
        /// <param name="max_size">最大词表数</param>
        /// <param name="min_freq">词重复数量需要大于等于的值</param>
        /// <returns></returns>
        public static Dictionary<string, int> build_vocab(string file_path, Func<string, string[]> tokenizer, int max_size = MAX_VOCAB_SIZE, int min_freq = 1)
        {
            Dictionary<string, int> vocab_dic = new Dictionary<string, int>();
            foreach (string line in File.ReadAllLines(file_path))
            {
                var split = line.Split("\t");
                if (split.Length == 2)
                {
                    foreach (var word in tokenizer(split[0]))
                    {
                        vocab_dic[word] = vocab_dic.ContainsKey(word) ? vocab_dic[word] + 1 : 0;
                    }
                }
            }

            //TODO 最大限制

            //排序
            var vocab_dic_order = vocab_dic.OrderByDescending(x => x.Value);

            vocab_dic = new Dictionary<string, int>();
            foreach (var word in vocab_dic_order)
            {
                vocab_dic[word.Key] = vocab_dic.Count;
            }

            //加入未知和填充
            vocab_dic[UNK] = vocab_dic.Count;
            vocab_dic[PAD] = vocab_dic.Count;

            return vocab_dic;

        }

        public static List<(long[], int, int)> load_dataset(string path, Func<string, string[]> tokenizer, Dictionary<string, int> vocab, int pad_size = 32)
        {
            return load_dataset_content(File.ReadAllText(path), tokenizer, vocab, pad_size);
        }

        public static List<(long[], int, int)> load_dataset_content(string contentText, Func<string, string[]> tokenizer, Dictionary<string, int> vocab, int pad_size = 32)
        {
            List<(long[], int, int)> contents = new();

            foreach (string line in contentText.Split("\n"))
            {
                var split = line.Split("\t");
                if (split.Length == 2)
                {
                    var content = split[0];
                    var label = split[1];
                    var token = new List<string>(tokenizer(content));
                    var seq_len = token.Count;
                    if (token.Count < pad_size)
                    {
                        //填充空
                        while (token.Count < pad_size)
                        {
                            token.Add(PAD);
                        }
                    }
                    else
                    {
                        token = token.Take(pad_size).ToList();
                        seq_len = token.Count;
                    }
                    List<long> words_line = new List<long>();
                    foreach (var word in token)
                    {
                        if (vocab.ContainsKey(word))
                        {
                            words_line.Add(vocab[word]);
                        }
                        else
                        {
                            words_line.Add(vocab[UNK]);
                        }
                    }
                    contents.Add((words_line.ToArray(), int.Parse(label), seq_len));
                }
            }
            return contents;
        }


        /// <summary>
        /// 文本使用词表转Tensor
        /// </summary>
        /// <param name="contentText"></param>
        /// <param name="tokenizer"></param>
        /// <param name="vocab"></param>
        /// <param name="pad_size"></param>
        /// <returns></returns>
        public static Tensor StringToTensor(string contentText, Func<string, string[]> tokenizer, Dictionary<string, int> vocab, int pad_size = 32)
        {
            var content = contentText;
            var token = new List<string>(tokenizer(content));
            var seq_len = token.Count;
            if (token.Count < pad_size)
            {
                //填充空
                while (token.Count < pad_size)
                {
                    token.Add(PAD);
                }
            }
            else
            {
                token = token.Take(pad_size).ToList();
                seq_len = token.Count;
            }
            List<long> words_line = new List<long>();
            foreach (var word in token)
            {
                if (vocab.ContainsKey(word))
                {
                    words_line.Add(vocab[word]);
                }
                else
                {
                    words_line.Add(vocab[UNK]);
                }
            }
            return tensor(words_line.ToArray());

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="config"></param>
        /// <returns></returns>
        public static (Dictionary<string, int>, List<(long[], int, int)>, List<(long[], int, int)>, List<(long[], int, int)>) build_dataset(ConfigBase config)
        {
            Func<string, string[]> tokenizer = (a) => {
                List<string> vs = new List<string>();
                foreach (var word in a)
                {
                    vs.Add(word.ToString());
                }
                return vs.ToArray();
            };
            Dictionary<string, int> vocab = new Dictionary<string, int>();
            if (File.Exists(config.vocab_path))
            {
                vocab = JsonConvert.DeserializeObject<Dictionary<string, int>>(File.ReadAllText(config.vocab_path));
            } 
            else
            {
                vocab = build_vocab(config.train_path, tokenizer: tokenizer, max_size: MAX_VOCAB_SIZE, min_freq: 1);
                //存储词表
                File.WriteAllText(config.vocab_path, JsonConvert.SerializeObject(vocab));
            }



            var train = load_dataset(config.train_path, tokenizer, vocab, config.pad_size);
            var dev = load_dataset(config.dev_path, tokenizer, vocab, config.pad_size);
            var test = load_dataset(config.test_path, tokenizer, vocab, config.pad_size);


            //todo

            Console.WriteLine($"Vocab size: {vocab.Count}");

            return (vocab, train, dev, test);
        }

        public static DatasetIterater build_iterator(List<(long[], int, int)> dataset, ConfigBase config)
        {
            var iter = new DatasetIterater(dataset, config.batch_size, config.pad_size, config.device);
            return iter;
        }

    }
}
