using NumSharp;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using static TorchSharp.torch;

namespace TextCNN.TorchSharp.Utils
{
    internal class Metrics
    {
        public static double accuracy_score(IList<long> t, IList<long> p)
        {
            var count = 0;
            for (int i = 0; i < p.Count; i++)
            {
                if (p[i] == t[i])
                {
                    count++;
                }
            }
            return (double)count / (double)p.Count;
        }

        public static double accuracy_score_tensor(Tensor t, Tensor p)
        {
            return accuracy_score(t.data<long>().ToList(), p.data<long>().ToList());
        }

        public static string classification_report(IList<long> t, IList<long> pred, string[] target_names)
        {
            var trueClass = new Dictionary<long, long>(); //正确预测的数量

            var predAllClass = pred.GroupBy(x => x).ToDictionary(x => x.Key, x => x.Count()); //所有预测数量统计
            var trueAllClass = t.GroupBy(x => x).ToDictionary(x => x.Key, x => x.Count()); //所有预测数量统计

            //正确预测数
            for (int i = 0; i < pred.Count; i++)
            {
                var key = (int)pred[i];
                if (pred[i] == t[i])
                {
                    if (trueClass.ContainsKey(key))
                    {
                        trueClass[key] = trueClass[key] + 1;
                    }
                    else
                    {
                        trueClass[key] = 1;
                    }
                }
            }

            //计算预测数
            var result = string.Empty;

            result += $"target\t";
            result += $"precision\t";
            result += $"recall\t";
            result += $"f1-score\t";
            result += $"support\n";

            for (int i = 0; i < target_names.Length; i++)
            {
                var predAllCount = predAllClass.ContainsKey(i) ? (double)predAllClass[i] : 0.0;
                var trueAllCount = trueAllClass.ContainsKey(i) ? (double)trueAllClass[i] : 0.0;

                trueClass.TryGetValue(i, out var trueCount);

                var precision = trueCount / predAllCount;
                var recall = trueCount / trueAllCount;

                result += $"{target_names[i]}\t";
                result += $"{precision:N4}\t"; //precision    
                result += $"{recall:N4}\t"; //recall  
                result += $"{((2 * precision * recall) / (precision + recall)):N4}\t"; //F1  
                result += $"{predAllCount}\n"; //F1  
            }
            return result;
        }
    }
}
