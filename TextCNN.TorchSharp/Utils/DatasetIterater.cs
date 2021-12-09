using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace TextCNN.TorchSharp.Utils
{

    public class DatasetIterater : IEnumerable
    {
        private int batch_size;
        private List<(long[], int, int)> batches;
        private int n_batches;
        private bool residue;
        private int index;
        private Device device;

        private int pad_size;

        public DatasetIterater(List<(long[], int, int)> dataset, int batch_size, int pad_size, Device device)
        {
            //def __init__(self, batches, batch_size, device):
            this.batch_size = batch_size;
            this.pad_size = pad_size;
            this.batches = dataset;
            this.n_batches = (int)Math.Floor((double)dataset.Count / (double)batch_size) + 1; // batch_size
            this.residue = false;//  # 记录batch数量是否为整数

            if (this.batches.Count % this.n_batches != 0)
            {
                this.residue = true;
            }

            this.index = 0;
            this.device = device;
        }

        private ((Tensor, Tensor), Tensor) _to_tensor(List<(long[], int, int)> datas)
        {
            List<long> x = new List<long>();
            datas.ForEach((a) => {
                x.AddRange(a.Item1);
            });

            List<long> y = new List<long>();
            datas.ForEach((a) => y.Add(a.Item2));

            List<long> seq_len = new List<long>();
            datas.ForEach((a) => seq_len.Add(a.Item3));

            Tensor itemX = torch.tensor(x, new long[] { datas.Count, this.pad_size }).to(this.device); //TODO 
            Tensor itemY = torch.tensor(y).to(this.device);
            Tensor itemSeqLen = torch.tensor(seq_len).to(this.device);
            return ((itemX, itemSeqLen), itemY);
        }


        public IEnumerator GetEnumerator()
        {
            //返回的是
            //this.batches[this.index * this.batch_size: (this.index + 1) * this.batch_size];
            Console.WriteLine($"All batches Count {n_batches}");
            for (int i = 0; i < this.n_batches; i++)
            {
                var takeCount = this.index * this.batch_size + this.batch_size > this.batches.Count ? this.batches.Count - (this.index * this.batch_size) : this.batch_size;
                var batches = this.batches.GetRange(this.index * this.batch_size, takeCount);
                Console.WriteLine($"GetRange{this.index},{this.index * this.batch_size},{takeCount},{batches.Count}");
                this.index += 1;

                yield return this._to_tensor(batches);
                if (this.residue && this.index == this.n_batches)
                {
                    break;
                }

            }
            this.index = 0;
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            // Invoke IEnumerator<string> GetEnumerator() above.
            return GetEnumerator();
        }
    }

}
