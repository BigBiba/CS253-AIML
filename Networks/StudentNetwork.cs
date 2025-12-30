using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    class Neuron
    {
        public static Func<double, double> activationFunction;
        public static Func<double, double> activationFunctionDerivative;

        public int id;
        public double Output;
        public int layer;

        public double error;


        public double[] weightsToPrevLayer;

        public void setInput(double input)
        {
            if (layer == 0)
            {
                Output = input;
                return;
            }

            Output = activationFunction(input);
        }

        public Neuron(int id, int layer, int prevLayerCapacity, Random random)
        {
            this.id = id;
            this.layer = layer;
            this.error = 0;


            if (layer == -1)
                Output = 1;


            if (layer < 1)
            {
                weightsToPrevLayer = null;
            }
            else
            {
                weightsToPrevLayer = new double[prevLayerCapacity + 1];
                for (int i = 0; i < weightsToPrevLayer.Length; i++)
                    weightsToPrevLayer[i] = random.NextDouble() * 2 - 1;
            }
        }
    }

    public class StudentNetwork : BaseNetwork
    {
        private const double learningRate = 0.15;

        private readonly Neuron biasNeuron;
        private readonly List<Neuron[]> layers;

        private readonly Func<double[], double[], double> lossFunction;
        private readonly Func<double, double, double> lossFunctionDerivative;

        public Stopwatch stopWatch = new Stopwatch();

        public StudentNetwork(int[] structure)
        {
            if (structure == null || structure.Length < 3)
                throw new ArgumentException("Нужно минимум 3 слоя: входной, скрытый, выходной.");


            lossFunction = (output, aim) =>
            {
                double res = 0;
                for (int i = 0; i < aim.Length; i++)
                {
                    double d = aim[i] - output[i];
                    res += d * d;
                }
                return res * 0.5;
            };


            lossFunctionDerivative = (output, aim) => aim - output;


            Neuron.activationFunction = x => 1.0 / (1.0 + Math.Exp(-x));
            Neuron.activationFunctionDerivative = y => y * (1.0 - y);

            Random random = new Random();

            biasNeuron = new Neuron(0, -1, -1, random);

            layers = new List<Neuron[]>();
            int id = 1;

            for (int layer = 0; layer < structure.Length; layer++)
            {
                layers.Add(new Neuron[structure[layer]]);

                for (int i = 0; i < structure[layer]; i++)
                {
                    if (layer == 0)
                        layers[layer][i] = new Neuron(id, layer, -1, random);
                    else
                        layers[layer][i] = new Neuron(id, layer, structure[layer - 1], random);

                    id++;
                }
            }
        }


        public void forwardPropagation(double[] input, bool parallel = false)
        {
            if (input.Length != layers[0].Length)
                throw new ArgumentException("Длина входного вектора не совпадает с входным слоем.");

            for (int i = 0; i < layers[0].Length; i++)
                layers[0][i].setInput(input[i]);

            for (int layer = 1; layer < layers.Count; layer++)
            {
                int curLayer = layer;
                int prevLayer = layer - 1;

                Action<int> computeNeuron = neuronIndex =>
                {
                    var neuron = layers[curLayer][neuronIndex];
                    var w = neuron.weightsToPrevLayer;

                    double sum = 0;

                    sum += biasNeuron.Output * w[0];

                    for (int i = 1; i < w.Length; i++)
                        sum += layers[prevLayer][i - 1].Output * w[i];

                    neuron.setInput(sum);
                };


                Parallel.For(0, layers[curLayer].Length, computeNeuron);

            }
        }

        public void backwardPropagation(Sample sample)
        {
            var aim = sample.outputVector;

            for (int l = 0; l < layers.Count; l++)
                for (int n = 0; n < layers[l].Length; n++)
                    layers[l][n].error = 0;

            int last = layers.Count - 1;
            if (aim.Length != layers[last].Length)
                throw new ArgumentException("Длина outputVector не совпадает с выходным слоем.");

            for (int i = 0; i < layers[last].Length; i++)
                layers[last][i].error = lossFunctionDerivative(layers[last][i].Output, aim[i]);

            for (int layer = layers.Count - 1; layer >= 1; layer--)
            {
                var curLayer = layers[layer];
                var prevLayer = layers[layer - 1];

                foreach (var neuron in curLayer)
                {
                    double delta = neuron.error * Neuron.activationFunctionDerivative(neuron.Output);

                    neuron.weightsToPrevLayer[0] += learningRate * delta * biasNeuron.Output;

                    for (int i = 1; i < neuron.weightsToPrevLayer.Length; i++)
                    {
                        int prevIndex = i - 1;

                        double oldWeight = neuron.weightsToPrevLayer[i];

                        prevLayer[prevIndex].error += delta * oldWeight;

                        neuron.weightsToPrevLayer[i] += learningRate * delta * prevLayer[prevIndex].Output;
                    }

                }
            }
        }

        private double TrainOnSample(Sample sample, bool parallel)
        {
            forwardPropagation(sample.input, parallel);
            double loss = lossFunction(layers.Last().Select(n => n.Output).ToArray(), sample.outputVector);
            backwardPropagation(sample);
            return loss;
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int cnt = 0;

            while (true)
            {
                cnt++;
                forwardPropagation(sample.input, parallel);

                double loss = lossFunction(layers.Last().Select(n => n.Output).ToArray(), sample.outputVector);
                if (loss <= acceptableError)
                    return cnt;

                backwardPropagation(sample);
            }
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            stopWatch.Restart();

            double meanError = double.PositiveInfinity;

            for (int epoch = 0; epoch < epochsCount; epoch++)
            {
                double sumError = 0.0;

                foreach (var sample in samplesSet.samples)
                    sumError += TrainOnSample(sample, parallel);

                meanError = sumError / samplesSet.Count;

                OnTrainProgress((double)epoch / epochsCount, meanError, stopWatch.Elapsed);

                if (meanError <= acceptableError)
                    break;
            }

            OnTrainProgress(1.0, meanError, stopWatch.Elapsed);
            stopWatch.Stop();

            return meanError;
        }

        public override void Save(string filePath)
        {
            using (var writer = new BinaryWriter(File.OpenWrite(filePath)))
            {
                // Сохраняем структуру сети
                writer.Write(layers.Count);
                foreach (var layer in layers)
                {
                    writer.Write(layer.Length);
                }

                // Сохраняем веса
                for (int layer = 1; layer < layers.Count; layer++)
                {
                    for (int neuron = 0; neuron < layers[layer].Length; neuron++)
                    {
                        var weights = layers[layer][neuron].weightsToPrevLayer;
                        writer.Write(weights.Length);
                        foreach (var weight in weights)
                        {
                            writer.Write(weight);
                        }
                    }
                }
            }
        }

        public override void Load(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("Файл модели не найден");

            using (var reader = new BinaryReader(File.OpenRead(filePath)))
            {
                // Загружаем структуру (для проверки)
                int layerCount = reader.ReadInt32();
                int[] structure = new int[layerCount];
                for (int i = 0; i < layerCount; i++)
                {
                    structure[i] = reader.ReadInt32();
                }

                // Проверяем совпадение структуры
                if (structure.Length != layers.Count)
                    throw new InvalidOperationException("Структура модели не совпадает");

                for (int i = 0; i < structure.Length; i++)
                {
                    if (structure[i] != layers[i].Length)
                        throw new InvalidOperationException($"Слой {i} не совпадает по размеру");
                }

                // Загружаем веса
                for (int layer = 1; layer < layers.Count; layer++)
                {
                    for (int neuron = 0; neuron < layers[layer].Length; neuron++)
                    {
                        int weightCount = reader.ReadInt32();
                        var weights = new double[weightCount];
                        for (int w = 0; w < weightCount; w++)
                        {
                            weights[w] = reader.ReadDouble();
                        }
                        layers[layer][neuron].weightsToPrevLayer = weights;
                    }
                }
            }
        }

        protected override double[] Compute(double[] input)
        {
            if (input.Length != layers[0].Length)
                throw new ArgumentException("Длина входа не совпадает с входным слоем.");

            forwardPropagation(input, false);
            return layers.Last().Select(n => n.Output).ToArray();
        }
    }
}