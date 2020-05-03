using System;
using System.Collections.Generic;



namespace NeuralNetwork.Network
{
    public class Network
    {
        public Network(Config configuration)
        {
            Configuration = configuration;
            Layers = new List<Layer>();

            CreateLayers();
        }

        

        public Config Configuration { get; }
        public List<Layer> Layers { get; }



        private void CreateLayers()
        {
            // Входной слой
            List<Neuron> inputLayerNeurons = new List<Neuron>();
            for (int i = 0; i < Configuration.InputNeuronsCount; i++)
            {
                Neuron neuron = new Neuron(1, NeuronType.Input);
                inputLayerNeurons.Add(neuron);
            }
            Layer inputLayer = new Layer(inputLayerNeurons, NeuronType.Input);
            Layers.Add(inputLayer);


            // Скрытые слои
            for (int k = 0; k < Configuration.HiddenNeuronsCount.Count; k++)
            {
                List<Neuron> hiddenLayerNeurons = new List<Neuron>();
                for (int i = 0; i < Configuration.HiddenNeuronsCount[k]; i++)
                {
                    Neuron neuron = new Neuron(Layers[Layers.Count - 1].NeuronsCount, NeuronType.Normal);
                    hiddenLayerNeurons.Add(neuron);
                }
                Layer hiddenLayer = new Layer(hiddenLayerNeurons, NeuronType.Normal);
                Layers.Add(hiddenLayer);
            }


            // Выходной слой
            List<Neuron> outputLayerNeurons = new List<Neuron>();
            for (int i = 0; i < Configuration.OutputNeuronsCount; i++)
            {
                Neuron neuron = new Neuron(Layers[Layers.Count - 1].NeuronsCount, NeuronType.Output);
                outputLayerNeurons.Add(neuron);
            }
            Layer outputLayer = new Layer(outputLayerNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }



        public Neuron StartNetwork(List<double> startSignals)
        {
            for (int k = 0; k < Layers.Count; k++)
            {
                for (int i = 0; i < Layers[k].NeuronsCount; i++)
                {
                    double signal;
                    Neuron neuron = Layers[k].Neurons[i];
                    List<double> inputSignals = new List<double>();

                    if (k == 0) /// Передача первичных данных к первому слою нейронов (входных)
                    {
                        signal = startSignals[i];
                        inputSignals = new List<double>() { signal };
                    }
                    else if (k > 0)
                    {
                        inputSignals = Layers[k - 1].GetOutputSignals();
                    }

                    neuron.FeedForward(inputSignals);
                }
            }


            if (Configuration.OutputNeuronsCount == 1)
            {
                return Layers[Layers.Count - 1].Neurons[0];
            }
            else
            {
                Neuron maxValueNeuron = Layers[Layers.Count - 1].Neurons[0];
                foreach (Neuron current in Layers[Layers.Count - 1].Neurons)
                {
                    if (current.OutputSignal > maxValueNeuron.OutputSignal)
                    {
                        maxValueNeuron = current;
                    }
                }
                return maxValueNeuron;
            }
        }



        public double LearnNetwork(List<Tuple<double, double[]>> dataset, int epochCount)
        {
            double error = 0;

            for (int i = 0; i < epochCount; i++)
            {
                Console.WriteLine("epoch: " + i);
                foreach (var data in dataset)
                {
                    error += CreateBackpropagation(data.Item1, new List<double>(data.Item2));
                }
            }
            return error / epochCount; /// Средняя квадратичная ошибка
        }



        private double CreateBackpropagation(double expectedResult, List<double> inputSignals)
        {
            double difference = 0;

            for (int k = Layers.Count - 1; k >= 0; k--)
            {
                if (k == Layers.Count - 1) // Последний (первый с конца) слой
                {
                    Neuron resultNeuron = StartNetwork(inputSignals);
                    double actualResult = resultNeuron.OutputSignal;
                    difference = actualResult - expectedResult; /// Разница (ошибка) = изначальный вес - ожидаемый вес

                    foreach (Neuron neuron in Layers[Layers.Count - 1].Neurons)
                    {
                        neuron.Recalculate(difference, Configuration.LearningRate);
                    }
                }
                else if (k < Layers.Count - 1) // Все остальные слои
                {
                    Layer currentLayer = Layers[k];
                    Layer previosLayer = Layers[k + 1]; /// С конца

                    for (int i = 0; i < currentLayer.NeuronsCount; i++)
                    {
                        Neuron currentNeuron = currentLayer.Neurons[i];

                        for (int j = 0; j < previosLayer.NeuronsCount; j++)
                        {
                            Neuron previousNeuron = previosLayer.Neurons[j];

                            difference = previousNeuron.Weights[i] * previousNeuron.Delta; /// Разница (ошибка) = изначальный вес * дельта
                            currentNeuron.Recalculate(difference, Configuration.LearningRate);
                        }
                    }
                }
            }

            return Math.Pow(difference, 2);
        }
    }
}
