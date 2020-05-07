using NeuralNetworkConsoleApp.Data;
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



        public double LearnNetwork(double[,] datasetArr, Type[] columnsTypes, SignalsScalingOptions scaling, int resultsColumnIndex, int epochCount, out Dataset dataset)
        {
            // Создание экземпляра класса Dataset, преобразование данных
            dataset = new Dataset(datasetArr, columnsTypes, resultsColumnIndex);


            // Масштабирование данных
            if (scaling == SignalsScalingOptions.Scale)
            {
                ScaleData(dataset.Data);
            }
            else if (scaling == SignalsScalingOptions.Normalize)
            {
                NormalizeData(dataset.Data);
            }


            // Запуск цикла обучения
            double error = 0;

            for (int epoch = 0; epoch < epochCount; epoch++)
            {
                Console.WriteLine("epoch: " + epoch);

                dataset.GetColumnType(resultsColumnIndex, out double[] resultsColumn);

                for (int i = 0; i < dataset.RowsCount; i++)
                {
                    error += StartBackpropagation(resultsColumn[i], new List<double>(dataset.GetRow(i, true)));
                }
            }
            return error / epochCount; /// Средняя квадратичная ошибка
        }



        private double StartBackpropagation(double expectedResult, List<double> inputSignals)
        {
            double difference = 0;

            for (int k = Layers.Count - 1; k >= 0; k--)
            {
                if (k == Layers.Count - 1) /// Последний (первый с конца) слой
                {
                    Neuron resultNeuron = StartNetwork(inputSignals);
                    double actualResult = resultNeuron.OutputSignal;
                    difference = actualResult - expectedResult; /// Разница (ошибка) = изначальный вес - ожидаемый вес

                    foreach (Neuron neuron in Layers[Layers.Count - 1].Neurons)
                    {
                        neuron.Recalculate(difference, Configuration.LearningRate);
                    }
                }
                else if (k < Layers.Count - 1) /// Все остальные слои
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



        // Масштабирование данных
        private double[,] ScaleData(double[,] datasetMatrix)
        {
            double[,] resultMatrix = new double[datasetMatrix.GetLength(0), datasetMatrix.GetLength(1)];

            for (int column = 0; column < datasetMatrix.GetLength(1); column++)
            {
                double min = double.MaxValue;
                double max = double.MinValue;

                for (int row = 0; row < datasetMatrix.GetLength(0); row++)
                {
                    double currentItem = datasetMatrix[row, column];

                    if (currentItem < min) /// Нахождениение минимального значения в столбце
                    {
                        min = currentItem;
                    }
                    if (currentItem > max) /// Нахождениение максимального значения в столбце
                    {
                        max = currentItem;
                    }
                }

                for (int row = 0; row < datasetMatrix.GetLength(0); row++) /// Запись маштабируемых данных в новый двумерный массив 
                {
                    resultMatrix[row, column] = (datasetMatrix[row, column] - min) / (max - min); /// Новое значение = (текущее - минимум) / (максимум - минимум)
                }
            }

            return resultMatrix;
        }



        // Нормализация данных
        private double[,] NormalizeData(double[,] datasetMatrix)
        {
            double[,] resultMatrix = new double[datasetMatrix.GetLength(0), datasetMatrix.GetLength(1)];

            for (int column = 0; column < datasetMatrix.GetLength(1); column++)
            {
                double xSum = 0;
                for (int row = 0; row < datasetMatrix.GetLength(1); row++)
                {
                    xSum += datasetMatrix[row, column]; /// Находим сумму всех значений x
                }

                double avarage = xSum / datasetMatrix.GetLength(1); /// Среднее арифметическое значение столбца ∑x / n


                double diffSum = 0;
                for (int row = 0; row < datasetMatrix.GetLength(0); row++)
                {
                    diffSum += Math.Pow(datasetMatrix[row, column] - avarage, 2); /// Находим суммау разности x и среднего, возведенной в квадрат
                }

                double deviation = Math.Sqrt(diffSum / datasetMatrix.GetLength(0)); /// Стандартное квадратичное отклонение √((∑(x - avr))² / n)


                for (int row = 0; row < datasetMatrix.GetLength(0); row++)
                {
                    resultMatrix[row, column] = (datasetMatrix[row, column] - avarage) / deviation; /// Новое значение сигнала (x - avr) / Δ
                }
            }

            return resultMatrix;
        }
    }



    public enum SignalsScalingOptions
    {
        Scale,
        Normalize,
        None
    }
}
