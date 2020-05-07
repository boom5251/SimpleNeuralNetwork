using NeuralNetwork.Network;
using NeuralNetworkConsoleApp.Data;
using System;
using System.Collections.Generic;



namespace NeuralNetworkConsoleApp
{
    class Program
    {
        static void Main(string[] args)
        {
            Config config = new Config(13, new int[] { 10, 5 }, 1, 0.1);
            Network network = new Network(config);


            //Результат - Пациент болен - 1
            // Пациент Здоров - 0

            // Неправильная температура T
            // Хороший возраст A
            // Курит S
            // Правильно питается F

            double[,] datasetArr = new double[,]
            {
                // T  A  S  F  Res
                { 0, 0, 0, 0, 0 },
                { 0, 0, 0, 1, 0 },
                { 0, 0, 1, 0, 1 },
                { 0, 0, 1, 1, 0 },
                { 0, 1, 0, 0, 0 },
                { 0, 1, 0, 1, 0 },
                { 0, 1, 1, 0, 1 },
                { 0, 1, 1, 1, 0 },
                { 1, 0, 0, 0, 1 },
                { 1, 0, 0, 1, 1 },
                { 1, 0, 1, 0, 1 },
                { 1, 0, 1, 1, 1 },
                { 1, 1, 0, 0, 1 },
                { 1, 1, 0, 1, 0 },
                { 1, 1, 1, 0, 1 },
                { 1, 1, 1, 1, 1 }
            };

            Type[] types = new Type[]
            {
                typeof(double),
                typeof(double),
                typeof(double),
                typeof(double),
                typeof(double)
            };

            network.LearnNetwork(datasetArr, types, SignalsScalingOptions.Scale, datasetArr.GetLength(1) - 1, 10000, out Dataset mainDataset);


            for (int i = 0; i < mainDataset.RowsCount; i++)
            {
                Neuron neuron = network.StartNetwork(new List<double>(mainDataset.GetRow(i, true)));
                Console.WriteLine(datasetArr[i, datasetArr.GetLength(1) - 1] + " " + Math.Round(neuron.OutputSignal, 5));
            }
        }
    }
}