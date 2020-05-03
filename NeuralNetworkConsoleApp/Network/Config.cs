using System.Collections.Generic;



namespace NeuralNetwork.Network
{
    public class Config
    {
        public Config(int inputNeuronsCount, int[] hiddenNeuronsCount, int outputNeuronsCount, double learningRate)
        {
            InputNeuronsCount = inputNeuronsCount;
            OutputNeuronsCount = outputNeuronsCount;

            HiddenNeuronsCount = new List<int>();
            HiddenNeuronsCount.AddRange(hiddenNeuronsCount);

            LearningRate = learningRate;
        }



        public int InputNeuronsCount { get; }
        public int OutputNeuronsCount { get; }
        public List<int> HiddenNeuronsCount { get; } /// List т.к. скрытых слоев может быть много
        public double LearningRate { get; }
    }
}
