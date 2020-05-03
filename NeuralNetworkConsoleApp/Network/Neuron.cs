using System;
using System.Collections.Generic;



namespace NeuralNetwork.Network
{
    public class Neuron
    {
        public Neuron(int inputsCount, NeuronType type)
        {
            Type = type;
            Weights = new List<double>();
            InputSignals = new List<double>();

            Random random = new Random();
            for (int i = 0; i < inputsCount; i++)
            {
                if (Type == NeuronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    // инициализация весов случайными значениями
                    Weights.Add(random.NextDouble());
                }
                InputSignals.Add(0);
            }
        }


        public List<double> Weights { get; }
        public List<double> InputSignals { get; }
        public NeuronType Type { get; }
        public double OutputSignal { get; private set; }
        public double Delta { get; private set; }



        public double FeedForward(List<double> signals)
        {
            double sum = 0;
            if (signals.Count == Weights.Count)
            {
                for (int i = 0; i < signals.Count; i++)
                {
                    InputSignals[i] = signals[i];
                    sum += InputSignals[i] * Weights[i]; /// Сумма произведений сигналов и весов
                }

                if (Type != NeuronType.Input)
                {
                    OutputSignal = CountSigmoid(sum); /// Сумма проходит через сигмоиду. outputsignal принадлежит промежутку [0,1]
                }
                else
                {
                    OutputSignal = sum; /// Т.к. сигнал на входном нейроне только один и его вес всегда равен 1
                }
            }
            return OutputSignal;
        }



        // Занчение сигмоиды при данном x
        private double CountSigmoid(double x)
        {
            double sigmoid = 1 / (1 + Math.Exp(-x));
            return sigmoid;
        }



        // Занчение производной сигмоиды при данном x
        private double CountSigmoidDerivative(double x)
        {
            double sigmoid = CountSigmoid(x);
            double sigmoidDerivative = sigmoid * (1 - sigmoid);
            return sigmoidDerivative;
        }



        // Вычисление новых весов
        public void Recalculate(double difference, double learningRate)
        {
            if (Type != NeuronType.Input)
            {
                Delta = difference * CountSigmoidDerivative(OutputSignal); /// Дельта = разница (ошибка) * производная сигмоида от x

                for (int i = 0; i < Weights.Count; i++)
                {
                    double currentWeight = Weights[i];
                    double currentInputSignal = InputSignals[i];

                    double recalculatedWeight = currentWeight - currentInputSignal * Delta * learningRate; /// Новый вес = изначальный вес - сигнал * дельта * кэффициент
                    Weights[i] = recalculatedWeight;
                }
            }
        }



        public override string ToString()
        {
            string neuronStr = string.Format("neuron type: {0}\nWeights:", Type);

            for (int i = 0; i < Weights.Count; i++)
            {
                neuronStr += string.Format("\nid: {0}, value: {1}", i, Weights[i]);
            }
            return neuronStr;
        }
    }



    public enum NeuronType
    {
        Input,
        Normal,
        Output
    }
}
