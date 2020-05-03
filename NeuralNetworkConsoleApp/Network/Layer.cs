using System.Collections.Generic;



namespace NeuralNetwork.Network
{
    public class Layer
    {
        public Layer(List<Neuron> neurons, NeuronType neuronsType)
        {
            Neurons = new List<Neuron>();

            for (int i = 0; i < neurons.Count; i++)
            {
                if (neurons[i].Type == neuronsType)
                {
                    Neurons.Add(neurons[i]);
                }
            }

            NeuronsCount = Neurons.Count;
            NeuronsType = neuronsType;
        }



        public List<Neuron> Neurons { get; }
        public int NeuronsCount { get; }
        public NeuronType NeuronsType { get; }



        // Возвращает список всех выходных значений нейронов для передачи на следующий уровень (слой)
        public List<double> GetOutputSignals()
        {
            List<double> outputSignals = new List<double>();
            foreach(Neuron current in Neurons)
            {
                outputSignals.Add(current.OutputSignal);
            }

            return outputSignals;
        }



        public override string ToString()
        {
            string layerStr = string.Format("layer children (neurons) type: {0}, neurons conunt: {1}", NeuronsType, NeuronsCount);
            return layerStr;
        }
    }
}
