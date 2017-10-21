using System;
using System.Linq;
using AForge.Neuro;
using AForge.Neuro.Learning;

namespace NetFest_NN
{
    class Program
    {
        static void Main(string[] args)
        {
            // load and prepare data
            var trainingData = Data.Load(@"PerformanceData\final_training_data.csv", ClassificationColumn.Last);
            trainingData.Normalize();
            var validationData = Data.Load(@"PerformanceData\final_test_data.csv", ClassificationColumn.Last);
            validationData.Normalize();

            // NN parameters for teaching
            var learningRate = 0.01;
            var sigmoidAlphaValue = 0.5;
            var neuronCounts = new[] {100, 50, trainingData.ClassCount};
            var layerCounts = new[] { trainingData.InputCount }.Concat(neuronCounts).ToArray();

            // construct the network and teacher
            var activationFunction = new BipolarSigmoidFunction(sigmoidAlphaValue);
            var network = new ActivationNetwork(activationFunction, trainingData.InputCount, neuronCounts);
            network.Randomize();
            var teacher = new ResilientBackpropagationLearning(network)
            {
                LearningRate = learningRate,
            };

            var name = $"{teacher}-{string.Join("-", layerCounts)}-{learningRate}-{sigmoidAlphaValue}"
                .Replace(".", "_")
                .Replace(",", "_");

            var iteration = 0;
            const double learningErrorLimit = 0.05;
            Console.WriteLine($"Starting to train a network: {name}");
            while (true)
            {
                var error = teacher.RunEpoch(trainingData.Inputs, trainingData.Outputs)/trainingData.SampleCount;
                // check if we need to stop
                if (error <= learningErrorLimit)
                {
                    Console.WriteLine($"DONE!!! ---- {name} - iter # {iteration}: error: {error}");
                    break;
                }

                if (++iteration % 100 == 0)
                {
                    Console.WriteLine($"iter # {iteration}: error: {error}");
                }
            }

            Console.ReadKey();
            Console.WriteLine($"Validating on training data for {name}");
            Validate(trainingData, network);
            Console.ReadKey();
            Console.WriteLine($"Validating on testing data for {name}");
            Validate(validationData, network);
            Console.ReadKey();
        }

        private static void Validate(Data validationData, Network network)
        {
            int successCount = 0;
            for (int i = 0; i < validationData.SampleCount; i++)
            {
                var results = new Classification(network.Compute(validationData.Inputs[i]));

                if (Success(validationData.Outputs[i], results.Classifications))
                {
                    successCount++;
                }
            }
            var successRate = (successCount / (double)validationData.SampleCount) * 100.0;
            Console.WriteLine($"Success rate is: {successRate:0.00}%");
        }

        private static bool Success(double[] expected, double[] actual)
        {
            double best = actual.Max();
            int bestIndex = -1;
            for (int i = 0; i < actual.Length; i++)
            {
                if (actual[i] == best)
                    bestIndex = i;
            }

            return expected[bestIndex] > 0;
        }
    }
}
