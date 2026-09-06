using MLFoundations.NeuralNetwork;
using Xunit;

namespace MLFoundations.Tests;

public class SimpleNeuralNetworkTests
{
    [Fact]
    public void Predict_BeforeTraining_ReturnsValueBetweenZeroAndOne()
    {
        var network = new SimpleNeuralNetwork(inputSize: 2, hiddenSize: 4, seed: 1);

        var output = network.Predict([0.5, 0.5]);

        Assert.InRange(output, 0.0, 1.0);
    }

    [Fact]
    public void TrainOne_ReducesErrorOverRepeatedCalls()
    {
        var network = new SimpleNeuralNetwork(inputSize: 2, hiddenSize: 4, learningRate: 0.5, seed: 1);

        var firstError = network.TrainOne([1, 0], expected: 1);
        double lastError = firstError;
        for (int i = 0; i < 500; i++)
            lastError = network.TrainOne([1, 0], expected: 1);

        // Training repeatedly on the same single example should drive its
        // error down substantially - this is a basic sanity check that
        // backprop is actually updating weights in the right direction.
        Assert.True(lastError < firstError,
            $"Expected error to decrease (first: {firstError}, last: {lastError})");
    }

    [Theory]
    [InlineData(0, 0, 0)]
    [InlineData(0, 1, 1)]
    [InlineData(1, 0, 1)]
    [InlineData(1, 1, 0)]
    public void Network_LearnsXor_AfterSufficientTraining(double a, double b, double expected)
    {
        // This is the key test: it proves the network can learn a function
        // that is NOT linearly separable, which is the entire point of
        // this project. A logistic regression model cannot pass this test
        // for all four cases simultaneously - the hidden layer is what
        // makes it possible.
        var network = new SimpleNeuralNetwork(inputSize: 2, hiddenSize: 4, learningRate: 0.5, seed: 42);

        double[][] inputs = [[0, 0], [0, 1], [1, 0], [1, 1]];
        double[] expectedOutputs = [0, 1, 1, 0];

        for (int epoch = 0; epoch < 10_000; epoch++)
            for (int i = 0; i < inputs.Length; i++)
                network.TrainOne(inputs[i], expectedOutputs[i]);

        var prediction = network.Predict([a, b]);
        var rounded = prediction > 0.5 ? 1 : 0;

        Assert.Equal((int)expected, rounded);
    }
}
