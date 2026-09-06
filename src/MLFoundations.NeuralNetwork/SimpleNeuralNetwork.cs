namespace MLFoundations.NeuralNetwork;

// A minimal feedforward neural network: one hidden layer, sigmoid
// activation, trained via backpropagation and plain gradient descent.
// No external ML library - this exists specifically to demonstrate
// understanding of the mechanics (forward pass, error, backprop, weight
// update), not to be a production-grade implementation.
//
// Architecture: 2 inputs -> N hidden neurons -> 1 output
public sealed class SimpleNeuralNetwork
{
    private readonly int _inputSize;
    private readonly int _hiddenSize;
    private readonly double _learningRate;

    private double[,] _weightsInputHidden;
    private double[] _biasHidden;
    private double[] _weightsHiddenOutput;
    private double _biasOutput;

    public SimpleNeuralNetwork(int inputSize, int hiddenSize, double learningRate = 0.5, int? seed = null)
    {
        _inputSize = inputSize;
        _hiddenSize = hiddenSize;
        _learningRate = learningRate;

        var random = seed.HasValue ? new Random(seed.Value) : new Random();

        _weightsInputHidden = new double[inputSize, hiddenSize];
        _biasHidden = new double[hiddenSize];
        _weightsHiddenOutput = new double[hiddenSize];

        // Small random initialization - breaks symmetry so hidden neurons
        // don't all learn the same thing.
        for (int i = 0; i < inputSize; i++)
            for (int h = 0; h < hiddenSize; h++)
                _weightsInputHidden[i, h] = (random.NextDouble() * 2 - 1);

        for (int h = 0; h < hiddenSize; h++)
        {
            _biasHidden[h] = (random.NextDouble() * 2 - 1);
            _weightsHiddenOutput[h] = (random.NextDouble() * 2 - 1);
        }

        _biasOutput = random.NextDouble() * 2 - 1;
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
    private static double SigmoidDerivative(double sigmoidOutput) => sigmoidOutput * (1 - sigmoidOutput);

    /// <summary>Runs a forward pass and returns the predicted output (between 0 and 1).</summary>
    public double Predict(double[] inputs) => ForwardPass(inputs).output;

    private (double[] hiddenActivations, double output) ForwardPass(double[] inputs)
    {
        var hiddenActivations = new double[_hiddenSize];
        for (int h = 0; h < _hiddenSize; h++)
        {
            double sum = _biasHidden[h];
            for (int i = 0; i < _inputSize; i++)
                sum += inputs[i] * _weightsInputHidden[i, h];
            hiddenActivations[h] = Sigmoid(sum);
        }

        double outputSum = _biasOutput;
        for (int h = 0; h < _hiddenSize; h++)
            outputSum += hiddenActivations[h] * _weightsHiddenOutput[h];

        return (hiddenActivations, Sigmoid(outputSum));
    }

    /// <summary>Trains on a single example via backpropagation and returns the squared error.</summary>
    public double TrainOne(double[] inputs, double expected)
    {
        var (hiddenActivations, output) = ForwardPass(inputs);

        double outputError = expected - output;
        double outputDelta = outputError * SigmoidDerivative(output);

        var hiddenDeltas = new double[_hiddenSize];
        for (int h = 0; h < _hiddenSize; h++)
        {
            double hiddenError = outputDelta * _weightsHiddenOutput[h];
            hiddenDeltas[h] = hiddenError * SigmoidDerivative(hiddenActivations[h]);
        }

        // Update hidden -> output weights
        for (int h = 0; h < _hiddenSize; h++)
            _weightsHiddenOutput[h] += _learningRate * outputDelta * hiddenActivations[h];
        _biasOutput += _learningRate * outputDelta;

        // Update input -> hidden weights
        for (int i = 0; i < _inputSize; i++)
            for (int h = 0; h < _hiddenSize; h++)
                _weightsInputHidden[i, h] += _learningRate * hiddenDeltas[h] * inputs[i];
        for (int h = 0; h < _hiddenSize; h++)
            _biasHidden[h] += _learningRate * hiddenDeltas[h];

        return outputError * outputError;
    }
}
