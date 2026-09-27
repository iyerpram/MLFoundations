using System;
using System.Linq;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

// --- Linear Regression with TensorFlow.NET ---
//
// Linear regression = 2 inputs → 1 output, no activation.
// Output is just a straight line fit: y = w1*x1 + w2*x2 + b.
// Difference from logistic regression: logistic adds sigmoid to turn output into probability.

var random = new Random(42);
const int sampleCount = 500;

// Arrays to hold house features and prices.
var sizes = new float[sampleCount];       // house size in sqft
var bedroomsArr = new float[sampleCount]; // number of bedrooms
var prices = new float[sampleCount];      // house price

for (int i = 0; i < sampleCount; i++)
{
    // Randomly generate house size (600–4000 sqft) and bedrooms (1–5).
    float sizeSqft = random.Next(600, 4000);
    float bedrooms = random.Next(1, 6);

    // Add some random noise so data isn’t too perfect.
    float noise = (float)(random.NextDouble() * 20_000 - 10_000);

    // Price formula: base + size*120 + bedrooms*15000 + noise.
    float price = 50_000f + (sizeSqft * 120f) + (bedrooms * 15_000f) + noise;

    sizes[i] = sizeSqft;
    bedroomsArr[i] = bedrooms;
    prices[i] = price;
}

// --- Normalization ---
// Method: Normalize values to 0–1 range.
// Reason: Keeps features on same scale so training is stable.
float[] Normalize(float[] values, out float min, out float max)
{
    min = values.Min();
    max = values.Max();
    float minValue = min;
    float maxValue = max;
    return values.Select(v => maxValue > minValue ? (v - minValue) / (maxValue - minValue) : 0f).ToArray();
}

var normSizes = Normalize(sizes, out float sizeMin, out float sizeMax);
var normBedrooms = Normalize(bedroomsArr, out float bedroomsMin, out float bedroomsMax);
var normPrices = Normalize(prices, out float priceMin, out float priceMax);

// Combine features into 2D array: [size, bedrooms].
var features = new float[sampleCount, 2];
for (int i = 0; i < sampleCount; i++)
{
    features[i, 0] = normSizes[i];
    features[i, 1] = normBedrooms[i];
}

// --- Train/Test Split ---
// 80% training, 20% testing.
int splitIndex = (int)(sampleCount * 0.8);
int testCount = sampleCount - splitIndex;

// Helper: slice 2D data into NDArray for TensorFlow.
NDArray Slice2D(float[,] data, int start, int count)
{
    var slice = new float[count, 2];
    for (int r = 0; r < count; r++)
        for (int c = 0; c < 2; c++)
            slice[r, c] = data[start + r, c];
    return np.array(slice);
}

// Helper: slice 1D data into NDArray.
NDArray Slice1D(float[] data, int start, int count) =>
    np.array(data.Skip(start).Take(count).ToArray());

// Training and test sets.
var xTrain = Slice2D(features, 0, splitIndex);
var yTrain = Slice1D(normPrices, 0, splitIndex);
var xTest = Slice2D(features, splitIndex, testCount);

// --- Model ---
// Sequential = stack of layers.
var model = keras.Sequential();

// Input: 2 features (size, bedrooms).
model.add(keras.layers.InputLayer(input_shape: new Shape(2)));

// Dense layer: 1 neuron, no activation.
// Reason: Linear regression needs raw numeric output (not squashed).
model.add(keras.layers.Dense(1));

// --- Compile ---
// Optimizer: Adam → adjusts learning rate automatically, faster training.
// Loss: Mean Squared Error (MSE).
// Reason: MSE is standard for regression → penalizes large errors more.
model.compile(
    optimizer: keras.optimizers.Adam(0.05f),
    loss: keras.losses.MeanSquaredError());

model.summary();

// --- Training ---
// Train with batch size 32, run 200 epochs (passes through data).
Console.WriteLine("Training...");
model.fit(xTrain, yTrain, batch_size: 32, epochs: 200, verbose: 0);

// --- Predictions on Test Set ---
// Model outputs normalized prices → convert back to real prices.
var testPredictionsNorm = model.predict(xTest).numpy().ToArray<float>();
var actualTestPrices = prices.Skip(splitIndex).Take(testCount).ToArray();
var predictedTestPrices = testPredictionsNorm.Select(p => p * (priceMax - priceMin) + priceMin).ToArray();

// --- Evaluation ---
// R-squared: how well line fits data (closer to 1 = better).
// RMSE: average error size in dollars.
double meanActual = actualTestPrices.Average();
double ssTotal = actualTestPrices.Sum(a => Math.Pow(a - meanActual, 2));
double ssResidual = actualTestPrices.Zip(predictedTestPrices, (a, p) => Math.Pow(a - p, 2)).Sum();
double rSquared = 1 - (ssResidual / ssTotal);
double rmse = Math.Sqrt(ssResidual / testCount);

Console.WriteLine("=== Evaluation ===");
Console.WriteLine($"R-squared: {rSquared:F4}  (closer to 1.0 = better fit)");
Console.WriteLine($"Root Mean Squared Error: ${rmse:N0}");

// --- Sample Predictions ---
// Try model on new houses.
Console.WriteLine("=== Sample predictions ===");
var sampleHouses = new (float size, float bedrooms)[]
{
    (1500, 3),
    (2500, 4),
    (900, 1)
};

foreach (var (size, bedrooms) in sampleHouses)
{
    // Normalize inputs same way as training.
    float normSize = (size - sizeMin) / (sizeMax - sizeMin);
    float normBedroomsVal = (bedrooms - bedroomsMin) / (bedroomsMax - bedroomsMin);

    var input = np.array(new float[,] { { normSize, normBedroomsVal } });

    // Predict normalized price → convert back to real price.
    float predictedNorm = model.predict(input).numpy().ToArray<float>()[0];
    float predictedPrice = predictedNorm * (priceMax - priceMin) + priceMin;

    Console.WriteLine($"  {size} sqft, {bedrooms} bed -> predicted ${predictedPrice:N0}");
}
