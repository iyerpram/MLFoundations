using System;
using System.Linq;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
// --- Logistic Regression with TensorFlow.NET ---
//
// Logistic regression = linear regression + sigmoid.
// Sigmoid makes output between 0 and 1 → probability of "pass".

var random = new Random(42);
const int sampleCount = 500;

// Arrays for study hours, prior scores, and pass/fail labels.
var hours = new float[sampleCount];
var scores = new float[sampleCount];
var passed = new float[sampleCount];

for (int i = 0; i < sampleCount; i++)
{
    // Random study hours (0–10) and prior score (30–100).
    float hoursStudied = (float)(random.NextDouble() * 10);
    float previousScore = random.Next(30, 100);

    // Weighted formula to decide pass/fail.
    float signal = (hoursStudied * 6) + (previousScore * 0.5f) - 30;

    // Add noise so data isn’t too perfect.
    bool didPass = signal + (float)(random.NextDouble() * 20 - 10) > 20;

    // Save values.
    hours[i] = hoursStudied;
    scores[i] = previousScore;
    passed[i] = didPass ? 1f : 0f; // 1 = pass, 0 = fail
}

// --- Normalization ---
// Scale values to 0–1 range so training is smoother.
float[] Normalize(float[] values, out float min, out float max)
{
    min = values.Min();
    max = values.Max();
    float minValue = min;
    float maxValue = max;
    return values.Select(v => maxValue > minValue ? (v - minValue) / (maxValue - minValue) : 0f).ToArray();
}

var normHours = Normalize(hours, out float hoursMin, out float hoursMax);
var normScores = Normalize(scores, out float scoresMin, out float scoresMax);

// Combine into feature array: [hours, scores].
var features = new float[sampleCount, 2];
for (int i = 0; i < sampleCount; i++)
{
    features[i, 0] = normHours[i];
    features[i, 1] = normScores[i];
}

// --- Train/Test Split ---
// 80% train, 20% test.
int splitIndex = (int)(sampleCount * 0.8);
int testCount = sampleCount - splitIndex;

// Helper functions to slice arrays into TensorFlow format.
NDArray Slice2D(float[,] data, int start, int count)
{
    var slice = new float[count, 2];
    for (int r = 0; r < count; r++)
        for (int c = 0; c < 2; c++)
            slice[r, c] = data[start + r, c];
    return np.array(slice);
}

NDArray Slice1D(float[] data, int start, int count) =>
    np.array(data.Skip(start).Take(count).ToArray());

// Training and test sets.
var xTrain = Slice2D(features, 0, splitIndex);
var yTrain = Slice1D(passed, 0, splitIndex);
var xTest = Slice2D(features, splitIndex, testCount);
var yTest = Slice1D(passed, splitIndex, testCount);

// --- Model ---
// Sequential = simple stack of layers.
var model = keras.Sequential();

// Input: 2 features (hours, scores).
model.add(keras.layers.InputLayer(input_shape: new Shape(2)));

// Dense layer: 1 neuron + sigmoid.
// Sigmoid chosen → gives probability (0–1).
model.add(keras.layers.Dense(1, activation: "sigmoid"));

// --- Compile ---
// Optimizer: Adam → fast and reliable.
// Loss: Binary Crossentropy → best for yes/no problems.
// Metric: Accuracy → % correct predictions.
model.compile(
    optimizer: keras.optimizers.Adam(0.1f),
    loss: keras.losses.BinaryCrossentropy(),
    metrics: new[] { "accuracy" });

model.summary();

// --- Training ---
// Train with batch size 32, run 200 times (epochs).
Console.WriteLine("Training...");
model.fit(xTrain, yTrain, batch_size: 32, epochs: 200, verbose: 0);

// --- Evaluation ---
// Test on unseen data.
Console.WriteLine("=== Evaluation on held-out test set ===");
model.evaluate(xTest, yTest);

// --- Predictions ---
// Try model on new students.
Console.WriteLine("=== Sample predictions ===");
var sampleStudents = new (float hours, float score)[]
{
    (1, 40),   // low effort → likely fail
    (8, 85),   // high effort → likely pass
    (4, 60)    // middle case
};

foreach (var (studentHours, studentScore) in sampleStudents)
{
    // Normalize inputs same way as training.
    float normStudentHours = (studentHours - hoursMin) / (hoursMax - hoursMin);
    float normStudentScore = (studentScore - scoresMin) / (scoresMax - scoresMin);

    var input = np.array(new float[,] { { normStudentHours, normStudentScore } });

    // Predict probability of passing.
    float probability = model.predict(input).numpy().ToArray<float>()[0];

    // If >0.5 → PASS, else FAIL.
    string label = probability > 0.5f ? "PASS" : "FAIL";

    Console.WriteLine($"  {studentHours}h studied, prior score {studentScore} " +
        $"-> predicted: {label} (probability: {probability:P1})");
}
