using Microsoft.ML;
using Microsoft.ML.Data;

// --- Logistic Regression: binary classification ---
//
// Task: predict whether a student passes (true/false) based on hours
// studied and their previous test score. Like the linear regression
// example, this uses synthetic data generated from a known rule, so we can
// sanity-check the model's behavior against ground truth.

var mlContext = new MLContext(seed: 42);

var trainingData = GenerateSyntheticStudentData(count: 500);
var dataView = mlContext.Data.LoadFromEnumerable(trainingData);
var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

var pipeline = mlContext.Transforms
    .Concatenate("Features", nameof(StudentData.HoursStudied), nameof(StudentData.PreviousScore))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(StudentData.Passed)));

Console.WriteLine("Training logistic regression model...");
var model = pipeline.Fit(split.TrainSet);

var predictions = model.Transform(split.TestSet);
var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: nameof(StudentData.Passed));

Console.WriteLine();
Console.WriteLine("=== Evaluation ===");
Console.WriteLine($"Accuracy:  {metrics.Accuracy:P1}");
Console.WriteLine($"F1 Score:  {metrics.F1Score:P1}");
Console.WriteLine($"AUC:       {metrics.AreaUnderRocCurve:F4}");
Console.WriteLine();
// NOTE: accuracy alone can be misleading on imbalanced datasets - F1/AUC
// matter more when pass/fail counts aren't roughly equal. This synthetic
// dataset is reasonably balanced, but this is worth calling out explicitly
// since it's a common real-world pitfall (see the IncidentsAi-adjacent
// discussion on imbalanced classes for a domain where this really bites).

var predictionEngine = mlContext.Model.CreatePredictionEngine<StudentData, PassPrediction>(model);

Console.WriteLine("=== Sample predictions ===");
foreach (var student in new[]
{
    new StudentData { HoursStudied = 1, PreviousScore = 40 },
    new StudentData { HoursStudied = 8, PreviousScore = 85 },
    new StudentData { HoursStudied = 4, PreviousScore = 60 }
})
{
    var prediction = predictionEngine.Predict(student);
    Console.WriteLine($"  {student.HoursStudied}h studied, prior score {student.PreviousScore} " +
        $"-> predicted: {(prediction.Passed ? "PASS" : "FAIL")} (probability: {prediction.Probability:P1})");
}

static List<StudentData> GenerateSyntheticStudentData(int count)
{
    var random = new Random(42);
    var data = new List<StudentData>(count);

    for (int i = 0; i < count; i++)
    {
        float hoursStudied = (float)(random.NextDouble() * 10);
        float previousScore = random.Next(30, 100);

        // "True" underlying rule: a weighted combination crossing a threshold,
        // with a little randomness so it's not perfectly separable (more realistic).
        float signal = (hoursStudied * 6) + (previousScore * 0.5f) - 30;
        bool passed = signal + (random.NextDouble() * 20 - 10) > 20;

        data.Add(new StudentData { HoursStudied = hoursStudied, PreviousScore = previousScore, Passed = passed });
    }

    return data;
}

public class StudentData
{
    public float HoursStudied { get; set; }
    public float PreviousScore { get; set; }
    public bool Passed { get; set; }
}

public class PassPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Passed { get; set; }
    public float Probability { get; set; }
}
