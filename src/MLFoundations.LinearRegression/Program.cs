using Microsoft.ML;
using Microsoft.ML.Data;

// --- Linear Regression: predicting a continuous value ---
//
// Task: predict a house's price from its size (sqft) and bedroom count.
// This is a synthetic dataset (generated below with a known underlying
// formula plus noise) rather than a real housing dataset, specifically so
// the "ground truth" relationship is known and we can sanity-check the
// model actually learned something close to it.

var mlContext = new MLContext(seed: 42);

var trainingData = GenerateSyntheticHousingData(count: 500);
var dataView = mlContext.Data.LoadFromEnumerable(trainingData);

var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

var pipeline = mlContext.Transforms
    .Concatenate("Features", nameof(HouseData.SizeSqft), nameof(HouseData.Bedrooms))
    .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: nameof(HouseData.Price)));

Console.WriteLine("Training linear regression model...");
var model = pipeline.Fit(split.TrainSet);

var predictions = model.Transform(split.TestSet);
var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: nameof(HouseData.Price));

Console.WriteLine();
Console.WriteLine("=== Evaluation ===");
Console.WriteLine($"R-squared: {metrics.RSquared:F4}  (closer to 1.0 = better fit)");
Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError:F2}");
Console.WriteLine();

// Try a few concrete examples
var predictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePricePrediction>(model);

Console.WriteLine("=== Sample predictions ===");
foreach (var house in new[]
{
    new HouseData { SizeSqft = 1500, Bedrooms = 3 },
    new HouseData { SizeSqft = 2500, Bedrooms = 4 },
    new HouseData { SizeSqft = 900, Bedrooms = 1 }
})
{
    var prediction = predictionEngine.Predict(house);
    Console.WriteLine($"  {house.SizeSqft} sqft, {house.Bedrooms} bed -> predicted ${prediction.PredictedPrice:N0}");
}

// Generates data from a known formula (price = base + perSqft*size + perBedroom*bedrooms + noise)
// so we can judge whether the model's learned coefficients are in a sane ballpark.
static List<HouseData> GenerateSyntheticHousingData(int count)
{
    var random = new Random(42);
    var data = new List<HouseData>(count);

    for (int i = 0; i < count; i++)
    {
        float sizeSqft = random.Next(600, 4000);
        float bedrooms = random.Next(1, 6);
        float noise = (float)(random.NextDouble() * 20_000 - 10_000);

        // "True" underlying relationship the model has to discover
        float price = 50_000 + (sizeSqft * 120) + (bedrooms * 15_000) + noise;

        data.Add(new HouseData { SizeSqft = sizeSqft, Bedrooms = bedrooms, Price = price });
    }

    return data;
}

public class HouseData
{
    public float SizeSqft { get; set; }
    public float Bedrooms { get; set; }
    public float Price { get; set; }
}

public class HousePricePrediction
{
    [ColumnName("Score")]
    public float PredictedPrice { get; set; }
}
