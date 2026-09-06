using MLFoundations.NeuralNetwork;

// --- Neural Network: a realistic example (loan default risk) ---
//
// Task: predict whether a loan applicant will default, from 4 features:
// annual income, credit score, debt-to-income ratio, and years employed.
//
// Why this needs a neural network and not just logistic regression:
// the synthetic "true" default rule below is deliberately an OR-of-AND
// combination of conditions:
//
//   defaults if:  (high debt-to-income AND poor credit score)
//              OR (very new to job AND low income)
//
// This is structurally the same kind of problem as XOR - no single straight
// decision boundary separates defaulters from non-defaulters, because
// "risky" applicants come from two different corners of the feature space
// for two different reasons. A logistic regression model, which can only
// combine features linearly, would need someone to manually engineer
// interaction terms (e.g., DebtToIncome * (CreditScore < 600)) to capture
// this. A neural network's hidden layer learns this kind of interaction
// automatically, directly from the raw features - that's the concrete,
// realistic reason to reach for one here instead of the simpler model.

var random = new Random(42);
const int sampleCount = 2000;

var applicants = new List<LoanApplicant>(sampleCount);
for (int i = 0; i < sampleCount; i++)
{
    double income = 20_000 + random.NextDouble() * 130_000;      // $20k - $150k
    double creditScore = 300 + random.NextDouble() * 550;         // 300 - 850
    double debtToIncome = random.NextDouble() * 0.6;              // 0 - 60%
    double employmentYears = random.NextDouble() * 30;            // 0 - 30 years

    bool highDebtPoorCredit = debtToIncome > 0.4 && creditScore < 600;
    bool newJobLowIncome = employmentYears < 1 && income < 30_000;

    // A little label noise, since real-world outcomes are never a perfectly
    // clean function of the inputs - this also keeps the task from being
    // trivially easy.
    bool noisyFlip = random.NextDouble() < 0.05;
    bool defaulted = (highDebtPoorCredit || newJobLowIncome) ^ noisyFlip;

    applicants.Add(new LoanApplicant(income, creditScore, debtToIncome, employmentYears, defaulted));
}

// Neural nets trained with sigmoid activations converge far better when
// inputs are on a similar scale - raw income (tens of thousands) would
// otherwise dominate credit score or debt ratio just by magnitude, not
// because it's actually more predictive. Min-max normalize each feature to [0, 1].
var (min, max) = ComputeFeatureRanges(applicants);
var normalized = applicants.Select(a => Normalize(a, min, max)).ToList();

int splitIndex = (int)(sampleCount * 0.8);
var trainSet = normalized.Take(splitIndex).ToList();
var testSet = normalized.Skip(splitIndex).ToList();

var network = new SimpleNeuralNetwork(inputSize: 4, hiddenSize: 6, learningRate: 0.3, seed: 42);

const int epochs = 3000;
Console.WriteLine($"Training on {trainSet.Count} applicants for {epochs} epochs...");
for (int epoch = 0; epoch < epochs; epoch++)
{
    double totalError = 0;
    foreach (var (features, label) in trainSet)
        totalError += network.TrainOne(features, label ? 1 : 0);

    if (epoch % 500 == 0)
        Console.WriteLine($"  Epoch {epoch,5}: total squared error = {totalError:F2}");
}

// Evaluate on the held-out test set
int truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;
foreach (var (features, label) in testSet)
{
    bool predicted = network.Predict(features) > 0.5;
    if (predicted && label) truePositive++;
    else if (!predicted && !label) trueNegative++;
    else if (predicted && !label) falsePositive++;
    else falseNegative++;
}

double accuracy = (double)(truePositive + trueNegative) / testSet.Count;
double precision = truePositive + falsePositive == 0 ? 0 : (double)truePositive / (truePositive + falsePositive);
double recall = truePositive + falseNegative == 0 ? 0 : (double)truePositive / (truePositive + falseNegative);

Console.WriteLine();
Console.WriteLine("=== Evaluation on held-out test set ===");
Console.WriteLine($"Accuracy:  {accuracy:P1}");
Console.WriteLine($"Precision: {precision:P1}  (of predicted defaults, how many actually defaulted)");
Console.WriteLine($"Recall:    {recall:P1}  (of actual defaults, how many were caught)");
Console.WriteLine($"Confusion: TP={truePositive} TN={trueNegative} FP={falsePositive} FN={falseNegative}");

Console.WriteLine();
Console.WriteLine("=== Sample predictions ===");
foreach (var applicant in new[]
{
    new LoanApplicant(Income: 25_000, CreditScore: 580, DebtToIncome: 0.5, EmploymentYears: 5, Defaulted: false), // high debt + poor credit -> expect risky
    new LoanApplicant(Income: 22_000, CreditScore: 720, DebtToIncome: 0.2, EmploymentYears: 0.3, Defaulted: false), // new job + low income -> expect risky
    new LoanApplicant(Income: 90_000, CreditScore: 750, DebtToIncome: 0.15, EmploymentYears: 10, Defaulted: false), // safe on all fronts
})
{
    var (features, _) = Normalize(applicant, min, max);
    var risk = network.Predict(features);
    Console.WriteLine($"  Income=${applicant.Income:N0}, Credit={applicant.CreditScore:F0}, " +
        $"DTI={applicant.DebtToIncome:P0}, Years={applicant.EmploymentYears:F1} " +
        $"-> predicted default risk: {risk:P1}");
}

static (LoanApplicant min, LoanApplicant max) ComputeFeatureRanges(List<LoanApplicant> data) => (
    new LoanApplicant(data.Min(a => a.Income), data.Min(a => a.CreditScore), data.Min(a => a.DebtToIncome), data.Min(a => a.EmploymentYears), false),
    new LoanApplicant(data.Max(a => a.Income), data.Max(a => a.CreditScore), data.Max(a => a.DebtToIncome), data.Max(a => a.EmploymentYears), false)
);

static (double[] features, bool label) Normalize(LoanApplicant a, LoanApplicant min, LoanApplicant max) => (
    [
        Scale(a.Income, min.Income, max.Income),
        Scale(a.CreditScore, min.CreditScore, max.CreditScore),
        Scale(a.DebtToIncome, min.DebtToIncome, max.DebtToIncome),
        Scale(a.EmploymentYears, min.EmploymentYears, max.EmploymentYears)
    ],
    a.Defaulted
);

static double Scale(double value, double min, double max) => max > min ? (value - min) / (max - min) : 0;

public record LoanApplicant(double Income, double CreditScore, double DebtToIncome, double EmploymentYears, bool Defaulted);
