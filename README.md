# MLFoundations

Three classic machine learning techniques — linear regression, logistic
regression, and a neural network — implemented and compared, structured to
show not just "I did all three" but *why* each one exists and what it can
do that the others can't.

## Structure

```
MLFoundations.sln
├── src/
│   ├── MLFoundations.LinearRegression/    # ML.NET - predicting a continuous value
│   ├── MLFoundations.LogisticRegression/  # ML.NET - binary classification
│   └── MLFoundations.NeuralNetwork/       # From scratch, no ML library -
│                                           # forward pass + backprop + gradient
│                                           # descent, solving XOR
└── tests/
    └── MLFoundations.Tests/               # xUnit tests for the from-scratch network
```

## The arc across the three projects

- **Linear regression** (`MLFoundations.LinearRegression`) — predicts house
  price from size and bedroom count, using ML.NET's SDCA trainer on a
  synthetic dataset generated from a known formula (so the model's learned
  behavior can be sanity-checked against ground truth).
- **Logistic regression** (`MLFoundations.LogisticRegression`) — predicts
  pass/fail from study hours and prior test score, using ML.NET's
  `SdcaLogisticRegression` trainer. Same idea as linear regression, but for
  a binary outcome instead of a continuous one.
- **Neural network** (`MLFoundations.NeuralNetwork`) — deliberately
  **not** built with ML.NET. It's a from-scratch feedforward network (one
  hidden layer, sigmoid activation, manual backpropagation) trained on a
  **loan default risk** task: predicting whether an applicant defaults from
  income, credit score, debt-to-income ratio, and years employed. The
  synthetic "true" rule is deliberately an **OR-of-AND** combination —
  "risky if (high debt-to-income AND poor credit) OR (new to job AND low
  income)" — which is structurally the same kind of problem as the classic
  XOR example: no single straight decision boundary separates the risky
  applicants from the safe ones, because they end up risky for two
  different, unrelated reasons. A logistic regression model would need
  someone to manually engineer an interaction term to capture this; the
  neural network's hidden layer learns it automatically from the raw
  features. Running this after the logistic regression project makes that
  contrast concrete rather than abstract.

## Why ML.NET for the first two, and hand-rolled for the third

ML.NET is a mature, stable part of the .NET ecosystem — reaching for it for
standard regression/classification tasks is what you'd actually do in
production. But if all three projects used ML.NET, the neural network would
just be "call a different trainer" — it wouldn't demonstrate any deeper
understanding. Implementing the network from scratch shows the mechanics
(forward pass, error, backpropagation, weight updates) explicitly, which is
the more interesting signal for a project like this.

## Prerequisites
- .NET 10 SDK

## Running each project
```bash
dotnet run --project src/MLFoundations.LinearRegression
dotnet run --project src/MLFoundations.LogisticRegression
dotnet run --project src/MLFoundations.NeuralNetwork
```
Each prints its training progress, evaluation metrics, and a few sample
predictions to the console.

## Running the tests
```bash
dotnet test
```
The existing tests exercise the `SimpleNeuralNetwork` class directly on the
XOR truth table (kept in the tests project even though `Program.cs` now
runs the loan default example) — XOR remains a fast, deterministic way to
verify the network can learn a non-linearly-separable function at all,
without needing the larger synthetic loan dataset in a unit test. The console
output when you run the project itself is where the realistic example lives.

## A note on the synthetic data
All datasets here are synthetically generated from a known formula (see the
`GenerateSynthetic...` functions in each project's `Program.cs`), not
pulled from a public dataset. This is intentional for a foundations-focused
project — it keeps the examples self-contained and dependency-free, and
lets the README/console output make direct claims about what the "true"
relationship is and how close the model got to learning it. A natural next
step, if you want to extend this into something closer to real-world data
work, would be swapping in a public dataset (e.g., Kaggle's housing or
student-performance datasets) for the first two projects and comparing how
much messier real data makes the same techniques.

## A note on build stability
Unlike the newer .NET AI stack used in the IncidentsAi project (Agent
Framework, Microsoft.Extensions.VectorData, Qdrant connectors — all still
pre-GA and prone to version-mismatch issues), everything in this repo uses
long-stable packages (`Microsoft.ML`) or no external package at all (the
neural network). This should build cleanly without the kind of dependency
troubleshooting IncidentsAi required — though as always, I wasn't able to
compile it myself before handing it over, so if something doesn't build,
share the exact error.
