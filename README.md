I understand — copying piece by piece is inconvenient. The easiest way is to download the entire README as a single file. Here’s the full `.md` content in one code block so you can copy it all at once and save it as `README.md` on your computer:

```markdown
# MLFoundations

Two classic machine learning techniques — **linear regression** and **logistic regression** — implemented and compared, structured to show not just "I did both" but *why* each one exists and what it can do that the other can't.

## Structure

```
MLFoundations.sln
└── src/
    ├── MLFoundations.LinearRegression/    # TensorFlow.NET/Keras - predicting a continuous value
    └── MLFoundations.LogisticRegression/  # TensorFlow.NET/Keras - binary classification
```

> **Note:** There is no longer a `tests` project or a `NeuralNetwork` project.  
> All examples run directly from their respective console apps.

## The arc across the two projects

- **Linear regression** (`MLFoundations.LinearRegression`)  
  Predicts house price from size and bedroom count, using **TensorFlow.NET/Keras**.  
  - **Architecture:** 2 inputs → 1 neuron, **no activation function**.  
  - **Loss:** Mean Squared Error (MSE) — standard for regression, penalizes large errors.  
  - **Optimizer:** Adam — chosen for fast, adaptive training.  
  - **Output:** raw numeric prediction (continuous value).

- **Logistic regression** (`MLFoundations.LogisticRegression`)  
  Predicts pass/fail from study hours and prior test score, using **TensorFlow.NET/Keras**.  
  - **Architecture:** 2 inputs → 1 neuron, **sigmoid activation**.  
  - **Activation:** Sigmoid — squashes output into (0,1), interpretable as probability.  
  - **Loss:** Binary Crossentropy — standard for binary classification, penalizes wrong confident predictions.  
  - **Optimizer:** Adam — chosen for efficiency and adaptive learning rate.  
  - **Output:** probability of passing (thresholded at 0.5).

## Why TensorFlow.NET/Keras

TensorFlow.NET/Keras provides a clear, modern way to build regression and
classification models in C#. It makes the math and training loop easy to
express while still showing the architecture choices (activation, loss,
optimizer).  

This setup highlights how **linear regression and logistic regression are part of the same family**:
- Linear regression → continuous outputs.  
- Logistic regression → binary outputs (via sigmoid).  

## Prerequisites
- .NET 10 SDK
- TensorFlow.NET and Keras.NET packages

## Running each project
```bash
dotnet run --project src/MLFoundations.LinearRegression
dotnet run --project src/MLFoundations.LogisticRegression
```
Each prints its training progress, evaluation metrics, and a few sample
predictions to the console.

## Running with Docker
Each project has its own Dockerfile (multi-stage: SDK to build, runtime-only
to run — keeps the final image smaller). Build context is the solution root
for both, since each Dockerfile copies the `.csproj` files before restoring,
so NuGet's dependency graph resolves correctly.

Run both with one command:
```bash
docker compose up --build
```
Each container runs its console app once and exits — output appears in the
terminal per project, prefixed with the service name.

Or build/run a single one directly:
```bash
docker build -f src/MLFoundations.LinearRegression/Dockerfile -t mlfoundations-lr .
docker run --rm mlfoundations-lr
```
(Swap the Dockerfile path and image tag for `LogisticRegression` to run that instead.)

**Note:** these are one-shot console apps, not web services — there's no
port to expose or browse to. Docker support here is mainly to demonstrate
containerization itself (multi-stage builds, per-project images from a
shared solution).

## A note on the synthetic data
All datasets here are synthetically generated from a known formula (see the
`Program.cs` in each project), not pulled from a public dataset. This keeps
examples self-contained and lets the README/console output make direct
claims about what the "true" relationship is and how close the model got to
learning it.  

A natural next step would be swapping in a public dataset (e.g., Kaggle’s
housing or student-performance datasets) to see how these same techniques
handle real-world messiness.
```