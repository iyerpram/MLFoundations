# Build stage
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

COPY MLFoundations.sln .
COPY src/MLFoundations.LinearRegression/MLFoundations.LinearRegression.csproj src/MLFoundations.LinearRegression/
COPY src/MLFoundations.LogisticRegression/MLFoundations.LogisticRegression.csproj src/MLFoundations.LogisticRegression/
COPY src/MLFoundations.NeuralNetwork/MLFoundations.NeuralNetwork.csproj src/MLFoundations.NeuralNetwork/
COPY tests/MLFoundations.Tests/MLFoundations.Tests.csproj tests/MLFoundations.Tests/

RUN dotnet restore src/MLFoundations.LogisticRegression/MLFoundations.LogisticRegression.csproj

COPY src/MLFoundations.LogisticRegression/ src/MLFoundations.LogisticRegression/
RUN dotnet publish src/MLFoundations.LogisticRegression/MLFoundations.LogisticRegression.csproj -c Release -o /app --no-restore

# Runtime stage
FROM mcr.microsoft.com/dotnet/runtime:10.0 AS runtime
WORKDIR /app
COPY --from=build /app .

ENTRYPOINT ["dotnet", "MLFoundations.LogisticRegression.dll"]
