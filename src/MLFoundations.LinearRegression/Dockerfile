# Build stage
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

COPY MLFoundations.sln .
COPY src/MLFoundations.LinearRegression/MLFoundations.LinearRegression.csproj src/MLFoundations.LinearRegression/
COPY src/MLFoundations.LogisticRegression/MLFoundations.LogisticRegression.csproj src/MLFoundations.LogisticRegression/
COPY src/MLFoundations.NeuralNetwork/MLFoundations.NeuralNetwork.csproj src/MLFoundations.NeuralNetwork/
COPY tests/MLFoundations.Tests/MLFoundations.Tests.csproj tests/MLFoundations.Tests/

RUN dotnet restore src/MLFoundations.LinearRegression/MLFoundations.LinearRegression.csproj

COPY src/MLFoundations.LinearRegression/ src/MLFoundations.LinearRegression/
RUN dotnet publish src/MLFoundations.LinearRegression/MLFoundations.LinearRegression.csproj -c Release -o /app --no-restore

# Runtime stage - smaller image, no SDK
FROM mcr.microsoft.com/dotnet/runtime:10.0 AS runtime
WORKDIR /app
COPY --from=build /app .

ENTRYPOINT ["dotnet", "MLFoundations.LinearRegression.dll"]
