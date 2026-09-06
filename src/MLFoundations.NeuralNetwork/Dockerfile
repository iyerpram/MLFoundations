# Build stage
FROM mcr.microsoft.com/dotnet/sdk:10.0 AS build
WORKDIR /src

COPY MLFoundations.sln .
COPY src/MLFoundations.LinearRegression/MLFoundations.LinearRegression.csproj src/MLFoundations.LinearRegression/
COPY src/MLFoundations.LogisticRegression/MLFoundations.LogisticRegression.csproj src/MLFoundations.LogisticRegression/
COPY src/MLFoundations.NeuralNetwork/MLFoundations.NeuralNetwork.csproj src/MLFoundations.NeuralNetwork/
COPY tests/MLFoundations.Tests/MLFoundations.Tests.csproj tests/MLFoundations.Tests/

RUN dotnet restore src/MLFoundations.NeuralNetwork/MLFoundations.NeuralNetwork.csproj

COPY src/MLFoundations.NeuralNetwork/ src/MLFoundations.NeuralNetwork/
RUN dotnet publish src/MLFoundations.NeuralNetwork/MLFoundations.NeuralNetwork.csproj -c Release -o /app --no-restore

# Runtime stage - no ML.NET here, so this image ends up noticeably smaller
# than the other two, which is itself a fair thing to point out in an
# interview (fewer dependencies, smaller attack surface, faster cold start).
FROM mcr.microsoft.com/dotnet/runtime:10.0 AS runtime
WORKDIR /app
COPY --from=build /app .

ENTRYPOINT ["dotnet", "MLFoundations.NeuralNetwork.dll"]
