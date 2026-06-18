"""
forecast/ — Market Forecast package.

Point-forecasts the S&P 500 total price return over the next 1, 3, 6, and 12
months from an equal-weight ENSEMBLE of model-family members (regularized
linear, k-NN, random forest, neural net) plus a prevailing-mean benchmark.

Design mirrors the bear/ package: all ML is done OFFLINE (scikit-learn) and the
results are written to data/*.csv + data/forecast_params.json so the Streamlit
app can read them with numpy/pandas only (no ML import on the app path).

Pipeline (run in order):
    python -m forecast.targets       # data/forecast_targets.csv
    python -m forecast.features      # data/forecast_features.csv
    python -m forecast.models        # data/forecast_members_{1,3,6,12}m.csv
    python -m forecast.ensemble      # forecast_ensemble_oos.csv + forecast_params.json
    python -m forecast.univariate    # data/univariate_forecast.csv
"""
