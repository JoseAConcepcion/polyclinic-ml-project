from autots import AutoTS

model = AutoTS(
    forecast_length=33,
    frequency='infer',
    ensemble='simple',
    max_generations=5,
    num_validations=2,
    validation_method="backwards",
)
model = model.fit(loaded_data[0], date_col='date', value_col='target',result_file='results.csv')

# Print the description of the best model
print(model)


predictions=model.predict()
print(predictions.forecast)  # Main forecast
print(predictions.lower_forecast)  # Lower confidence bound
print(predictions.upper_forecast)  # Upper confidence bound


import matplotlib.pyplot as plt

forecast=predictions.forecast
lower_bound=predictions.lower_forecast
upper_bound=predictions.upper_forecast

predictions.plot(model.df_wide_numeric,
                series=model.df_wide_numeric.columns[0])