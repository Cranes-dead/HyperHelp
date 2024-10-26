# %%
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report
import numpy as np

# %%
data = pd.read_excel('Inventory_Management_Data.xlsx')
categorical_features = ['product', 'category', 'sub_category', 'brand', 'type', 'Supplier Info']
features = ['sale_price', 'market_price', 'rating', 'Stock Quantity', 'Lead Time (Days)', 'Units Sold']
target = 'Units Sold' 
# Encode categorical variables
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Add encoded categorical columns to features
features += categorical_features

# Split the data into features (X) and target (y)
X = data[features]
y = data[target]

# Scale the features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest model on all data
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_scaled, y)

# Make predictions for all data
all_predictions = model.predict(X_scaled)

# Add predictions to original dataframe
data['Predicted_Sales'] = all_predictions

# Add demand categories
mean_sales = data[target].mean()
def categorize_demand(pred):
    if pred > mean_sales * 1.2:  # 20% above mean
        return 'High'
    elif pred < mean_sales * 0.8:  # 20% below mean
        return 'Low'
    else:
        return 'Medium'

data['Demand_Category'] = data['Predicted_Sales'].apply(categorize_demand)

# Save to CSV
data.to_csv('data_with_predictions.csv', index=False)

print("Added columns:")
print("\nFirst few rows of predictions:")
print(data[['Predicted_Sales', 'Demand_Category']].head())

# Print basic accuracy metrics
accuracy = (data['Predicted_Sales'] - data[target]).abs().mean()
print(f"\nAverage prediction error: {accuracy:.2f}")

# %%
import pandas as pd

# Load the updated data
data = pd.read_excel('Updated_Inventory_Management_Data.xlsx')

# Set 'date' as the index and sort by date
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Verify data structure
print(data.head())

# %%
from statsmodels.tsa.stattools import adfuller

# Apply Dickey-Fuller test
result = adfuller(data['Units Sold'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])

if result[1] > 0.05:
    print("Data is non-stationary. Differencing is required.")
    # Difference the data
    data['Units_Sold_Diff'] = data['Units Sold'].diff().dropna()
else:
    print("Data is stationary. No differencing required.")

# %%
import pmdarima as pm

# Use auto_arima to determine the best SARIMA model
auto_model = pm.auto_arima(data['Units Sold'], seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
print(auto_model.summary())


# %%
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set SARIMA parameters (example parameters here; replace with those from auto_arima)
p, d, q = 1, 1, 1  # ARIMA component
P, D, Q, s = 1, 1, 1, 12  # Seasonal component with s = 12 for monthly seasonality

# Fit the SARIMA model
sarima_model = SARIMAX(data['Units Sold'], order=(p, d, q), seasonal_order=(P, D, Q, s), enforce_stationarity=False, enforce_invertibility=False)
sarima_results = sarima_model.fit(disp=False)
print(sarima_results.summary())


# %%
# Forecast for the next 12 months (or desired period)
forecast_steps = 12
forecast = sarima_results.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

# Append forecasted values to the original data for visual comparison
forecast_data = forecast.predicted_mean
print("Forecasted reorder levels:\n", forecast_data)

# Optional: Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(data['Units Sold'], label='Historical Units Sold')
plt.plot(forecast_data, label='Forecasted Units Sold', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.title("SARIMA Forecast for Inventory Reorder Levels")
plt.show()


# %%
# Assuming SARIMA model has already been fitted as 'sarima_results'
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd

# Load the data
data = pd.read_excel('Updated_Inventory_Management_Data.xlsx', index_col='date', parse_dates=True)

# Forecast using SARIMA for the date range in data
# Adjust the index for SARIMA prediction if necessary
data['Forecasted_Reorder'] = sarima_results.predict(start=data.index[0], end=data.index[-1])

# Save to Excel file
data.to_csv('Inventory_with_SARIMA_Predictions.csv')
print("Updated inventory data with SARIMA forecasts saved as 'Inventory_with_SARIMA_Predictions.xlsx'")

# Display the head of the forecasted data for verification
print(data[['Forecasted_Reorder']].head())



