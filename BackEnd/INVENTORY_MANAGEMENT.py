# %%
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

# Set basic plot settings without using style files
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
sns.set_theme(style="whitegrid")  # Using seaborn's built-in theme

def load_and_preprocess_data(file_path):
    """Load and preprocess the inventory data"""
    
    # Load data
    data = pd.read_excel(file_path)
    
    # Define feature categories
    categorical_features = ['product', 'category', 'sub_category', 'brand', 'type', 'Supplier Info']
    numerical_features = ['sale_price', 'market_price', 'rating', 'Stock Quantity', 'Lead Time (Days)']
    target = 'Units Sold'
    
    # Store original data
    original_data = data.copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    # Combine features
    features = numerical_features + categorical_features
    
    # Split features and target
    X = data[features]
    y = data[target]
    
    # Scale numerical features
    scaler = StandardScaler()
    X_numerical = X[numerical_features]
    X_numerical_scaled = scaler.fit_transform(X_numerical)
    
    # Create final scaled dataframe
    X_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_features)
    for col in categorical_features:
        X_scaled_df[col] = X[col]
    
    return X_scaled_df, y, original_data, features, label_encoders, numerical_features, categorical_features

def train_random_forest(X_scaled_df, y):
    """Train and evaluate Random Forest model"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    return model, metrics

def generate_model_insights(model, X_scaled_df, features):
    """Generate comprehensive model insights"""
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create feature importance plot
    plt.figure()
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance in Inventory Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Generate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled_df)
    
    # Create SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_scaled_df, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()
    
    # Create feature interactions plot
    plt.figure()
    most_important_feature = feature_importance.iloc[0]['feature']
    shap.dependence_plot(
        ind=most_important_feature,
        shap_values=shap_values,
        features=X_scaled_df,
        feature_names=features,
        show=False
    )
    plt.savefig('feature_interactions.png')
    plt.close()
    
    return feature_importance, shap_values

def generate_prediction_explanation(model, X_scaled_df, original_data, shap_values, 
                                 index, features, numerical_features):
    """Generate detailed explanation for a specific prediction"""
    
    prediction = model.predict(X_scaled_df.iloc[index].values.reshape(1, -1))[0]
    shap_values_single = shap_values[index]
    
    explanation = {
        'predicted_units': round(prediction, 2),
        'key_factors': [],
        'confidence_metrics': {
            'feature_contributions': {}
        }
    }
    
    # Analyze feature contributions
    feature_impacts = list(zip(features, shap_values_single))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for feature, impact in feature_impacts[:5]:
        original_value = original_data.iloc[index][feature]
        impact_direction = "increased" if impact > 0 else "decreased"
        
        explanation['key_factors'].append({
            'feature': feature,
            'value': original_value,
            'impact': abs(impact),
            'direction': impact_direction,
            'relative_contribution': abs(impact) / sum(abs(shap_values_single)) * 100
        })
        
        explanation['confidence_metrics']['feature_contributions'][feature] = abs(impact)
    
    return explanation

def analyze_time_series(ts_file_path):
    """Perform time series analysis and forecasting"""
    try:
        # Load time series data
        ts_data = pd.read_excel(ts_file_path)
        ts_data['date'] = pd.to_datetime(ts_data['date'])
        ts_data.set_index('date', inplace=True)
        
        # Perform seasonal decomposition
        decomposition = seasonal_decompose(ts_data['Units Sold'], period=12)
        
        # Fit SARIMA model
        auto_model = pm.auto_arima(ts_data['Units Sold'], seasonal=True, m=12, trace=True,
                                  error_action='ignore', suppress_warnings=True)
        
        sarima_model = SARIMAX(ts_data['Units Sold'],
                              order=auto_model.order,
                              seasonal_order=auto_model.seasonal_order)
        sarima_results = sarima_model.fit(disp=False)
        
        # Generate forecasts
        forecast_steps = 12
        forecast = sarima_results.get_forecast(steps=forecast_steps)
        forecast_data = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        # Create visualizations
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        
        # Plot components
        decomposition.trend.plot(ax=axes[0])
        axes[0].set_title('Trend Component')
        
        decomposition.seasonal.plot(ax=axes[1])
        axes[1].set_title('Seasonal Component')
        
        decomposition.resid.plot(ax=axes[2])
        axes[2].set_title('Residual Component')
        
        # Plot forecast
        axes[3].plot(forecast_data.index, forecast_data, label='Forecast', color='blue')
        axes[3].fill_between(
            forecast_data.index,
            forecast_ci.iloc[:, 0],
            forecast_ci.iloc[:, 1],
            color='gray',
            alpha=0.2,
            label='95% Confidence Interval'
        )
        axes[3].set_title('Forecast with Uncertainty')
        axes[3].legend()
        
        plt.tight_layout()
        plt.savefig('time_series_components.png')
        plt.close()
        
        # Create forecast summary
        forecast_summary = pd.DataFrame({
            'Forecast': forecast_data,
            'Lower_CI': forecast_ci.iloc[:, 0],
            'Upper_CI': forecast_ci.iloc[:, 1],
            'Uncertainty_Range': forecast_ci.iloc[:, 1] - forecast_ci.iloc[:, 0]
        })
        
        return forecast_summary, decomposition
    except Exception as e:
        print(f"Error in time series analysis: {str(e)}")
        return None, None

def main():
    """Main execution function"""
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        X_scaled_df, y, original_data, features, label_encoders, numerical_features, categorical_features = \
            load_and_preprocess_data('Inventory_Management_Data.xlsx')
        
        # Train model
        print("Training Random Forest model...")
        model, metrics = train_random_forest(X_scaled_df, y)
        print("\nModel Performance Metrics:")
        print(f"R² Score: {metrics['r2']:.3f}")
        print(f"Mean Absolute Error: {metrics['mae']:.3f}")
        print(f"Root Mean Squared Error: {metrics['rmse']:.3f}")
        
        # Generate model insights
        print("Generating model insights...")
        feature_importance, shap_values = generate_model_insights(model, X_scaled_df, features)
        
        # Make predictions and generate explanations
        print("Generating predictions and explanations...")
        predictions = model.predict(X_scaled_df)
        explanations = []
        
        for i in range(len(original_data)):
            explanation = generate_prediction_explanation(
                model, X_scaled_df, original_data, shap_values, i, 
                features, numerical_features
            )
            explanations.append(explanation)
        
        # Add predictions and explanations to original data
        results_df = original_data.copy()
        results_df['Predicted_Units'] = predictions
        results_df['Prediction_Explanation'] = [
            f"Predicted {exp['predicted_units']} units. Key factors:\n" +
            "\n".join([
                f"- {f['feature']}: {f['value']} {f['direction']} prediction by {f['impact']:.2f} " +
                f"({f['relative_contribution']:.1f}% contribution)"
                for f in exp['key_factors']
            ])
            for exp in explanations
        ]
        
        # Perform time series analysis
        print("Performing time series analysis...")
        forecast_summary, _ = analyze_time_series('Updated_Inventory_Management_Data.xlsx')
        
        # Save results
        print("Saving results...")
        results_df.to_csv('inventory_predictions_with_explanations.csv', index=False)
        if forecast_summary is not None:
            forecast_summary.to_csv('forecast_with_uncertainty.csv')
        
        print("\nAnalysis completed! Generated files:")
        print("1. inventory_predictions_with_explanations.csv - ML predictions with explanations")
        print("2. forecast_with_uncertainty.csv - Time series forecasts with uncertainty")
        print("3. feature_importance.png - Feature importance visualization")
        print("4. shap_summary.png - SHAP values analysis")
        print("5. feature_interactions.png - Feature interaction analysis")
        print("6. time_series_components.png - Time series decomposition and forecast")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()

# %%
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

# [Previous imports and settings remain the same...]

def calculate_reorder_levels(data, service_level=0.95):
    """
    Calculate reorder levels for each product considering:
    - Lead time
    - Average daily demand
    - Demand variability
    - Desired service level
    - Safety stock
    
    Parameters:
    data (pd.DataFrame): DataFrame containing inventory data
    service_level (float): Desired service level (default: 0.95)
    
    Returns:
    pd.DataFrame: Original data with reorder level calculations
    """
    # Calculate average daily demand
    data['avg_daily_demand'] = data['Units Sold'] / 30  # Assuming monthly data
    
    # Calculate standard deviation of daily demand
    # Group by product to get demand variation if multiple records exist
    demand_std = data.groupby('product')['avg_daily_demand'].transform('std')
    # Fill NaN values with a small percentage of mean demand
    demand_std = demand_std.fillna(data['avg_daily_demand'] * 0.1)
    
    # Safety factor (z-score) based on service level
    # Using normal distribution assumption
    z_score = np.abs(np.percentile(np.random.normal(0, 1, 10000), service_level * 100))
    
    # Calculate safety stock
    # Safety Stock = Z × σ × √(Lead Time)
    data['safety_stock'] = z_score * demand_std * np.sqrt(data['Lead Time (Days)'])
    
    # Calculate reorder level
    # Reorder Level = (Average Daily Demand × Lead Time) + Safety Stock
    data['reorder_level'] = (data['avg_daily_demand'] * data['Lead Time (Days)']) + data['safety_stock']
    
    # Calculate economic order quantity (EOQ)
    # Assuming ordering cost is $100 and holding cost is 20% of unit cost annually
    ordering_cost = 100
    holding_cost_rate = 0.20
    data['holding_cost'] = data['market_price'] * holding_cost_rate
    data['eoq'] = np.sqrt((2 * ordering_cost * data['Units Sold']) / data['holding_cost'])
    
    # Round all calculated fields to reasonable numbers
    data['safety_stock'] = data['safety_stock'].round(0)
    data['reorder_level'] = data['reorder_level'].round(0)
    data['eoq'] = data['eoq'].round(0)
    
    return data

def analyze_inventory_status(data):
    """
    Analyze current inventory status and generate recommendations
    """
    inventory_status = data.copy()
    
    # Determine inventory status
    inventory_status['status'] = 'Adequate'
    inventory_status.loc[inventory_status['Stock Quantity'] <= inventory_status['reorder_level'], 'status'] = 'Reorder'
    inventory_status.loc[inventory_status['Stock Quantity'] <= inventory_status['safety_stock'], 'status'] = 'Critical'
    
    # Generate recommendations
    def get_recommendation(row):
        if row['status'] == 'Critical':
            return f"URGENT: Place order for {row['eoq']} units immediately. Current stock ({row['Stock Quantity']}) below safety stock level ({row['safety_stock']:.0f})"
        elif row['status'] == 'Reorder':
            return f"Place order for {row['eoq']} units. Current stock ({row['Stock Quantity']}) below reorder level ({row['reorder_level']:.0f})"
        else:
            return f"Stock adequate. Review in {((row['Stock Quantity'] - row['reorder_level']) / row['avg_daily_demand']):.0f} days"
    
    inventory_status['recommendation'] = inventory_status.apply(get_recommendation, axis=1)
    
    return inventory_status

def main():
    """Main execution function"""
    try:
        # [Previous loading and preprocessing code remains the same...]
        X_scaled_df, y, original_data, features, label_encoders, numerical_features, categorical_features = \
            load_and_preprocess_data('Inventory_Management_Data.xlsx')
        
        # Train model and generate insights as before
        model, metrics = train_random_forest(X_scaled_df, y)
        feature_importance, shap_values = generate_model_insights(model, X_scaled_df, features)
        
        # Calculate reorder levels and analyze inventory status
        print("Calculating reorder levels and analyzing inventory status...")
        inventory_data = calculate_reorder_levels(original_data)
        inventory_status = analyze_inventory_status(inventory_data)
        
        # Add inventory status visualizations
        plt.figure(figsize=(12, 6))
        status_counts = inventory_status['status'].value_counts()
        sns.barplot(x=status_counts.index, y=status_counts.values)
        plt.title('Inventory Status Distribution')
        plt.tight_layout()
        plt.savefig('inventory_status.png')
        plt.close()
        
        # Save enhanced results
        inventory_status.to_csv('inventory_analysis_with_reorder_levels.csv', index=False)
        
        print("\nAnalysis completed! Additional generated files:")
        print("7. inventory_analysis_with_reorder_levels.csv - Complete inventory analysis with reorder levels")
        print("8. inventory_status.png - Inventory status distribution visualization")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()

# %%
from sklearn.metrics import accuracy_score
accuracy = accuracy_score()


