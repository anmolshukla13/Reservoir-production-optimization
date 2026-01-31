# Reservoir Production Optimization - User Guide & API Documentation

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [API Documentation](#api-documentation)
3. [Dashboard Usage](#dashboard-usage)
4. [Python SDK](#python-sdk)
5. [Advanced Features](#advanced-features)
6. [Best Practices](#best-practices)
7. [Examples](#examples)

---

## 🚀 Quick Start

### 5-Minute Setup

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/reservoir-production-optimization.git
cd reservoir-production-optimization
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Generate data and train model
python data_generator.py
python model_training.py

# 3. Start services
uvicorn api_main:app --reload &
streamlit run dashboard.py

# Done! Visit:
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

---

## 📡 API Documentation

### Base URL
```
Production: https://api.reservoir-optimizer.com
Staging: https://staging-api.reservoir-optimizer.com
Local: http://localhost:8000
```

### Authentication (Optional)

```bash
# If authentication is enabled
curl -X POST https://api.reservoir-optimizer.com/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=your_username&password=your_password"
```

### Endpoints

#### 1. Health Check

**GET** `/health`

Check API status and model availability.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-30T12:00:00Z",
  "version": "1.0.0"
}
```

---

#### 2. Production Prediction

**POST** `/predict`

Predict oil, gas, and water production rates.

**Request Body:**
```json
{
  "well_properties": {
    "porosity": 0.22,
    "permeability": 150.0,
    "net_pay": 25.0,
    "initial_pressure": 3500.0,
    "reservoir_temperature": 180.0,
    "measured_depth": 8500.0,
    "true_vertical_depth": 7800.0,
    "skin_factor": 2.0,
    "tubing_diameter": 3.5,
    "choke_size": 32,
    "oil_api": 35.0,
    "gas_gravity": 0.65
  },
  "production_data": {
    "days_on_production": 365,
    "oil_rate": 500.0,
    "gas_rate": 5000.0,
    "water_rate": 200.0,
    "reservoir_pressure": 3200.0,
    "wellhead_pressure": 800.0,
    "water_cut": 30.0,
    "gor": 1000.0
  }
}
```

**Response:**
```json
{
  "predicted_oil_rate": 525.5,
  "predicted_gas_rate": 5255.0,
  "predicted_water_rate": 195.3,
  "confidence_interval_95": {
    "oil_rate": [446.7, 604.3],
    "gas_rate": [4466.8, 6043.3],
    "water_rate": [166.0, 224.6]
  },
  "prediction_timestamp": "2024-01-30T12:00:00Z",
  "model_version": "1.0.0"
}
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "well_properties": {
        "porosity": 0.22,
        "permeability": 150.0,
        # ... other properties
    },
    "production_data": {
        "days_on_production": 365,
        # ... other data
    }
}

response = requests.post(url, json=data)
result = response.json()
print(f"Predicted oil rate: {result['predicted_oil_rate']} bbl/day")
```

---

#### 3. Production Optimization

**POST** `/optimize`

Get optimization recommendations for well production.

**Request Body:**
```json
{
  "well_properties": { ... },
  "production_data": { ... },
  "optimization_target": "oil_rate",
  "constraints": {
    "max_water_cut": 80,
    "min_reservoir_pressure": 1000
  }
}
```

**Response:**
```json
{
  "current_production": {
    "oil_rate": 500.0,
    "gas_rate": 5000.0,
    "water_rate": 200.0
  },
  "optimized_production": {
    "oil_rate": 575.0,
    "gas_rate": 5250.0,
    "water_rate": 190.0
  },
  "recommendations": [
    "Adjust choke size from 32/64\" to 40/64\"",
    "Consider acid stimulation to reduce skin factor",
    "Monitor reservoir pressure - implement pressure maintenance"
  ],
  "potential_improvement": {
    "oil_rate_percent": 15.0,
    "annual_revenue_usd": 1920000
  },
  "optimal_choke_size": 40
}
```

---

#### 4. Batch Prediction

**POST** `/batch-predict`

Process multiple predictions in one request.

**Request Body:**
```json
[
  {
    "well_properties": { ... },
    "production_data": { ... }
  },
  {
    "well_properties": { ... },
    "production_data": { ... }
  }
]
```

**Python Example:**
```python
wells_data = [
    {"well_properties": {...}, "production_data": {...}},
    {"well_properties": {...}, "production_data": {...}}
]

response = requests.post(
    "http://localhost:8000/batch-predict",
    json=wells_data
)
results = response.json()
```

---

## 🖥️ Dashboard Usage

### Accessing the Dashboard

```bash
streamlit run dashboard.py
# Visit: http://localhost:8501
```

### Dashboard Features

#### 1. Production Dashboard
- **Real-time Metrics**: View current oil, gas, water rates
- **Trend Analysis**: Visualize production trends over time
- **Water Cut Monitoring**: Track water breakthrough
- **Pressure Decline**: Monitor reservoir pressure depletion

#### 2. Production Prediction
- **Interactive Inputs**: Adjust well and production parameters
- **Instant Predictions**: Get ML predictions in real-time
- **Confidence Intervals**: View uncertainty ranges
- **Scenario Analysis**: Compare multiple what-if scenarios

#### 3. Optimization
- **AI Recommendations**: Get actionable optimization advice
- **Revenue Impact**: See potential financial improvements
- **Parameter Tuning**: Optimize choke size, injection rates
- **Constraint Handling**: Set operational constraints

#### 4. Data Explorer
- **Raw Data View**: Browse production history
- **Statistical Analysis**: View summary statistics
- **Data Export**: Download data as CSV
- **Filtering**: Filter by date range, well, zone

---

## 🐍 Python SDK

### Installation

```bash
pip install reservoir-optimizer-sdk
```

### Basic Usage

```python
from reservoir_optimizer import ReservoirClient

# Initialize client
client = ReservoirClient(api_url="http://localhost:8000")

# Make prediction
prediction = client.predict(
    porosity=0.22,
    permeability=150.0,
    oil_rate=500.0,
    days_on_production=365
)

print(f"Predicted rate: {prediction.oil_rate} bbl/day")
```

### Advanced Usage

```python
from reservoir_optimizer import ReservoirClient, Well, ProductionData

# Create well object
well = Well(
    porosity=0.22,
    permeability=150.0,
    net_pay=25.0,
    initial_pressure=3500.0
)

# Create production data
production = ProductionData(
    days_on_production=365,
    oil_rate=500.0,
    gas_rate=5000.0,
    water_rate=200.0
)

# Get predictions
client = ReservoirClient()
prediction = client.predict(well, production)

# Get optimization
optimization = client.optimize(well, production)
print(optimization.recommendations)

# Batch processing
wells = [well1, well2, well3]
productions = [prod1, prod2, prod3]
results = client.batch_predict(wells, productions)
```

---

## 🎯 Advanced Features

### Feature Engineering

```python
from preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor(scaling_method='robust')

# Load and process data
df = preprocessor.load_data('data/synthetic/production_data.csv')
df = preprocessor.create_time_features(df)
df = preprocessor.create_lag_features(df, ['oil_rate', 'gas_rate'])
df = preprocessor.create_rolling_features(df, ['oil_rate'], windows=[7, 30])
```

### Custom Model Training

```python
from model_training import ReservoirMLModels

# Initialize trainer
trainer = ReservoirMLModels(random_state=42)

# Add custom model
from sklearn.ensemble import GradientBoostingRegressor
custom_model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=10
)
trainer.models['Custom GBM'] = custom_model

# Train all models
results = trainer.train_all_models(X_train, y_train, X_test, y_test)

# Save best model
trainer.save_model()
```

### Production Optimization

```python
# Define optimization function
def optimize_well_production(well_params, constraints):
    """
    Optimize production parameters
    """
    from scipy.optimize import minimize
    
    def objective(x):
        # x = [choke_size, drawdown]
        predicted_rate = model.predict([[x[0], x[1]]])
        return -predicted_rate  # Negative because we minimize
    
    # Constraints
    cons = [
        {'type': 'ineq', 'fun': lambda x: x[0] - 16},  # Min choke
        {'type': 'ineq', 'fun': lambda x: 64 - x[0]},  # Max choke
        {'type': 'ineq', 'fun': lambda x: constraints['max_drawdown'] - x[1]}
    ]
    
    # Optimize
    result = minimize(
        objective,
        x0=[32, 1500],  # Initial guess
        method='SLSQP',
        constraints=cons
    )
    
    return result
```

---

## ✅ Best Practices

### Data Quality

1. **Clean Your Data**
```python
# Remove outliers
df = preprocessor.remove_outliers(df, method='iqr', threshold=3)

# Handle missing values
df = preprocessor.handle_missing_values(df, strategy='knn')

# Validate ranges
assert df['porosity'].between(0, 1).all()
assert df['oil_rate'].ge(0).all()
```

2. **Feature Engineering**
```python
# Create domain-specific features
df['productivity_index'] = df['oil_rate'] / df['drawdown']
df['gor'] = df['gas_rate'] / df['oil_rate']
df['water_cut'] = df['water_rate'] / (df['oil_rate'] + df['water_rate'])
```

### Model Selection

1. **Start Simple**: Begin with Linear Regression, then try tree-based models
2. **Cross-Validate**: Always use cross-validation (k=5 or k=10)
3. **Ensemble**: Combine multiple models for better predictions
4. **Monitor**: Track model performance over time

### API Usage

1. **Batch Requests**: Use batch endpoints for multiple predictions
2. **Caching**: Cache frequent predictions to reduce latency
3. **Error Handling**: Always handle API errors gracefully

```python
import requests
from requests.exceptions import RequestException

try:
    response = requests.post(url, json=data, timeout=10)
    response.raise_for_status()
    result = response.json()
except RequestException as e:
    print(f"API Error: {e}")
    # Use fallback prediction
```

---

## 💡 Examples

### Example 1: Single Well Prediction

```python
import requests

# Well data
well = {
    "porosity": 0.22,
    "permeability": 150.0,
    "net_pay": 25.0,
    "initial_pressure": 3500.0,
    "reservoir_temperature": 180.0,
    "measured_depth": 8500.0,
    "true_vertical_depth": 7800.0,
    "skin_factor": 2.0,
    "tubing_diameter": 3.5,
    "choke_size": 32,
    "oil_api": 35.0,
    "gas_gravity": 0.65
}

production = {
    "days_on_production": 365,
    "oil_rate": 500.0,
    "gas_rate": 5000.0,
    "water_rate": 200.0,
    "reservoir_pressure": 3200.0,
    "wellhead_pressure": 800.0,
    "water_cut": 30.0,
    "gor": 1000.0
}

# Predict
response = requests.post(
    "http://localhost:8000/predict",
    json={"well_properties": well, "production_data": production}
)

result = response.json()
print(f"Predicted oil rate: {result['predicted_oil_rate']:.1f} bbl/day")
```

### Example 2: Field-Wide Optimization

```python
import pandas as pd
import requests

# Load field data
field_data = pd.read_csv('field_wells.csv')

results = []
for _, well in field_data.iterrows():
    # Prepare data
    well_props = well[['porosity', 'permeability', ...]].to_dict()
    prod_data = well[['oil_rate', 'gas_rate', ...]].to_dict()
    
    # Get optimization
    response = requests.post(
        "http://localhost:8000/optimize",
        json={
            "well_properties": well_props,
            "production_data": prod_data
        }
    )
    
    results.append(response.json())

# Analyze results
total_improvement = sum(r['potential_improvement']['oil_rate_percent'] for r in results)
print(f"Field-wide potential improvement: {total_improvement:.1f}%")
```

### Example 3: Automated Monitoring

```python
import schedule
import time

def monitor_well(well_id):
    """Monitor well production and send alerts"""
    
    # Fetch current data
    current_data = fetch_well_data(well_id)
    
    # Get prediction
    prediction = predict_production(current_data)
    
    # Check for anomalies
    if abs(current_data['oil_rate'] - prediction['oil_rate']) > 100:
        send_alert(f"Well {well_id}: Production deviation detected!")
    
    # Check water cut
    if current_data['water_cut'] > 70:
        send_alert(f"Well {well_id}: High water cut - {current_data['water_cut']:.1f}%")

# Schedule monitoring every hour
schedule.every(1).hours.do(monitor_well, well_id='WELL_001')

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 📞 Support & Resources

- **API Documentation**: http://localhost:8000/docs
- **GitHub**: https://github.com/yourusername/reservoir-production-optimization
- **Issues**: https://github.com/yourusername/reservoir-production-optimization/issues
- **Email**: support@reservoir-optimizer.com

---

**Happy Optimizing! 🚀**
