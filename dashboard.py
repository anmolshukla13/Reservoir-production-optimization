"""
Streamlit Dashboard for Reservoir Production Optimization
==========================================================
Interactive dashboard for production monitoring and optimization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Reservoir Production Optimizer",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = "http://localhost:8000"


# ============================================================================
# Helper Functions
# ============================================================================

def load_sample_data():
    """Load sample production data"""
    try:
        df = pd.read_csv('data/synthetic/production_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
        # Generate sample data if file doesn't exist
        dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='D')
        return pd.DataFrame({
            'date': dates,
            'well_id': 'WELL_001',
            'oil_rate': np.random.uniform(300, 600, len(dates)),
            'gas_rate': np.random.uniform(3000, 6000, len(dates)),
            'water_rate': np.random.uniform(100, 300, len(dates)),
            'reservoir_pressure': np.linspace(3500, 2800, len(dates)),
            'water_cut': np.linspace(20, 60, len(dates))
        })


def predict_production(well_props, prod_data):
    """Call prediction API"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "well_properties": well_props,
                "production_data": prod_data
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        # Mock response if API unavailable
        return {
            "predicted_oil_rate": prod_data.get('oil_rate', 500) * 1.05,
            "predicted_gas_rate": prod_data.get('gas_rate', 5000) * 1.05,
            "predicted_water_rate": prod_data.get('water_rate', 200) * 0.95,
            "confidence_interval_95": {
                "oil_rate": [450, 550],
                "gas_rate": [4500, 5500],
                "water_rate": [180, 220]
            }
        }


def optimize_production(well_props, prod_data):
    """Call optimization API"""
    try:
        response = requests.post(
            f"{API_URL}/optimize",
            json={
                "well_properties": well_props,
                "production_data": prod_data,
                "optimization_target": "oil_rate"
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        # Mock response
        return {
            "current_production": {
                "oil_rate": prod_data.get('oil_rate', 500),
                "gas_rate": prod_data.get('gas_rate', 5000),
                "water_rate": prod_data.get('water_rate', 200)
            },
            "optimized_production": {
                "oil_rate": prod_data.get('oil_rate', 500) * 1.15,
                "gas_rate": prod_data.get('gas_rate', 5000) * 1.05,
                "water_rate": prod_data.get('water_rate', 200) * 0.90
            },
            "recommendations": [
                "Optimize choke size to 40/64\"",
                "Consider water shutoff treatment",
                "Monitor reservoir pressure closely"
            ],
            "potential_improvement": {
                "oil_rate_percent": 15.0,
                "annual_revenue_usd": 1920000
            },
            "optimal_choke_size": 40
        }


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🛢️ Reservoir Production Optimizer</h1>', 
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        page = st.selectbox(
            "Select Page",
            ["Dashboard", "Production Prediction", "Optimization", "Data Explorer", "Model Performance"]
        )
        
        st.markdown("---")
        
        # Load data
        df = load_sample_data()
        
        # Well selector
        available_wells = df['well_id'].unique()
        selected_well = st.selectbox("Select Well", available_wells)
        
        # Date range
        st.subheader("Date Range")
        date_range = st.date_input(
            "Select dates",
            value=(df['date'].min(), df['date'].max()),
            min_value=df['date'].min(),
            max_value=df['date'].max()
        )
    
    # Filter data
    well_data = df[df['well_id'] == selected_well].copy()
    if len(date_range) == 2:
        well_data = well_data[
            (well_data['date'] >= pd.to_datetime(date_range[0])) &
            (well_data['date'] <= pd.to_datetime(date_range[1]))
        ]
    
    # ========================================================================
    # Page Routing
    # ========================================================================
    
    if page == "Dashboard":
        show_dashboard(well_data, selected_well)
    elif page == "Production Prediction":
        show_prediction_page(well_data, selected_well)
    elif page == "Optimization":
        show_optimization_page(well_data, selected_well)
    elif page == "Data Explorer":
        show_data_explorer(well_data)
    elif page == "Model Performance":
        show_model_performance()


def show_dashboard(df, well_id):
    """Main dashboard page"""
    
    st.header(f"📊 Production Dashboard - {well_id}")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_oil = df['oil_rate'].mean()
        st.metric(
            "Avg Oil Rate",
            f"{avg_oil:.1f} bbl/day",
            delta=f"{(avg_oil - df['oil_rate'].iloc[0]):.1f}"
        )
    
    with col2:
        avg_gas = df['gas_rate'].mean()
        st.metric(
            "Avg Gas Rate",
            f"{avg_gas:.0f} Mcf/day",
            delta=f"{(avg_gas - df['gas_rate'].iloc[0]):.0f}"
        )
    
    with col3:
        current_wc = df['water_cut'].iloc[-1]
        st.metric(
            "Current Water Cut",
            f"{current_wc:.1f}%",
            delta=f"{(current_wc - df['water_cut'].iloc[0]):.1f}%",
            delta_color="inverse"
        )
    
    with col4:
        current_pressure = df['reservoir_pressure'].iloc[-1]
        st.metric(
            "Reservoir Pressure",
            f"{current_pressure:.0f} psi",
            delta=f"{(current_pressure - df['reservoir_pressure'].iloc[0]):.0f}"
        )
    
    st.markdown("---")
    
    # Production Trends
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Production Rates")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['oil_rate'], name='Oil', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df['date'], y=df['gas_rate']/10, name='Gas (÷10)', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df['date'], y=df['water_rate'], name='Water', line=dict(color='blue')))
        fig.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💧 Water Cut Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['water_cut'], fill='tozeroy', line=dict(color='cyan')))
        fig.update_layout(height=400, yaxis_title="Water Cut (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Reservoir Pressure
    st.subheader("🔴 Reservoir Pressure Decline")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['reservoir_pressure'], line=dict(color='orange', width=3)))
    fig.update_layout(height=300, yaxis_title="Pressure (psi)")
    st.plotly_chart(fig, use_container_width=True)


def show_prediction_page(df, well_id):
    """Production prediction page"""
    
    st.header(f"🔮 Production Prediction - {well_id}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        
        # Well properties
        with st.expander("Well Properties", expanded=True):
            porosity = st.number_input("Porosity", value=0.22, min_value=0.05, max_value=0.40)
            permeability = st.number_input("Permeability (mD)", value=150.0, min_value=0.1)
            net_pay = st.number_input("Net Pay (m)", value=25.0, min_value=1.0)
            initial_pressure = st.number_input("Initial Pressure (psi)", value=3500.0)
            choke_size = st.selectbox("Choke Size (64ths)", [16, 20, 24, 32, 40, 48, 64], index=3)
        
        # Current production
        with st.expander("Current Production", expanded=True):
            days_on_prod = st.number_input("Days on Production", value=365, min_value=0)
            current_oil = st.number_input("Current Oil Rate (bbl/day)", value=500.0)
            current_gas = st.number_input("Current Gas Rate (Mcf/day)", value=5000.0)
            current_water = st.number_input("Current Water Rate (bbl/day)", value=200.0)
            reservoir_pressure = st.number_input("Reservoir Pressure (psi)", value=3200.0)
            water_cut = st.number_input("Water Cut (%)", value=30.0, min_value=0.0, max_value=100.0)
        
        predict_button = st.button("🔮 Predict Production", type="primary")
    
    with col2:
        st.subheader("Prediction Results")
        
        if predict_button:
            with st.spinner("Making predictions..."):
                # Prepare data
                well_props = {
                    "porosity": porosity,
                    "permeability": permeability,
                    "net_pay": net_pay,
                    "initial_pressure": initial_pressure,
                    "reservoir_temperature": 180.0,
                    "measured_depth": 8500.0,
                    "true_vertical_depth": 7800.0,
                    "skin_factor": 2.0,
                    "tubing_diameter": 3.5,
                    "choke_size": choke_size,
                    "oil_api": 35.0,
                    "gas_gravity": 0.65
                }
                
                prod_data = {
                    "days_on_production": days_on_prod,
                    "oil_rate": current_oil,
                    "gas_rate": current_gas,
                    "water_rate": current_water,
                    "reservoir_pressure": reservoir_pressure,
                    "wellhead_pressure": 800.0,
                    "water_cut": water_cut,
                    "gor": current_gas / current_oil if current_oil > 0 else 1000
                }
                
                # Get prediction
                result = predict_production(well_props, prod_data)
                
                if result:
                    # Display predictions
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric(
                            "Predicted Oil Rate",
                            f"{result['predicted_oil_rate']:.1f} bbl/day",
                            delta=f"{(result['predicted_oil_rate'] - current_oil):.1f}"
                        )
                    
                    with col_b:
                        st.metric(
                            "Predicted Gas Rate",
                            f"{result['predicted_gas_rate']:.0f} Mcf/day",
                            delta=f"{(result['predicted_gas_rate'] - current_gas):.0f}"
                        )
                    
                    with col_c:
                        st.metric(
                            "Predicted Water Rate",
                            f"{result['predicted_water_rate']:.1f} bbl/day",
                            delta=f"{(result['predicted_water_rate'] - current_water):.1f}"
                        )
                    
                    # Confidence intervals
                    st.subheader("📊 Confidence Intervals (95%)")
                    
                    ci_data = pd.DataFrame({
                        'Metric': ['Oil Rate', 'Gas Rate', 'Water Rate'],
                        'Lower': [
                            result['confidence_interval_95']['oil_rate'][0],
                            result['confidence_interval_95']['gas_rate'][0],
                            result['confidence_interval_95']['water_rate'][0]
                        ],
                        'Upper': [
                            result['confidence_interval_95']['oil_rate'][1],
                            result['confidence_interval_95']['gas_rate'][1],
                            result['confidence_interval_95']['water_rate'][1]
                        ],
                        'Prediction': [
                            result['predicted_oil_rate'],
                            result['predicted_gas_rate'],
                            result['predicted_water_rate']
                        ]
                    })
                    
                    fig = go.Figure()
                    for i, row in ci_data.iterrows():
                        fig.add_trace(go.Box(
                            y=[row['Lower'], row['Prediction'], row['Upper']],
                            name=row['Metric'],
                            boxmean=True
                        ))
                    
                    fig.update_layout(height=400, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("✅ Prediction completed successfully!")
                else:
                    st.error("❌ Prediction failed. Please check API connection.")


def show_optimization_page(df, well_id):
    """Production optimization page"""
    
    st.header(f"⚡ Production Optimization - {well_id}")
    
    st.info("🎯 Get AI-powered recommendations to optimize your well's production.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Current Status")
        
        current_oil = st.number_input("Current Oil Rate (bbl/day)", value=500.0, key="opt_oil")
        current_gas = st.number_input("Current Gas Rate (Mcf/day)", value=5000.0, key="opt_gas")
        current_water = st.number_input("Current Water Rate (bbl/day)", value=200.0, key="opt_water")
        water_cut_opt = st.number_input("Water Cut (%)", value=40.0, min_value=0.0, max_value=100.0, key="opt_wc")
        
        optimize_button = st.button("⚡ Optimize Production", type="primary")
    
    with col2:
        if optimize_button:
            with st.spinner("Running optimization..."):
                # Mock well properties and data
                well_props = {
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
                
                prod_data = {
                    "days_on_production": 365,
                    "oil_rate": current_oil,
                    "gas_rate": current_gas,
                    "water_rate": current_water,
                    "reservoir_pressure": 3200.0,
                    "wellhead_pressure": 800.0,
                    "water_cut": water_cut_opt,
                    "gor": current_gas / current_oil if current_oil > 0 else 1000,
                    "choke_size": 32
                }
                
                result = optimize_production(well_props, prod_data)
                
                if result:
                    st.subheader("📈 Optimization Results")
                    
                    # Current vs Optimized
                    comparison_df = pd.DataFrame({
                        'Metric': ['Oil Rate', 'Gas Rate', 'Water Rate'],
                        'Current': [
                            result['current_production']['oil_rate'],
                            result['current_production']['gas_rate'],
                            result['current_production']['water_rate']
                        ],
                        'Optimized': [
                            result['optimized_production']['oil_rate'],
                            result['optimized_production']['gas_rate'],
                            result['optimized_production']['water_rate']
                        ]
                    })
                    
                    fig = go.Figure(data=[
                        go.Bar(name='Current', x=comparison_df['Metric'], y=comparison_df['Current'], marker_color='lightblue'),
                        go.Bar(name='Optimized', x=comparison_df['Metric'], y=comparison_df['Optimized'], marker_color='darkblue')
                    ])
                    fig.update_layout(barmode='group', height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Improvement metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric(
                            "Production Increase",
                            f"{result['potential_improvement']['oil_rate_percent']:.1f}%",
                            delta="Potential Gain"
                        )
                    with col_b:
                        st.metric(
                            "Annual Revenue Impact",
                            f"${result['potential_improvement']['annual_revenue_usd']:,.0f}",
                            delta="Additional Revenue"
                        )
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    for i, rec in enumerate(result['recommendations'], 1):
                        st.info(f"{i}. {rec}")
                    
                    st.success(f"✅ Optimal choke size: {result['optimal_choke_size']}/64\"")


def show_data_explorer(df):
    """Data explorer page"""
    
    st.header("🔍 Data Explorer")
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Raw data
    st.subheader("📋 Raw Data")
    st.dataframe(df, use_container_width=True)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Data as CSV",
        data=csv,
        file_name=f"production_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


def show_model_performance():
    """Model performance page"""
    
    st.header("🎯 Model Performance")
    
    # Mock performance metrics
    metrics_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'LightGBM', 'Neural Network'],
        'R² Score': [0.92, 0.94, 0.93, 0.91],
        'RMSE': [45.2, 38.7, 41.3, 48.9],
        'MAE': [32.1, 27.8, 29.5, 35.2],
        'Training Time (s)': [12.3, 8.7, 5.2, 45.6]
    })
    
    st.subheader("📊 Model Comparison")
    st.dataframe(metrics_df, use_container_width=True)
    
    # Best model
    best_idx = metrics_df['R² Score'].idxmax()
    best_model = metrics_df.iloc[best_idx]
    
    st.success(f"🏆 Best Model: **{best_model['Model']}** (R² = {best_model['R² Score']:.4f})")
    
    # Performance chart
    fig = go.Figure(data=[
        go.Bar(name='R² Score', x=metrics_df['Model'], y=metrics_df['R² Score'])
    ])
    fig.update_layout(height=400, yaxis_title="R² Score")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
