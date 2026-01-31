"""
FastAPI Application for Reservoir Production Optimization
==========================================================
REST API for production prediction and optimization recommendations.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import os

# Initialize FastAPI
app = FastAPI(
    title="Reservoir Production Optimization API",
    description="ML-powered API for oil & gas production prediction and optimization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class WellProperties(BaseModel):
    """Well static properties"""
    porosity: float = Field(..., ge=0, le=1, description="Porosity (fraction)")
    permeability: float = Field(..., gt=0, description="Permeability (mD)")
    net_pay: float = Field(..., gt=0, description="Net pay thickness (m)")
    initial_pressure: float = Field(..., gt=0, description="Initial reservoir pressure (psi)")
    reservoir_temperature: float = Field(..., description="Reservoir temperature (°F)")
    measured_depth: float = Field(..., gt=0, description="Measured depth (ft)")
    true_vertical_depth: float = Field(..., gt=0, description="TVD (ft)")
    skin_factor: float = Field(..., description="Skin factor (dimensionless)")
    tubing_diameter: float = Field(..., gt=0, description="Tubing diameter (inches)")
    choke_size: int = Field(..., gt=0, description="Choke size (64ths of inch)")
    oil_api: float = Field(..., gt=0, description="Oil API gravity")
    gas_gravity: float = Field(..., gt=0, description="Gas specific gravity")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


class ProductionData(BaseModel):
    """Current production data"""
    days_on_production: int = Field(..., ge=0)
    oil_rate: Optional[float] = Field(None, ge=0)
    gas_rate: Optional[float] = Field(None, ge=0)
    water_rate: Optional[float] = Field(None, ge=0)
    reservoir_pressure: float = Field(..., gt=0)
    wellhead_pressure: float = Field(..., gt=0)
    water_cut: float = Field(..., ge=0, le=100)
    gor: float = Field(..., ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
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


class PredictionRequest(BaseModel):
    """Prediction request"""
    well_properties: WellProperties
    production_data: ProductionData


class PredictionResponse(BaseModel):
    """Prediction response"""
    predicted_oil_rate: float
    predicted_gas_rate: float
    predicted_water_rate: float
    confidence_interval_95: Dict[str, List[float]]
    prediction_timestamp: str
    model_version: str


class OptimizationRequest(BaseModel):
    """Optimization request"""
    well_properties: WellProperties
    production_data: ProductionData
    optimization_target: str = Field(default="oil_rate", description="oil_rate or npv")
    constraints: Optional[Dict] = None


class OptimizationResponse(BaseModel):
    """Optimization response"""
    current_production: Dict[str, float]
    optimized_production: Dict[str, float]
    recommendations: List[str]
    potential_improvement: Dict[str, float]
    optimal_choke_size: int


# ============================================================================
# Global Variables
# ============================================================================

MODEL = None
SCALER = None
FEATURE_NAMES = None
MODEL_VERSION = "1.0.0"


# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def load_model():
    """Load ML model on startup"""
    global MODEL, SCALER, FEATURE_NAMES
    
    try:
        # Load model (try XGBoost first, fallback to Random Forest)
        model_paths = [
            'models/xgboost_model.pkl',
            'models/random_forest_model.pkl',
            'models/best_model.pkl'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                MODEL = joblib.load(path)
                print(f"✓ Model loaded from {path}")
                break
        
        if MODEL is None:
            print("⚠ Warning: No pre-trained model found. Using mock predictions.")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        MODEL = None


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Reservoir Production Optimization API",
        "version": MODEL_VERSION,
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "optimize": "/optimize",
            "batch_predict": "/batch-predict"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.utcnow().isoformat(),
        "version": MODEL_VERSION
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_production(request: PredictionRequest):
    """
    Predict production rates
    
    Returns predicted oil, gas, and water rates with confidence intervals.
    """
    try:
        # Prepare features
        features = prepare_features(request.well_properties, request.production_data)
        
        # Make prediction
        if MODEL is not None:
            prediction = MODEL.predict(features)[0]
        else:
            # Mock prediction if model not loaded
            prediction = request.production_data.oil_rate or 500.0
        
        # Calculate confidence intervals (simplified - would use actual model uncertainty)
        ci_lower = prediction * 0.85
        ci_upper = prediction * 1.15
        
        # Predict gas and water based on ratios
        predicted_gas = prediction * request.production_data.gor / 1000
        predicted_water = prediction * request.production_data.water_cut / (100 - request.production_data.water_cut)
        
        return PredictionResponse(
            predicted_oil_rate=float(prediction),
            predicted_gas_rate=float(predicted_gas),
            predicted_water_rate=float(predicted_water),
            confidence_interval_95={
                "oil_rate": [float(ci_lower), float(ci_upper)],
                "gas_rate": [float(predicted_gas * 0.85), float(predicted_gas * 1.15)],
                "water_rate": [float(predicted_water * 0.85), float(predicted_water * 1.15)]
            },
            prediction_timestamp=datetime.utcnow().isoformat(),
            model_version=MODEL_VERSION
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_production(request: OptimizationRequest):
    """
    Optimize production parameters
    
    Returns recommendations for optimizing production.
    """
    try:
        current = request.production_data
        well = request.well_properties
        
        recommendations = []
        
        # Optimize choke size
        optimal_choke = optimize_choke_size(well, current)
        
        # Generate recommendations
        if current.water_cut > 70:
            recommendations.append("High water cut detected. Consider water shutoff treatment.")
        
        if current.reservoir_pressure < 0.5 * well.initial_pressure:
            recommendations.append("Low reservoir pressure. Consider pressure maintenance (water/gas injection).")
        
        if well.skin_factor > 5:
            recommendations.append("High skin factor. Consider acid stimulation or fracturing.")
        
        if current.gor > 2000:
            recommendations.append("High GOR. Monitor for gas coning. Consider reducing drawdown.")
        
        if optimal_choke != current.choke_size:
            recommendations.append(f"Adjust choke size from {current.choke_size}/64\" to {optimal_choke}/64\"")
        
        # Calculate potential improvement (simplified)
        potential_oil_improvement = 15.0 if len(recommendations) > 2 else 5.0
        
        optimized_oil = (current.oil_rate or 0) * (1 + potential_oil_improvement / 100)
        
        return OptimizationResponse(
            current_production={
                "oil_rate": current.oil_rate or 0,
                "gas_rate": current.gas_rate or 0,
                "water_rate": current.water_rate or 0
            },
            optimized_production={
                "oil_rate": optimized_oil,
                "gas_rate": (current.gas_rate or 0) * 1.05,
                "water_rate": (current.water_rate or 0) * 0.95
            },
            recommendations=recommendations,
            potential_improvement={
                "oil_rate_percent": potential_oil_improvement,
                "annual_revenue_usd": potential_oil_improvement * (current.oil_rate or 0) * 365 * 70  # $70/bbl
            },
            optimal_choke_size=optimal_choke
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")


@app.post("/batch-predict")
async def batch_predict(requests: List[PredictionRequest]):
    """Batch prediction endpoint"""
    
    results = []
    for req in requests:
        try:
            result = await predict_production(req)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})
    
    return {"predictions": results, "count": len(results)}


# ============================================================================
# Helper Functions
# ============================================================================

def prepare_features(well: WellProperties, production: ProductionData) -> np.ndarray:
    """Prepare features for model prediction"""
    
    # Create feature dictionary
    features = {
        'porosity': well.porosity,
        'permeability': well.permeability,
        'net_pay': well.net_pay,
        'initial_pressure': well.initial_pressure,
        'reservoir_temperature': well.reservoir_temperature,
        'measured_depth': well.measured_depth,
        'true_vertical_depth': well.true_vertical_depth,
        'skin_factor': well.skin_factor,
        'tubing_diameter': well.tubing_diameter,
        'choke_size': well.choke_size,
        'oil_api': well.oil_api,
        'gas_gravity': well.gas_gravity,
        'days_on_production': production.days_on_production,
        'gas_rate': production.gas_rate or 0,
        'water_rate': production.water_rate or 0,
        'reservoir_pressure': production.reservoir_pressure,
        'wellhead_pressure': production.wellhead_pressure,
        'water_cut': production.water_cut,
        'gor': production.gor
    }
    
    # Convert to array (in practice, would match training features exactly)
    feature_array = np.array(list(features.values())).reshape(1, -1)
    
    return feature_array


def optimize_choke_size(well: WellProperties, production: ProductionData) -> int:
    """Optimize choke size based on current conditions"""
    
    # Simplified optimization logic
    current_choke = production.choke_size
    drawdown = production.reservoir_pressure - production.wellhead_pressure
    
    # Available choke sizes
    choke_sizes = [16, 20, 24, 32, 40, 48, 64]
    
    # Optimize based on drawdown and production
    if drawdown > 2000:  # High drawdown - reduce choke
        optimal = max([c for c in choke_sizes if c < current_choke], default=current_choke)
    elif drawdown < 500:  # Low drawdown - increase choke
        optimal = min([c for c in choke_sizes if c > current_choke], default=current_choke)
    else:
        optimal = current_choke
    
    return optimal


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.utcnow().isoformat()}
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
