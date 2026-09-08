"""
Flood Risk API - In-Process Prediction Service
===============================================
FastAPI endpoint that loads latest station readings from CSV,
computes features using the flood_risk_model module, and returns
risk predictions without requiring a separate ML service call.

Endpoints:
    GET /api/flood-risk/{station_id} - Get flood risk for a station
    
Error Handling:
    404 - Station ID not found in data
    503 - Data file or model file missing (with helpful message)
    500 - Unexpected prediction failure
"""

import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd

# Import model components from pipeline
try:
    from flood_risk_model import FloodRiskModel
except ImportError as e:
    print(f"Warning: Could not import flood_risk_model: {e}")
    FloodRiskModel = None


# Constants
DATA_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "sample_readings.csv"
)

# Fallback to sample_data.csv if sample_readings.csv doesn't exist
FALLBACK_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "sample_data.csv"
)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models"
)

MODEL_FILE = os.path.join(MODEL_DIR, "flood_risk_model.pkl")
ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.pkl")


app = FastAPI(
    title="Flood Risk Prediction API",
    description="In-process flood risk prediction using loaded CSV data and pre-trained model",
    version="1.0.0"
)


def get_data_file_path() -> Optional[str]:
    """Get the path to the data file, checking both primary and fallback locations."""
    if os.path.exists(DATA_FILE_PATH):
        return DATA_FILE_PATH
    elif os.path.exists(FALLBACK_DATA_FILE):
        return FALLBACK_DATA_FILE
    return None


def load_latest_reading(station_id: str) -> pd.DataFrame:
    """
    Load the latest reading for a specific station from the CSV file.
    
    Args:
        station_id: The station identifier to look up
        
    Returns:
        DataFrame with the latest reading for the station
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If station_id not found in data
    """
    data_path = get_data_file_path()
    
    if data_path is None:
        raise FileNotFoundError(
            f"Data file not found. Expected at '{DATA_FILE_PATH}' or '{FALLBACK_DATA_FILE}'. "
            "Please run data_ingestion.py first to generate sample data."
        )
    
    # Load CSV
    df = pd.read_csv(data_path)
    
    # Validate required columns
    required_columns = ['timestamp', 'station_id', 'rainfall_mm', 'river_level_m']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in data file: {missing_cols}")
    
    # Filter for the specific station
    station_data = df[df['station_id'] == station_id].copy()
    
    if station_data.empty:
        available_stations = df['station_id'].unique().tolist()
        raise ValueError(
            f"Station '{station_id}' not found in data. "
            f"Available stations: {available_stations}"
        )
    
    # Get the latest reading (sort by timestamp and take last)
    station_data['timestamp'] = pd.to_datetime(station_data['timestamp'])
    station_data = station_data.sort_values('timestamp')
    latest_reading = station_data.tail(1)
    
    return latest_reading


def get_or_create_model() -> FloodRiskModel:
    """
    Get or create a FloodRiskModel instance.
    
    Returns:
        FloodRiskModel instance (loaded if available)
        
    Raises:
        FileNotFoundError: If model files don't exist
    """
    if FloodRiskModel is None:
        raise ImportError(
            "Could not import FloodRiskModel. "
            "Please ensure flood_risk_model.py is available and dependencies are installed."
        )
    
    model = FloodRiskModel()
    
    # Check if model files exist
    if not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODER_FILE):
        raise FileNotFoundError(
            f"Trained model not found. Expected at '{MODEL_FILE}' and '{ENCODER_FILE}'. "
            "Please run flood_risk_model.py with --train flag first to train the model."
        )
    
    # Load the pre-trained model
    if not model.load_model():
        raise RuntimeError(
            "Failed to load pre-trained model. "
            "Please run flood_risk_model.py with --train flag to retrain the model."
        )
    
    return model


@app.get("/")
async def root():
    """Health check endpoint."""
    data_path = get_data_file_path()
    model_exists = os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE)
    
    return {
        "service": "Flood Risk Prediction API",
        "status": "running",
        "data_file_ready": data_path is not None,
        "model_ready": model_exists,
        "endpoints": {
            "flood_risk": "/api/flood-risk/{station_id}"
        }
    }


@app.get("/api/flood-risk/{station_id}")
async def get_flood_risk(station_id: str):
    """
    Get flood risk prediction for a specific station.
    
    Loads the latest reading from CSV, computes features using the
    flood_risk_model module, and returns the risk level prediction.
    
    Args:
        station_id: The station identifier (e.g., "STN_001")
        
    Returns:
        JSON with stationId, riskLevel, predictedRainfallMM, timestamp
        
    Raises:
        404: If station_id not found in data
        503: If data file or model file doesn't exist
        500: For unexpected prediction failures
    """
    try:
        # Step 1: Load latest reading for the station
        try:
            latest_data = load_latest_reading(station_id)
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=str(e)
            )
        except ValueError as e:
            raise HTTPException(
                status_code=404,
                detail=str(e)
            )
        
        # Step 2: Load the trained model
        try:
            model = get_or_create_model()
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail=str(e)
            )
        except ImportError as e:
            raise HTTPException(
                status_code=503,
                detail=str(e)
            )
        
        # Step 3: Make prediction using the model's predict method
        # This internally calls _engineer_features (build_features equivalent)
        predictions = model.predict(latest_data)
        
        if not predictions or len(predictions) == 0:
            raise RuntimeError("Model returned empty predictions")
        
        prediction = predictions[0]
        
        # Step 4: Format response as required
        result = {
            "stationId": station_id,
            "riskLevel": prediction['risk_level'],
            "predictedRainfallMM": float(latest_data['rainfall_mm'].iloc[0]) if not pd.isna(latest_data['rainfall_mm'].iloc[0]) else 0.0,
            "timestamp": prediction['timestamp']
        }
        
        return JSONResponse(content=result)
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Catch any unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected prediction failure: {str(e)}"
        )


@app.get("/api/flood-risk/{station_id}/details")
async def get_flood_risk_detailed(station_id: str):
    """
    Get detailed flood risk prediction including confidence and probabilities.
    
    Args:
        station_id: The station identifier
        
    Returns:
        Detailed JSON with all prediction information
    """
    try:
        # Load latest reading
        try:
            latest_data = load_latest_reading(station_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        # Load model
        try:
            model = get_or_create_model()
        except (FileNotFoundError, ImportError) as e:
            raise HTTPException(status_code=503, detail=str(e))
        
        # Get prediction
        predictions = model.predict(latest_data)
        
        if not predictions:
            raise RuntimeError("Model returned empty predictions")
        
        prediction = predictions[0]
        
        # Return detailed response
        return JSONResponse(content={
            "stationId": station_id,
            "riskLevel": prediction['risk_level'],
            "predictedRainfallMM": float(latest_data['rainfall_mm'].iloc[0]) if not pd.isna(latest_data['rainfall_mm'].iloc[0]) else 0.0,
            "riverLevelM": float(latest_data['river_level_m'].iloc[0]) if not pd.isna(latest_data['river_level_m'].iloc[0]) else 0.0,
            "timestamp": prediction['timestamp'],
            "confidence": prediction['confidence'],
            "probabilities": prediction['probabilities'],
            "dataUsed": {
                "rainfall_mm": prediction.get('rainfall_mm', float(latest_data['rainfall_mm'].iloc[0])),
                "river_level_m": prediction.get('river_level_m', float(latest_data['river_level_m'].iloc[0]))
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected prediction failure: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Flood Risk Prediction API...")
    print(f"📁 Data file: {get_data_file_path() or 'NOT FOUND'}")
    print(f"🤖 Model file: {MODEL_FILE if os.path.exists(MODEL_FILE) else 'NOT FOUND'}")
    print("\n📡 Endpoints:")
    print("   GET /                          - Health check")
    print("   GET /api/flood-risk/{station_id} - Get risk prediction")
    print("   GET /api/flood-risk/{station_id}/details - Get detailed prediction")
    print("\n🌐 Running on http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
