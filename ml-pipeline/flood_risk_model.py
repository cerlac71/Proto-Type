"""
Flood Risk Prediction Model
===========================
Lightweight XGBoost classifier to predict flood risk level (Low/Medium/High)
for the next 6 hours based on historical rainfall and river gauge data.

Includes:
- Feature engineering from time-series data
- Model training with synthetic labels
- Prediction function for API exposure
- FastAPI wrapper for Spring Boot integration
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import joblib
import os
from datetime import datetime, timedelta

# ML Libraries
try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score
except ImportError:
    print("Installing required packages...")
    os.system("pip install xgboost scikit-learn joblib fastapi uvicorn pydantic")
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score

# FastAPI for REST API
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Installing FastAPI dependencies...")
    os.system("pip install fastapi uvicorn pydantic")
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn


class FloodRiskModel:
    """
    Lightweight flood risk prediction model using XGBoost.
    
    Predicts risk level (Low/Medium/High) for next 6 hours based on:
    - Recent rainfall patterns
    - River level trends
    - Rate of change metrics
    """
    
    MODEL_PATH = "models/flood_risk_model.pkl"
    ENCODER_PATH = "models/label_encoder.pkl"
    
    # Risk thresholds (configurable)
    RISK_THRESHOLDS = {
        'rainfall_critical': 50.0,  # mm/hour
        'river_level_critical': 8.0,  # meters
        'rate_of_change_critical': 0.5  # m/hour
    }
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            'rainfall_mm',
            'river_level_m',
            'rainfall_3h_sum',
            'rainfall_6h_sum',
            'river_level_3h_avg',
            'river_level_6h_avg',
            'rainfall_rate',
            'river_level_rate',
            'hour_of_day',
            'day_of_week'
        ]
        self.is_trained = False
    
    def _generate_synthetic_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate synthetic flood risk labels based on heuristic rules.
        In production, this would be actual historical flood events.
        
        Logic:
        - High: rainfall > 50mm OR river_level > 8m OR rapid rise
        - Medium: rainfall > 25mm OR river_level > 6m
        - Low: otherwise
        """
        labels = []
        
        for idx, row in df.iterrows():
            rainfall = row['rainfall_mm']
            river_level = row['river_level_m']
            
            # Calculate rate of change if possible
            if idx > 0:
                prev_river = df.iloc[idx - 1]['river_level_m']
                river_rate = abs(river_level - prev_river)
            else:
                river_rate = 0
            
            # Heuristic labeling
            if rainfall > self.RISK_THRESHOLDS['rainfall_critical'] or \
               river_level > self.RISK_THRESHOLDS['river_level_critical'] or \
               river_rate > self.RISK_THRESHOLDS['rate_of_change_critical']:
                labels.append('High')
            elif rainfall > 25.0 or river_level > 6.0:
                labels.append('Medium')
            else:
                labels.append('Low')
        
        return pd.Series(labels)
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-series features from raw data.
        """
        feature_df = df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in feature_df.columns:
            feature_df['timestamp'] = pd.to_datetime(feature_df['timestamp'])
            feature_df = feature_df.sort_values(['station_id', 'timestamp'])
        
        # Rolling statistics per station
        feature_df['rainfall_3h_sum'] = feature_df.groupby('station_id')['rainfall_mm'].transform(
            lambda x: x.rolling(window=3, min_periods=1).sum()
        )
        feature_df['rainfall_6h_sum'] = feature_df.groupby('station_id')['rainfall_mm'].transform(
            lambda x: x.rolling(window=6, min_periods=1).sum()
        )
        feature_df['river_level_3h_avg'] = feature_df.groupby('station_id')['river_level_m'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        feature_df['river_level_6h_avg'] = feature_df.groupby('station_id')['river_level_m'].transform(
            lambda x: x.rolling(window=6, min_periods=1).mean()
        )
        
        # Rate of change
        feature_df['rainfall_rate'] = feature_df.groupby('station_id')['rainfall_mm'].transform(
            lambda x: x.diff().fillna(0)
        )
        feature_df['river_level_rate'] = feature_df.groupby('station_id')['river_level_m'].transform(
            lambda x: x.diff().fillna(0).abs()
        )
        
        # Temporal features
        if 'timestamp' in feature_df.columns:
            feature_df['hour_of_day'] = feature_df['timestamp'].dt.hour
            feature_df['day_of_week'] = feature_df['timestamp'].dt.dayofweek
        else:
            feature_df['hour_of_day'] = 12  # Default
            feature_df['day_of_week'] = 1   # Default
        
        # Fill any remaining NaN values
        feature_df[self.feature_columns] = feature_df[self.feature_columns].fillna(0)
        
        return feature_df
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Train the XGBoost model on ingested data.
        
        Args:
            df: DataFrame with columns [timestamp, station_id, rainfall_mm, river_level_m]
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary with training metrics
        """
        print("🔄 Engineering features...")
        feature_df = self._engineer_features(df)
        
        print("🔄 Generating synthetic labels...")
        y = self._generate_synthetic_labels(feature_df)
        
        # Prepare features
        X = feature_df[self.feature_columns]
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"📊 Dataset size: {len(X)} samples")
        print(f"📈 Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        print(f"📚 Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Train XGBoost model
        print("🤖 Training XGBoost model...")
        self.model = xgb.XGBClassifier(
            n_estimators=50,           # Lightweight for demo
            max_depth=4,
            learning_rate=0.1,
            objective='multi:softprob',
            num_class=3,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained! Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Save model
        self._save_model()
        self.is_trained = True
        
        return {
            'accuracy': accuracy,
            'classes': list(self.label_encoder.classes_),
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
    
    def _save_model(self):
        """Save model and encoder to disk."""
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, self.MODEL_PATH)
        joblib.dump(self.label_encoder, self.ENCODER_PATH)
        print(f"💾 Model saved to {self.MODEL_PATH}")
        print(f"💾 Encoder saved to {self.ENCODER_PATH}")
    
    def load_model(self) -> bool:
        """Load pre-trained model from disk."""
        if os.path.exists(self.MODEL_PATH) and os.path.exists(self.ENCODER_PATH):
            self.model = joblib.load(self.MODEL_PATH)
            self.label_encoder = joblib.load(self.ENCODER_PATH)
            self.is_trained = True
            print(f"✅ Model loaded from {self.MODEL_PATH}")
            return True
        return False
    
    def predict(self, df: pd.DataFrame) -> List[Dict]:
        """
        Predict flood risk for given data.
        
        Args:
            df: DataFrame with columns [timestamp, station_id, rainfall_mm, river_level_m]
            
        Returns:
            List of predictions with station_id, timestamp, risk_level, confidence
        """
        if not self.is_trained:
            if not self.load_model():
                raise RuntimeError("No trained model available. Call train() first.")
        
        # Engineer features
        feature_df = self._engineer_features(df)
        X = feature_df[self.feature_columns]
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Decode labels
        risk_levels = self.label_encoder.inverse_transform(predictions)
        
        # Build results
        results = []
        for idx, (risk, prob) in enumerate(zip(risk_levels, probabilities)):
            result = {
                'station_id': df.iloc[idx]['station_id'] if 'station_id' in df.columns else 'unknown',
                'timestamp': str(df.iloc[idx]['timestamp']) if 'timestamp' in df.columns else str(datetime.now()),
                'risk_level': risk,
                'confidence': float(max(prob)),
                'probabilities': {
                    'Low': float(prob[0]) if len(prob) > 0 else 0.0,
                    'Medium': float(prob[1]) if len(prob) > 1 else 0.0,
                    'High': float(prob[2]) if len(prob) > 2 else 0.0
                }
            }
            results.append(result)
        
        return results
    
    def predict_single(self, station_id: str, rainfall_mm: float, 
                       river_level_m: float, timestamp: Optional[str] = None) -> Dict:
        """
        Predict flood risk for a single observation.
        
        Args:
            station_id: Station identifier
            rainfall_mm: Rainfall in mm
            river_level_m: River level in meters
            timestamp: Optional timestamp string
            
        Returns:
            Prediction dictionary
        """
        # Create minimal DataFrame
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        df = pd.DataFrame([{
            'station_id': station_id,
            'timestamp': timestamp,
            'rainfall_mm': rainfall_mm,
            'river_level_m': river_level_m
        }])
        
        predictions = self.predict(df)
        return predictions[0] if predictions else None


# FastAPI Application
app = FastAPI(
    title="Flood Risk Prediction API",
    description="Lightweight API for flood risk prediction using XGBoost",
    version="1.0.0"
)

# Global model instance
model_instance: Optional[FloodRiskModel] = None


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""
    station_id: str
    rainfall_mm: float
    river_level_m: float
    timestamp: Optional[str] = None


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""
    data: List[Dict]


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model_instance
    model_instance = FloodRiskModel()
    if not model_instance.load_model():
        print("⚠️ No pre-trained model found. Model will need training via /train endpoint.")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Flood Risk Prediction API",
        "status": "running",
        "model_loaded": model_instance.is_trained if model_instance else False
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_ready": model_instance.is_trained if model_instance else False,
        "classes": model_instance.label_encoder.classes_.tolist() if model_instance and model_instance.is_trained else []
    }


@app.post("/predict")
async def predict_risk(request: PredictionRequest):
    """Predict flood risk for a single observation."""
    if not model_instance or not model_instance.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    try:
        prediction = model_instance.predict_single(
            station_id=request.station_id,
            rainfall_mm=request.rainfall_mm,
            river_level_m=request.river_level_m,
            timestamp=request.timestamp
        )
        return JSONResponse(content=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch_risk(request: BatchPredictionRequest):
    """Predict flood risk for multiple observations."""
    if not model_instance or not model_instance.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained")
    
    try:
        df = pd.DataFrame(request.data)
        predictions = model_instance.predict(df)
        return JSONResponse(content={"predictions": predictions, "count": len(predictions)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train_model():
    """
    Train model using synthetic data (for demo purposes).
    In production, this would accept uploaded training data.
    """
    if not model_instance:
        raise HTTPException(status_code=503, detail="Model instance not initialized")
    
    try:
        # Import data ingestion for synthetic data
        from data_ingestion import generate_synthetic_data, ingest_data
        
        print("🔄 Generating synthetic training data...")
        df_gen = generate_synthetic_data(num_stations=10, days=30, seed=42)
        df_gen.to_csv("data/training_data.csv", index=False)
        
        print("🔄 Loading and cleaning data...")
        df = ingest_data(source="csv", file_path="data/training_data.csv")
        
        print("🤖 Training model...")
        metrics = model_instance.train(df)
        
        return JSONResponse(content={
            "status": "success",
            "message": "Model trained successfully",
            "metrics": metrics
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    print(f"🚀 Starting Flood Risk Prediction API on {host}:{port}")
    print(f"📡 Endpoints:")
    print(f"   GET  /          - Health check")
    print(f"   GET  /health    - Detailed health")
    print(f"   POST /predict   - Single prediction")
    print(f"   POST /predict/batch - Batch predictions")
    print(f"   POST /train     - Train model (demo)")
    print(f"\n🧪 Test with:")
    print(f"   curl -X POST http://localhost:{port}/predict \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{{\"station_id\": \"ST001\", \"rainfall_mm\": 45.5, \"river_level_m\": 7.2}}'")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import sys
    
    # Check if we should train first
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        print("🎯 Training mode: Building model with synthetic data...")
        
        # Import and generate training data
        from data_ingestion import generate_synthetic_data, ingest_data
        
        # Generate larger dataset for training
        df_gen = generate_synthetic_data(num_stations=10, days=30, seed=42)
        df_gen.to_csv("data/training_data.csv", index=False)
        
        # Load and clean
        df = ingest_data(source="csv", file_path="data/training_data.csv")
        
        # Train model
        model = FloodRiskModel()
        metrics = model.train(df)
        
        print("\n✅ Training complete!")
        print(f"📊 Final Accuracy: {metrics['accuracy']:.2%}")
        print(f"🏷️ Classes: {metrics['classes']}")
        print(f"📈 Top Features: {sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:3]}")
        
    elif len(sys.argv) > 1 and sys.argv[1] == "--api":
        # Run API server
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        run_api(port=port)
        
    else:
        # Demo mode: train and show sample prediction
        print("🎯 Demo Mode: Training and testing model...")
        
        from data_ingestion import generate_synthetic_data, ingest_data
        
        # Generate data
        df_gen = generate_synthetic_data(num_stations=5, days=10, seed=123)
        df_gen.to_csv("data/demo_data.csv", index=False)
        df = ingest_data(source="csv", file_path="data/demo_data.csv")
        
        # Train
        model = FloodRiskModel()
        model.train(df)
        
        # Test predictions
        print("\n🧪 Sample Predictions:")
        test_cases = [
            ("ST001", 10.5, 4.2, "Normal conditions"),
            ("ST002", 55.0, 7.8, "Heavy rainfall"),
            ("ST003", 30.0, 8.5, "High river level"),
            ("ST001", 60.0, 9.2, "Critical conditions"),
        ]
        
        for station, rain, river, desc in test_cases:
            pred = model.predict_single(station, rain, river)
            print(f"  {desc}: {pred['risk_level']} ({pred['confidence']:.1%} confidence)")
        
        print(f"\n💾 Model saved. Run with --api to start server.")
