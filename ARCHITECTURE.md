# System Architecture: AI/ML-Based Heavy Rainfall Early Warning & Flood Inundation Prediction System

## Overview
A 36-hour hackathon prototype for SIH26071 that integrates satellite data, radar imagery, and weather station readings to predict heavy rainfall events and forecast flood inundation zones. The system provides real-time alerts through a dashboard interface.

---

## High-Level Architecture Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Data Sources  │────▶│  Python ML/Data  │────▶│  Spring Boot    │
│                 │     │     Pipeline     │     │   Backend API   │
│ • Satellite     │     │                  │     │                 │
│ • Radar         │     │ • Data Ingestion │     │ • REST APIs     │
│ • Weather Stns  │     │ • Preprocessing  │     │ • Alert Logic   │
│ • Historical DB │     │ • ML Prediction  │     │ • Data Serving  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │   React/Vue.js  │
                                               │    Frontend     │
                                               │                 │
                                               │ • Alert Dashboard
                                               │ • Map Visualizations
                                               │ • Historical Charts
                                               └─────────────────┘
```

---

## Module Breakdown

### 1. Python ML/Data Pipeline (`ml-pipeline/`)

**Responsibilities:**
- **Data Ingestion**: Fetch real-time and historical data from satellites (IMD, NASA GPM), radar stations, and weather APIs
- **Preprocessing**: Clean, normalize, and feature-engineer multi-source data
- **ML Models**: 
  - Rainfall prediction model (LSTM/GRU or XGBoost for time-series forecasting)
  - Flood inundation model (CNN for spatial analysis or hydraulic simulation approximation)
- **Prediction Engine**: Generate 6hr, 12hr, 24hr, and 36hr forecasts
- **Data Export**: Store predictions in formats consumable by the backend (JSON/PostgreSQL)

**Key Files:**
```
ml-pipeline/
├── config/
│   ├── data_sources.yaml      # API endpoints, credentials, refresh intervals
│   └── model_params.yaml      # Hyperparameters, thresholds
├── data_ingestion/
│   ├── satellite_fetcher.py   # IMERG, GPM, INSAT data collectors
│   ├── radar_fetcher.py       # Doppler radar image/data parser
│   ├── weather_station.py     # IMD/aws weather station API client
│   └── historical_loader.py   # Load historical training data
├── preprocessing/
│   ├── cleaner.py             # Handle missing values, outliers
│   ├── normalizer.py          # Scale features, coordinate alignment
│   └── feature_engineer.py    # Create derived features (rainfall intensity, accumulation)
├── models/
│   ├── rainfall_predictor.py  # Time-series forecasting model
│   ├── flood_inundation.py    # Spatial flood extent predictor
│   └── model_trainer.py       # Training pipeline with cross-validation
├── prediction/
│   ├── forecast_engine.py     # Run predictions on latest data
│   └── threshold_checker.py   # Determine alert levels based on predictions
├── output/
│   ├── predictions.json       # Latest forecasts (consumed by backend)
│   └── flood_maps/            # GeoJSON/PNG flood inundation overlays
├── requirements.txt           # Python dependencies
└── main.py                    # Orchestration script (run full pipeline)
```

---

### 2. Spring Boot Backend (`backend/`)

**Responsibilities:**
- **REST API Layer**: Serve prediction data, historical trends, and alert status to frontend
- **Alert Logic**: Evaluate prediction thresholds and trigger alert notifications
- **Data Aggregation**: Combine ML outputs with static geographic data (districts, rivers, settlements)
- **Database Integration**: Store predictions, alerts, and user feedback
- **Scheduled Tasks**: Poll ML pipeline outputs at regular intervals

**Key Files:**
```
backend/
├── src/main/java/com/floodwarn/
│   ├── FloodWarnApplication.java
│   ├── config/
│   │   ├── SecurityConfig.java      # Basic auth/CORS setup
│   │   └── SchedulerConfig.java     # Cron jobs for data refresh
│   ├── controller/
│   │   ├── AlertController.java     # GET /api/alerts, POST /api/alerts/acknowledge
│   │   ├── PredictionController.java# GET /api/predictions/{region}/{hours}
│   │   ├── HistoricalController.java# GET /api/historical/{region}/{dateRange}
│   │   └── MapController.java       # GET /api/flood-map/{region}
│   ├── service/
│   │   ├── PredictionService.java   # Fetch/process ML predictions
│   │   ├── AlertService.java        # Evaluate thresholds, manage alert lifecycle
│   │   ├── DataService.java         # Aggregate multi-source data
│   │   └── NotificationService.java # Send SMS/email alerts (mock for hackathon)
│   ├── repository/
│   │   ├── PredictionRepository.java
│   │   ├── AlertRepository.java
│   │   └── RegionRepository.java
│   ├── model/
│   │   ├── Prediction.java          # Entity: timestamp, region, rainfall_mm, flood_risk
│   │   ├── Alert.java               # Entity: level, message, status, created_at
│   │   └── Region.java              # Entity: name, coordinates, population, vulnerability_score
│   └── dto/
│       ├── PredictionResponse.java
│       ├── AlertResponse.java
│       └── MapOverlayResponse.java
├── src/main/resources/
│   ├── application.properties       # DB config, API ports, scheduler intervals
│   ├── application-dev.properties   # Hackathon demo settings
│   └── data.sql                     # Seed data for regions, historical samples
├── pom.xml                          # Maven dependencies (Spring Web, JPA, PostgreSQL driver)
└── README.md                        # Setup instructions
```

---

### 3. Frontend Dashboard (`frontend/`)

**Responsibilities:**
- **Real-Time Dashboard**: Display current alert status, predicted rainfall, and flood risk maps
- **Interactive Maps**: Show inundation zones overlaid on geographic maps (Leaflet/Mapbox)
- **Historical Visualization**: Charts showing past rainfall vs. predictions
- **Alert Management**: View active alerts, acknowledge them, filter by severity
- **Responsive Design**: Mobile-friendly for field workers

**Key Files:**
```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Dashboard.jsx            # Main layout with widgets
│   │   ├── AlertPanel.jsx           # List of active alerts with severity colors
│   │   ├── RainfallChart.jsx        # Time-series chart (Recharts/Chart.js)
│   │   ├── FloodMap.jsx             # Interactive map with inundation overlays
│   │   ├── RegionSelector.jsx       # Dropdown/filter by district/state
│   │   └── HistoricalComparison.jsx # Past vs. predicted rainfall comparison
│   ├── services/
│   │   ├── api.js                   # Axios calls to Spring Boot endpoints
│   │   └── websocket.js             # Optional: real-time alert pushes
│   ├── hooks/
│   │   ├── usePredictions.js        # Fetch prediction data
│   │   └── useAlerts.js             # Fetch/manage alert state
│   ├── utils/
│   │   ├── formatters.js            # Date/number formatting helpers
│   │   └── constants.js             # Alert level colors, thresholds
│   ├── App.jsx
│   ├── index.jsx
│   └── styles/
│       └── global.css
├── package.json                     # React + Vite dependencies
├── vite.config.js
└── README.md                        # Setup instructions
```

---

## Data Flow

### 1. Data Ingestion → ML Pipeline
```
[Satellite/Radar/Weather APIs] 
       │
       ▼
[ml-pipeline/data_ingestion/*.py] → Raw data stored temporarily
       │
       ▼
[ml-pipeline/preprocessing/*.py] → Cleaned, normalized datasets
       │
       ▼
[ml-pipeline/models/*.py] → Load pre-trained models (or train on-the-fly for demo)
       │
       ▼
[ml-pipeline/prediction/forecast_engine.py] → Generate 6/12/24/36hr forecasts
       │
       ▼
[ml-pipeline/output/predictions.json] + [flood_maps/*.geojson]
```

### 2. ML Pipeline → Spring Boot Backend
```
[predictions.json] 
       │
       ▼
[Scheduled Task: SchedulerConfig.java] (every 15 min)
       │
       ▼
[PredictionService.java] reads JSON → validates → saves to PostgreSQL
       │
       ▼
[AlertService.java] evaluates thresholds → creates Alert entities if needed
       │
       ▼
[REST Controllers] expose endpoints for frontend consumption
```

### 3. Backend → Frontend
```
[React Dashboard mounts]
       │
       ▼
[usePredictions.js] calls GET /api/predictions?region=assam
       │
       ▼
[PredictionController.java] returns JSON with rainfall forecasts + flood risk scores
       │
       ▼
[RainfallChart.jsx] renders time-series graph
[FloodMap.jsx] fetches GET /api/flood-map/assam → renders GeoJSON overlay
       │
       ▼
[AlertPanel.jsx] polls GET /api/alerts every 30s → displays active warnings
```

### 4. Alert Trigger Flow
```
[Prediction: >150mm rainfall in 24hr + flood risk >0.8]
       │
       ▼
[threshold_checker.py] flags as "RED ALERT"
       │
       ▼
[Backend: AlertService.java] creates Alert entity with level=RED
       │
       ▼
[NotificationService.java] (mock) logs SMS/email intent
       │
       ▼
[Frontend: AlertPanel.jsx] shows red banner + plays sound notification
```

---

## Technology Stack Summary

| Component | Technology | Justification |
|-----------|------------|---------------|
| **ML Pipeline** | Python 3.10+, scikit-learn, TensorFlow/Keras, pandas, xarray | Rich ecosystem for geospatial + time-series ML |
| **Backend** | Spring Boot 3.x, Java 17, PostgreSQL, Spring Data JPA | Rapid API development, robust scheduling, easy deployment |
| **Frontend** | React 18, Vite, Leaflet/Mapbox GL, Recharts, Tailwind CSS | Fast dev server, lightweight, excellent mapping libraries |
| **Data Storage** | PostgreSQL + PostGIS (for geo-queries) | Handles both structured predictions and spatial data |
| **Deployment** | Docker Compose (all 3 services) | Single-command setup for hackathon demo |
| **Mock Data** | Pre-generated JSON fixtures | Ensure demo works without live API dependencies |

---

## Hackathon Timeline (36 Hours)

### Phase 1: Foundation (Hours 0-8)
- Set up project structure (all 3 modules)
- Create mock data generators for satellite/radar/weather
- Implement basic data ingestion scripts (Python)
- Set up Spring Boot project with health-check endpoint
- Initialize React app with dummy dashboard

### Phase 2: Core ML (Hours 8-18)
- Build preprocessing pipeline
- Train simple baseline models (XGBoost for rainfall, logistic regression for flood risk)
- Create forecast engine that outputs predictions.json
- Generate static flood inundation GeoJSON files for demo regions

### Phase 3: Backend Integration (Hours 18-26)
- Implement REST controllers for predictions/alerts
- Connect backend to ML pipeline outputs
- Add scheduled polling task
- Seed database with sample regions (Assam, Kerala, Bihar flood-prone areas)

### Phase 4: Frontend Polish (Hours 26-32)
- Integrate real API calls into dashboard components
- Build interactive flood map with Leaflet
- Add alert notification UI (color-coded panels)
- Implement historical comparison charts

### Phase 5: Demo Prep (Hours 32-36)
- End-to-end testing with mock data scenarios
- Create presentation slides with architecture diagram
- Record demo video as backup
- Prepare Q&A talking points about scalability, real-world deployment

---

## Key Assumptions for Hackathon Scope

1. **Pre-trained Models**: Use simplified models trained on sample datasets (no real-time training during demo)
2. **Mock Live Data**: Simulate real-time satellite/radar feeds with pre-recorded data sequences
3. **Limited Geography**: Focus on 2-3 flood-prone districts (e.g., Kamrup in Assam, Patna in Bihar)
4. **Basic Auth**: Skip complex authentication; use API keys or no auth for demo
5. **Static Maps**: Use pre-generated flood inundation overlays rather than dynamic hydraulic modeling
6. **Notification Mock**: Log alerts to console/database instead of actual SMS/email gateways

---

## File Structure (Root Level)

```
sih26071-flood-warning-system/
├── ml-pipeline/                 # Python ML module
├── backend/                     # Spring Boot module
├── frontend/                    # React dashboard module
├── docker-compose.yml           # Orchestrate all 3 services + PostgreSQL
├── README.md                    # Project overview, setup instructions
├── ARCHITECTURE.md              # This document
├── demo-data/                   # Pre-generated mock datasets for offline demo
│   ├── satellite_samples.json
│   ├── radar_samples.json
│   └── weather_station_samples.json
└── docs/
    ├── api-spec.md              # OpenAPI/Swagger documentation
    ├── model-card.md            # ML model details, accuracy metrics
    └── deployment-guide.md      # Instructions for judges to run locally
```

---

## Next Steps

Once this architecture is approved, the implementation plan will be:
1. Scaffold all three projects with the file structures above
2. Create mock data generators to simulate live feeds
3. Implement the ML pipeline's data ingestion and preprocessing modules
4. Build baseline prediction models
5. Develop Spring Boot REST APIs to serve predictions
6. Construct the React dashboard with real-time data binding
7. Integrate end-to-end and prepare demo scenarios

This architecture balances hackathon feasibility with a demonstrable proof-of-concept that addresses the core problem statement: early warning through integrated multi-source data and ML-based prediction.
