# Spring Boot Flood Warning Backend

## Overview
Spring Boot REST API that integrates with the Python ML prediction service.

## Endpoint
- `GET /api/flood-risk/{stationId}` - Returns flood risk prediction

## Response Format
```json
{
  "stationId": "STN_001",
  "riskLevel": "High",
  "predictedRainfallMM": 120.0,
  "timestamp": "2026-09-08T22:30:00"
}
```

## Prerequisites
- Java 17+
- Maven (or use Maven wrapper)
- Python ML service running on `localhost:5000`

## Running Without Maven Installed

### Option 1: Install Maven
```bash
# Debian/Ubuntu
sudo apt-get update && sudo apt-get install -y maven

# Or download from https://maven.apache.org/download.cgi
```

### Option 2: Use Maven Wrapper (Recommended for Hackathon)
Generate wrapper in a project with Maven, then copy to this directory:
```bash
mvn -N io.takari:maven:wrapper
./mvnw spring-boot:run
```

### Option 3: Run Directly with Java
Compile manually or use an IDE like IntelliJ IDEA / Eclipse.

## Configuration
Edit `src/main/resources/application.properties`:
- `server.port` - Default: 8080
- `ml.service.url` - Default: http://localhost:5000

## Error Handling
If the Python ML service is unreachable, the API returns a fallback response:
```json
{
  "stationId": "STN_001",
  "riskLevel": "Service Unavailable",
  "predictedRainfallMM": 0.0,
  "timestamp": "2026-09-08T22:30:00"
}
```

## Testing
```bash
# Start Python ML service first
cd ../ml-pipeline && python flood_risk_model.py --api 5000

# Then start Spring Boot
cd backend && mvn spring-boot:run

# Test endpoint
curl http://localhost:8080/api/flood-risk/STN_001
```
