package com.floodwarning.system.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.LocalDateTime;

/**
 * DTO for Flood Risk Response returned to the frontend.
 * Matches the requirement: {stationId, riskLevel, predictedRainfallMM, timestamp}
 */
public class FloodRiskResponse {

    private String stationId;
    private String riskLevel;
    private Double predictedRainfallMM;
    private LocalDateTime timestamp;

    public FloodRiskResponse() {}

    public FloodRiskResponse(String stationId, String riskLevel, Double predictedRainfallMM, LocalDateTime timestamp) {
        this.stationId = stationId;
        this.riskLevel = riskLevel;
        this.predictedRainfallMM = predictedRainfallMM;
        this.timestamp = timestamp;
    }

    // Getters and Setters
    public String getStationId() { return stationId; }
    public void setStationId(String stationId) { this.stationId = stationId; }

    public String getRiskLevel() { return riskLevel; }
    public void setRiskLevel(String riskLevel) { this.riskLevel = riskLevel; }

    @JsonProperty("predictedRainfallMM")
    public Double getPredictedRainfallMM() { return predictedRainfallMM; }
    @JsonProperty("predictedRainfallMM")
    public void setPredictedRainfallMM(Double predictedRainfallMM) { this.predictedRainfallMM = predictedRainfallMM; }

    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
}
