package com.floodwarning.system.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.floodwarning.system.dto.FloodRiskResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientRequestException;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Service to communicate with the Python ML Prediction Service (FastAPI).
 */
@Service
public class FloodRiskService {

    private static final Logger logger = LoggerFactory.getLogger(FloodRiskService.class);

    private final WebClient webClient;
    private final String mlServiceUrl;

    public FloodRiskService(@Value("${ml.service.url:http://localhost:5000}") String mlServiceUrl) {
        this.mlServiceUrl = mlServiceUrl;
        this.webClient = WebClient.builder().build();
    }

    /**
     * Calls the Python ML service to get flood risk prediction for a station.
     * Includes basic error handling if ML service is unreachable.
     */
    public FloodRiskResponse getFloodRisk(String stationId) {
        logger.info("Fetching flood risk for station: {}", stationId);

        // Prepare request payload matching Python model's expected input
        Map<String, Object> requestPayload = new HashMap<>();
        requestPayload.put("station_id", stationId);
        // In a real scenario, these would come from recent sensor data
        // For demo, we send placeholder values that the ML service can use
        requestPayload.put("rainfall_mm", 0.0); 
        requestPayload.put("river_level_m", 0.0);

        try {
            JsonNode responseNode = webClient.post()
                    .uri(mlServiceUrl + "/predict")
                    .bodyValue(requestPayload)
                    .retrieve()
                    .bodyToMono(JsonNode.class)
                    .block();

            if (responseNode == null) {
                logger.error("Null response from ML service for station: {}", stationId);
                return createFallbackResponse(stationId, "Unknown");
            }

            String riskLevel = responseNode.path("risk_level").asText("Unknown");
            double predictedRainfall = responseNode.path("predicted_rainfall_mm").asDouble(0.0);
            
            // If predicted_rainfall_mm is not in response, derive from risk level for demo
            if (predictedRainfall == 0.0 && !riskLevel.equals("Unknown")) {
                predictedRainfall = estimateRainfallFromRisk(riskLevel);
            }

            FloodRiskResponse response = new FloodRiskResponse();
            response.setStationId(stationId);
            response.setRiskLevel(riskLevel);
            response.setPredictedRainfallMM(predictedRainfall);
            response.setTimestamp(LocalDateTime.now());

            logger.info("Successfully received risk level '{}' for station {}", riskLevel, stationId);
            return response;

        } catch (WebClientRequestException e) {
            logger.error("HTTP client error calling ML service for station {}: {}", 
                         stationId, e.getMessage());
            return createFallbackResponse(stationId, "Unknown");
        } catch (Exception e) {
            logger.error("Failed to connect to ML service for station {}. Error: {}", 
                         stationId, e.getMessage());
            return createFallbackResponse(stationId, "Service Unavailable");
        }
    }

    /**
     * Creates a fallback response when ML service is unavailable.
     */
    private FloodRiskResponse createFallbackResponse(String stationId, String riskLevel) {
        FloodRiskResponse response = new FloodRiskResponse();
        response.setStationId(stationId);
        response.setRiskLevel(riskLevel);
        response.setPredictedRainfallMM(0.0);
        response.setTimestamp(LocalDateTime.now());
        return response;
    }

    /**
     * Estimates rainfall based on risk level for demo purposes.
     */
    private double estimateRainfallFromRisk(String riskLevel) {
        switch (riskLevel.toLowerCase()) {
            case "low": return 10.0;
            case "medium": return 50.0;
            case "high": return 120.0;
            default: return 0.0;
        }
    }
}
