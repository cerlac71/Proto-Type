package com.floodwarning.system.controller;

import com.floodwarning.system.dto.FloodRiskResponse;
import com.floodwarning.system.service.FloodRiskService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * REST Controller for Flood Risk API endpoints.
 */
@RestController
@RequestMapping("/api/flood-risk")
@CrossOrigin(origins = "*") // Enable CORS for frontend access
public class FloodRiskController {

    private static final Logger logger = LoggerFactory.getLogger(FloodRiskController.class);

    private final FloodRiskService floodRiskService;

    public FloodRiskController(FloodRiskService floodRiskService) {
        this.floodRiskService = floodRiskService;
    }

    /**
     * GET /api/flood-risk/{stationId}
     * Returns JSON: {stationId, riskLevel, predictedRainfallMM, timestamp}
     * Includes basic error handling if ML service is unreachable.
     */
    @GetMapping("/{stationId}")
    public ResponseEntity<FloodRiskResponse> getFloodRisk(@PathVariable String stationId) {
        logger.info("Received request for flood risk of station: {}", stationId);
        
        try {
            FloodRiskResponse response = floodRiskService.getFloodRisk(stationId);
            
            if ("Service Unavailable".equals(response.getRiskLevel()) || 
                "Unknown".equals(response.getRiskLevel())) {
                logger.warn("Returning degraded response for station: {}", stationId);
                // Return 200 but with status indicating ML service issue
                return ResponseEntity.ok(response);
            }
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            logger.error("Unexpected error processing request for station {}: {}", 
                         stationId, e.getMessage());
            
            // Return a safe fallback response instead of 500 error
            FloodRiskResponse fallback = new FloodRiskResponse();
            fallback.setStationId(stationId);
            fallback.setRiskLevel("Unavailable");
            fallback.setPredictedRainfallMM(0.0);
            fallback.setTimestamp(java.time.LocalDateTime.now());
            
            return ResponseEntity.ok(fallback);
        }
    }
}
