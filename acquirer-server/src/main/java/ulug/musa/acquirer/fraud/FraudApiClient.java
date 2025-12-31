package ulug.musa.acquirer.fraud;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

import java.time.Duration;
import java.util.Map;

/**
 * HTTP client for the Python fraud detection API.
 * Sends features to the XGBoost model and receives prediction.
 */
@Component
public class FraudApiClient {

    private static final Logger log = LoggerFactory.getLogger(FraudApiClient.class);

    private final RestClient restClient;
    private final boolean enabled;

    public FraudApiClient(
            @Value("${fraud.service.url:http://localhost:8000}") String fraudServiceUrl,
            @Value("${fraud.service.enabled:true}") boolean enabled,
            @Value("${fraud.service.timeout-ms:100}") int timeoutMs) {

        this.enabled = enabled;
        SimpleClientHttpRequestFactory requestFactory = new SimpleClientHttpRequestFactory();
        requestFactory.setConnectTimeout(timeoutMs);
        requestFactory.setReadTimeout(timeoutMs);
        this.restClient = RestClient.builder()
                .baseUrl(fraudServiceUrl)
                .defaultHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
                .requestFactory(requestFactory)
                .build();

        log.info("FraudApiClient initialized: url={}, enabled={}, timeout={}ms",
                fraudServiceUrl, enabled, timeoutMs);
    }

    /**
     * Send features to fraud API and get prediction.
     * Returns fallback prediction if API is disabled or fails.
     */
    public FraudPrediction predict(FraudFeatures features) {
        if (!enabled) {
            log.debug("Fraud API disabled, returning default prediction");
            return fallbackPrediction(features);
        }

        try {
            Map<String, Object> featureMap = features.toMap();

            FraudApiResponse response = restClient.post()
                    .uri("/predict")
                    .body(featureMap)
                    .retrieve()
                    .body(FraudApiResponse.class);

            if (response == null) {
                log.warn("Null response from fraud API, using fallback");
                return fallbackPrediction(features);
            }

            log.debug("Fraud API response: probability={}, prediction={}",
                    response.probability(), response.prediction());

            return new FraudPrediction(
                    response.probability(),
                    response.prediction(),
                    response.risk_level(),
                    response.threshold());

        } catch (RestClientException e) {
            log.error("Fraud API call failed: {}", e.getMessage());
            return fallbackPrediction(features);
        }
    }

    /**
     * Fallback prediction using simple rules when API is unavailable.
     * This ensures the payment system continues to work.
     */
    private FraudPrediction fallbackPrediction(FraudFeatures features) {
        // Simple rule-based fallback
        double score = 0.0;

        // Night transaction + high amount = suspicious
        if (features.isNight() == 1 && features.amt() > 1000) {
            score += 0.3;
        }

        // Very high amount deviation
        if (features.amtZscore() > 3) {
            score += 0.3;
        }

        // Long distance transaction
        if (features.distanceKm() > 500) {
            score += 0.2;
        }

        // First transaction for new user
        if (features.cardTxCount() == 0) {
            score += 0.1;
        }

        score = Math.min(score, 1.0);

        return new FraudPrediction(
                score,
                score >= 0.5 ? "FRAUD" : "LEGITIMATE",
                FraudPrediction.classifyRisk(score),
                0.5);
    }

    /**
     * DTO for fraud API response
     */
    record FraudApiResponse(
            double probability,
            String prediction,
            String risk_level,
            double threshold) {
    }
}
