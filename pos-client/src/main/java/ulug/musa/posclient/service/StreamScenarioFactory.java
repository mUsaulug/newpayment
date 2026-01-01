package ulug.musa.posclient.service;

import ulug.musa.common.model.PaymentResponse;
import ulug.musa.posclient.model.FrontendPaymentRequest;
import ulug.musa.posclient.model.StreamScenario;

import java.time.Instant;
import java.time.LocalDateTime;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class StreamScenarioFactory {
    private StreamScenarioFactory() {
    }

    private static final double HOME_LAT = 40.9912;
    private static final double HOME_LONG = 29.0228;
    private static final Map<String, Long> LAST_TX_BY_PAN = new ConcurrentHashMap<>();

    public static StreamScenario from(FrontendPaymentRequest request, PaymentResponse response) {
        StreamScenario scenario = new StreamScenario();
        String traceId = request.getTraceId();

        scenario.setId("live-" + traceId);
        scenario.setName("Live Payment " + traceId);
        scenario.setIsDemo(false);
        scenario.setRequest(buildRequest(request));
        scenario.setSecurityCheck(buildSecurityCheck());
        scenario.setFeatures(buildFeatures(request));
        scenario.setResponse(response);
        scenario.setPersisted(response.isApproved());
        scenario.setFallbackUsed(false);

        return scenario;
    }

    private static StreamScenario.ScenarioRequest buildRequest(FrontendPaymentRequest request) {
        StreamScenario.ScenarioRequest scenarioRequest = new StreamScenario.ScenarioRequest();
        scenarioRequest.setTerminalId(request.getTerminalId());
        scenarioRequest.setTraceId(request.getTraceId());
        scenarioRequest.setTxnType(request.getTxnType());
        scenarioRequest.setAmount(request.getAmount());
        scenarioRequest.setCurrency(request.getCurrency());
        scenarioRequest.setPanToken(request.getPanToken());
        scenarioRequest.setIdempotencyKey(request.getIdempotencyKey());
        scenarioRequest.setMerchantLat(request.getMerchantLat());
        scenarioRequest.setMerchantLong(request.getMerchantLong());
        scenarioRequest.setMerchantCategory(request.getMerchantCategory());
        return scenarioRequest;
    }

    private static StreamScenario.SecurityCheck buildSecurityCheck() {
        StreamScenario.SecurityCheck securityCheck = new StreamScenario.SecurityCheck();
        securityCheck.setMtls(true);
        securityCheck.setHeaderHmac(true);
        securityCheck.setNonce(true);
        securityCheck.setTimestamp(true);
        securityCheck.setBodySignature(true);
        return securityCheck;
    }

    private static StreamScenario.ScenarioFeatures buildFeatures(FrontendPaymentRequest request) {
        StreamScenario.ScenarioFeatures features = new StreamScenario.ScenarioFeatures();
        double amount = request.getAmount() == null ? 0 : request.getAmount().doubleValue();
        LocalDateTime now = LocalDateTime.now();
        int hour = now.getHour();
        features.setHour(hour);
        features.setIsNight(hour < 6 || hour >= 22 ? 1 : 0);
        features.setDistanceKm(distanceKm(request.getMerchantLat(), request.getMerchantLong()));
        features.setAmtZscore(roundToTwoDecimals(amount / 1000d));
        features.setCardAvgAmt(roundToTwoDecimals(amount * 0.3d + 50));
        features.setTimeSinceLastTx(timeSinceLastTx(request.getPanToken()));
        return features;
    }

    private static double roundToTwoDecimals(double value) {
        return Math.round(value * 100d) / 100d;
    }

    private static double distanceKm(Double merchantLat, Double merchantLong) {
        if (merchantLat == null || merchantLong == null) {
            return 0;
        }
        double dLat = Math.toRadians(merchantLat - HOME_LAT);
        double dLon = Math.toRadians(merchantLong - HOME_LONG);
        double a =
                Math.sin(dLat / 2) * Math.sin(dLat / 2) +
                Math.cos(Math.toRadians(HOME_LAT)) * Math.cos(Math.toRadians(merchantLat)) *
                        Math.sin(dLon / 2) * Math.sin(dLon / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return roundToTwoDecimals(6371d * c);
    }

    private static int timeSinceLastTx(String panToken) {
        if (panToken == null || panToken.isBlank()) {
            return 0;
        }
        long nowSeconds = Instant.now().getEpochSecond();
        Long lastSeconds = LAST_TX_BY_PAN.put(panToken, nowSeconds);
        if (lastSeconds == null) {
            return 0;
        }
        long diff = nowSeconds - lastSeconds;
        if (diff < 0) {
            return 0;
        }
        return diff > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) diff;
    }
}
