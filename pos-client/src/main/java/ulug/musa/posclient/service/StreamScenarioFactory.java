package ulug.musa.posclient.service;

import ulug.musa.common.model.PaymentResponse;
import ulug.musa.posclient.model.FrontendPaymentRequest;
import ulug.musa.posclient.model.StreamScenario;

import java.time.LocalDateTime;

public final class StreamScenarioFactory {
    private StreamScenarioFactory() {
    }

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
        LocalDateTime now = LocalDateTime.now();
        int hour = now.getHour();
        features.setHour(hour);
        features.setIsNight(hour < 6 || hour >= 22 ? 1 : 0);
        features.setDistanceKm(0);
        features.setAmtZscore(roundToTwoDecimals(request.getAmount() / 1000d));
        features.setCardAvgAmt(roundToTwoDecimals(request.getAmount() * 0.3d + 50));
        features.setTimeSinceLastTx(0);
        return features;
    }

    private static double roundToTwoDecimals(double value) {
        return Math.round(value * 100d) / 100d;
    }
}
