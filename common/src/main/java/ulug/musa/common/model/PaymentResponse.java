package ulug.musa.common.model;

import java.util.List;

public class PaymentResponse {
    // İstekten gelen traceId aynen geri döner (eşleştirme için)
    private String traceId;

    // Onay mı? (true=approved, false=declined)
    private boolean approved;

    // Örn: "00" (onay) / "05" (genel ret)
    private String responseCode;

    // Onaylandıysa authCode (örn 6 haneli)
    private String authCode;

    // Retrieval Reference Number (benzersiz referans)
    private String rrn;

    // İnsan okunur mesaj: APPROVED / DECLINED
    private String message;

    // === FRAUD DETECTION FIELDS ===

    // Fraud skoru (0.0 - 1.0)
    private Double fraudScore;

    // Risk seviyesi: MINIMAL, LOW, MEDIUM, HIGH, CRITICAL
    private String riskLevel;

    // Fraud nedenleri listesi (Türkçe açıklamalar)
    private List<String> fraudReasons;

    public PaymentResponse() {
    }

    public PaymentResponse(
            String traceId,
            boolean approved,
            String responseCode,
            String authCode,
            String rrn,
            String message) {
        this.traceId = traceId;
        this.approved = approved;
        this.responseCode = responseCode;
        this.authCode = authCode;
        this.rrn = rrn;
        this.message = message;
    }

    // Extended constructor with fraud fields
    public PaymentResponse(
            String traceId,
            boolean approved,
            String responseCode,
            String authCode,
            String rrn,
            String message,
            Double fraudScore,
            String riskLevel,
            List<String> fraudReasons) {
        this(traceId, approved, responseCode, authCode, rrn, message);
        this.fraudScore = fraudScore;
        this.riskLevel = riskLevel;
        this.fraudReasons = fraudReasons;
    }

    public String getTraceId() {
        return traceId;
    }

    public void setTraceId(String traceId) {
        this.traceId = traceId;
    }

    public boolean isApproved() {
        return approved;
    }

    public void setApproved(boolean approved) {
        this.approved = approved;
    }

    public String getResponseCode() {
        return responseCode;
    }

    public void setResponseCode(String responseCode) {
        this.responseCode = responseCode;
    }

    public String getAuthCode() {
        return authCode;
    }

    public void setAuthCode(String authCode) {
        this.authCode = authCode;
    }

    public String getRrn() {
        return rrn;
    }

    public void setRrn(String rrn) {
        this.rrn = rrn;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public Double getFraudScore() {
        return fraudScore;
    }

    public void setFraudScore(Double fraudScore) {
        this.fraudScore = fraudScore;
    }

    public String getRiskLevel() {
        return riskLevel;
    }

    public void setRiskLevel(String riskLevel) {
        this.riskLevel = riskLevel;
    }

    public List<String> getFraudReasons() {
        return fraudReasons;
    }

    public void setFraudReasons(List<String> fraudReasons) {
        this.fraudReasons = fraudReasons;
    }
}
