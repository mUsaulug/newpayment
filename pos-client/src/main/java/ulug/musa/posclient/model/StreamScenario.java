package ulug.musa.posclient.model;

import ulug.musa.common.model.PaymentResponse;

public class StreamScenario {
    private String id;
    private String name;
    private ScenarioRequest request;
    private DemoPlaceholders demoPlaceholders;
    private Boolean isDemo;
    private SecurityCheck securityCheck;
    private ScenarioFeatures features;
    private PaymentResponse response;
    private boolean persisted;
    private boolean fallbackUsed;

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public ScenarioRequest getRequest() {
        return request;
    }

    public void setRequest(ScenarioRequest request) {
        this.request = request;
    }

    public DemoPlaceholders getDemoPlaceholders() {
        return demoPlaceholders;
    }

    public void setDemoPlaceholders(DemoPlaceholders demoPlaceholders) {
        this.demoPlaceholders = demoPlaceholders;
    }

    public Boolean getIsDemo() {
        return isDemo;
    }

    public void setIsDemo(Boolean demo) {
        isDemo = demo;
    }

    public SecurityCheck getSecurityCheck() {
        return securityCheck;
    }

    public void setSecurityCheck(SecurityCheck securityCheck) {
        this.securityCheck = securityCheck;
    }

    public ScenarioFeatures getFeatures() {
        return features;
    }

    public void setFeatures(ScenarioFeatures features) {
        this.features = features;
    }

    public PaymentResponse getResponse() {
        return response;
    }

    public void setResponse(PaymentResponse response) {
        this.response = response;
    }

    public boolean isPersisted() {
        return persisted;
    }

    public void setPersisted(boolean persisted) {
        this.persisted = persisted;
    }

    public boolean isFallbackUsed() {
        return fallbackUsed;
    }

    public void setFallbackUsed(boolean fallbackUsed) {
        this.fallbackUsed = fallbackUsed;
    }

    public static class ScenarioRequest {
        private String terminalId;
        private String traceId;
        private String txnType;
        private double amount;
        private String currency;
        private String panToken;
        private String idempotencyKey;
        private Double merchantLat;
        private Double merchantLong;
        private String merchantCategory;

        public String getTerminalId() {
            return terminalId;
        }

        public void setTerminalId(String terminalId) {
            this.terminalId = terminalId;
        }

        public String getTraceId() {
            return traceId;
        }

        public void setTraceId(String traceId) {
            this.traceId = traceId;
        }

        public String getTxnType() {
            return txnType;
        }

        public void setTxnType(String txnType) {
            this.txnType = txnType;
        }

        public double getAmount() {
            return amount;
        }

        public void setAmount(double amount) {
            this.amount = amount;
        }

        public String getCurrency() {
            return currency;
        }

        public void setCurrency(String currency) {
            this.currency = currency;
        }

        public String getPanToken() {
            return panToken;
        }

        public void setPanToken(String panToken) {
            this.panToken = panToken;
        }

        public String getIdempotencyKey() {
            return idempotencyKey;
        }

        public void setIdempotencyKey(String idempotencyKey) {
            this.idempotencyKey = idempotencyKey;
        }

        public Double getMerchantLat() {
            return merchantLat;
        }

        public void setMerchantLat(Double merchantLat) {
            this.merchantLat = merchantLat;
        }

        public Double getMerchantLong() {
            return merchantLong;
        }

        public void setMerchantLong(Double merchantLong) {
            this.merchantLong = merchantLong;
        }

        public String getMerchantCategory() {
            return merchantCategory;
        }

        public void setMerchantCategory(String merchantCategory) {
            this.merchantCategory = merchantCategory;
        }
    }

    public static class DemoPlaceholders {
        private String nonce;
        private String signature;

        public String getNonce() {
            return nonce;
        }

        public void setNonce(String nonce) {
            this.nonce = nonce;
        }

        public String getSignature() {
            return signature;
        }

        public void setSignature(String signature) {
            this.signature = signature;
        }
    }

    public static class SecurityCheck {
        private boolean mtls;
        private boolean headerHmac;
        private boolean nonce;
        private boolean timestamp;
        private boolean bodySignature;

        public boolean isMtls() {
            return mtls;
        }

        public void setMtls(boolean mtls) {
            this.mtls = mtls;
        }

        public boolean isHeaderHmac() {
            return headerHmac;
        }

        public void setHeaderHmac(boolean headerHmac) {
            this.headerHmac = headerHmac;
        }

        public boolean isNonce() {
            return nonce;
        }

        public void setNonce(boolean nonce) {
            this.nonce = nonce;
        }

        public boolean isTimestamp() {
            return timestamp;
        }

        public void setTimestamp(boolean timestamp) {
            this.timestamp = timestamp;
        }

        public boolean isBodySignature() {
            return bodySignature;
        }

        public void setBodySignature(boolean bodySignature) {
            this.bodySignature = bodySignature;
        }
    }

    public static class ScenarioFeatures {
        private int hour;
        private int isNight;
        private double distanceKm;
        private double amtZscore;
        private double cardAvgAmt;
        private int timeSinceLastTx;

        public int getHour() {
            return hour;
        }

        public void setHour(int hour) {
            this.hour = hour;
        }

        public int getIsNight() {
            return isNight;
        }

        public void setIsNight(int isNight) {
            this.isNight = isNight;
        }

        public double getDistanceKm() {
            return distanceKm;
        }

        public void setDistanceKm(double distanceKm) {
            this.distanceKm = distanceKm;
        }

        public double getAmtZscore() {
            return amtZscore;
        }

        public void setAmtZscore(double amtZscore) {
            this.amtZscore = amtZscore;
        }

        public double getCardAvgAmt() {
            return cardAvgAmt;
        }

        public void setCardAvgAmt(double cardAvgAmt) {
            this.cardAvgAmt = cardAvgAmt;
        }

        public int getTimeSinceLastTx() {
            return timeSinceLastTx;
        }

        public void setTimeSinceLastTx(int timeSinceLastTx) {
            this.timeSinceLastTx = timeSinceLastTx;
        }
    }
}
