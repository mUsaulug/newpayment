package ulug.musa.acquirer.domain;

import jakarta.persistence.*;
import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * Transaction history entity for fraud detection.
 * Stores past transactions for rolling statistics and audit trail.
 */
@Entity
@Table(name = "transaction_history")
public class TransactionHistory {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "pan_token", nullable = false, length = 64)
    private String panToken;

    @Column(name = "trace_id", nullable = false, length = 64)
    private String traceId;

    @Column(name = "idempotency_key", length = 64)
    private String idempotencyKey;

    @Column(nullable = false, precision = 12, scale = 2)
    private BigDecimal amount;

    @Column(name = "terminal_id", length = 32)
    private String terminalId;

    @Column(name = "merchant_lat", precision = 10, scale = 6)
    private BigDecimal merchantLat;

    @Column(name = "merchant_long", precision = 10, scale = 6)
    private BigDecimal merchantLong;

    @Column(name = "merchant_category", length = 50)
    private String merchantCategory;

    @Column(name = "fraud_score", precision = 5, scale = 4)
    private BigDecimal fraudScore;

    @Column(name = "risk_level", length = 20)
    private String riskLevel;

    @Column(nullable = false, length = 20)
    private String decision;

    @Column(name = "fraud_reasons", columnDefinition = "TEXT")
    private String fraudReasons;

    @Column(name = "transaction_time", nullable = false)
    private LocalDateTime transactionTime;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    public TransactionHistory() {
    }

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
    }

    // Static factory method for creating from payment request
    public static TransactionHistory from(String panToken, String traceId, BigDecimal amount,
            String terminalId, BigDecimal fraudScore,
            String riskLevel, String decision, String reasons,
            LocalDateTime transactionTime, String idempotencyKey) {
        TransactionHistory history = new TransactionHistory();
        history.panToken = panToken;
        history.traceId = traceId;
        history.idempotencyKey = idempotencyKey;
        history.amount = amount;
        history.terminalId = terminalId;
        history.fraudScore = fraudScore;
        history.riskLevel = riskLevel;
        history.decision = decision;
        history.fraudReasons = reasons;
        history.transactionTime = transactionTime;
        return history;
    }

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getPanToken() {
        return panToken;
    }

    public void setPanToken(String panToken) {
        this.panToken = panToken;
    }

    public String getTraceId() {
        return traceId;
    }

    public void setTraceId(String traceId) {
        this.traceId = traceId;
    }

    public String getIdempotencyKey() {
        return idempotencyKey;
    }

    public void setIdempotencyKey(String idempotencyKey) {
        this.idempotencyKey = idempotencyKey;
    }

    public BigDecimal getAmount() {
        return amount;
    }

    public void setAmount(BigDecimal amount) {
        this.amount = amount;
    }

    public String getTerminalId() {
        return terminalId;
    }

    public void setTerminalId(String terminalId) {
        this.terminalId = terminalId;
    }

    public BigDecimal getMerchantLat() {
        return merchantLat;
    }

    public void setMerchantLat(BigDecimal merchantLat) {
        this.merchantLat = merchantLat;
    }

    public BigDecimal getMerchantLong() {
        return merchantLong;
    }

    public void setMerchantLong(BigDecimal merchantLong) {
        this.merchantLong = merchantLong;
    }

    public String getMerchantCategory() {
        return merchantCategory;
    }

    public void setMerchantCategory(String merchantCategory) {
        this.merchantCategory = merchantCategory;
    }

    public BigDecimal getFraudScore() {
        return fraudScore;
    }

    public void setFraudScore(BigDecimal fraudScore) {
        this.fraudScore = fraudScore;
    }

    public String getRiskLevel() {
        return riskLevel;
    }

    public void setRiskLevel(String riskLevel) {
        this.riskLevel = riskLevel;
    }

    public String getDecision() {
        return decision;
    }

    public void setDecision(String decision) {
        this.decision = decision;
    }

    public String getFraudReasons() {
        return fraudReasons;
    }

    public void setFraudReasons(String fraudReasons) {
        this.fraudReasons = fraudReasons;
    }

    public LocalDateTime getTransactionTime() {
        return transactionTime;
    }

    public void setTransactionTime(LocalDateTime transactionTime) {
        this.transactionTime = transactionTime;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }
}
