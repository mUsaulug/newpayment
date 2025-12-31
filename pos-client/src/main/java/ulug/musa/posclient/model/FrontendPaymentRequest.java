package ulug.musa.posclient.model;

import ulug.musa.common.model.TxnType;

import java.math.BigDecimal;

public class FrontendPaymentRequest {

    private String terminalId;
    private String traceId;
    private TxnType txnType;
    private BigDecimal amount;
    private String currency;
    private String panToken;
    private String idempotencyKey;

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

    public TxnType getTxnType() {
        return txnType;
    }

    public void setTxnType(TxnType txnType) {
        this.txnType = txnType;
    }

    public BigDecimal getAmount() {
        return amount;
    }

    public void setAmount(BigDecimal amount) {
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
}
