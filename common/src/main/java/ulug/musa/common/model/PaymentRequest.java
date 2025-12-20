package ulug.musa.common.model;

import java.math.BigDecimal;

public class PaymentRequest
{
    // Terminal kimliği (örn: TERM001)
    private String terminalId;

    // İsteği takip etmek için benzersiz ID (genelde UUID)
    private String traceId;

    // İşlem tipi (AUTH/CAPTURE/VOID/REVERSAL)
    private TxnType txnType;

    // Ödeme tutarı (örn: 150.00)
    private BigDecimal amount;

    // Para birimi (örn: TRY)
    private String currency;

    // Gerçek PAN yerine token (örn: tok_abc123)
    private String panToken;

    // Unix epoch saniye (doküman böyle anlatıyor)
    private long timestamp;

    // Tek kullanımlık rastgele değer
    private String nonce;

    // Aynı işlem retry edilirse aynı kalacak anahtar
    private String idempotencyKey;

    // HMAC anahtar versiyonu (demo: 1)
    private int keyVersion;

    // HMAC imzası (Base64 url-safe gibi bir formatla string)
    private String signature;

    // Jackson için boş constructor şart (POJO)
    public PaymentRequest() {}

    // İstersen hızlı oluşturmak için full constructor:


    public PaymentRequest
    (
            String terminalId,
            String traceId,
            TxnType txnType,
            BigDecimal amount,
            String currency,
            String panToken,
            long timestamp,
            String nonce,
            String idempotencyKey,
            int keyVersion,
            String signature
    )
    {
        this.terminalId = terminalId;
        this.traceId = traceId;
        this.txnType = txnType;
        this.amount = amount;
        this.currency = currency;
        this.panToken = panToken;
        this.timestamp = timestamp;
        this.nonce = nonce;
        this.idempotencyKey = idempotencyKey;
        this.keyVersion = keyVersion;
        this.signature = signature;
    }

    // --- Getter / Setter ---

    public String getTerminalId() { return terminalId; }
    public void setTerminalId(String terminalId) { this.terminalId = terminalId; }

    public String getTraceId() { return traceId; }
    public void setTraceId(String traceId) { this.traceId = traceId; }

    public TxnType getTxnType() { return txnType; }
    public void setTxnType(TxnType txnType) { this.txnType = txnType; }

    public BigDecimal getAmount() { return amount; }
    public void setAmount(BigDecimal amount) { this.amount = amount; }

    public String getCurrency() { return currency; }
    public void setCurrency(String currency) { this.currency = currency; }

    public String getPanToken() { return panToken; }
    public void setPanToken(String panToken) { this.panToken = panToken; }

    public long getTimestamp() { return timestamp; }
    public void setTimestamp(long timestamp) { this.timestamp = timestamp; }

    public String getNonce() { return nonce; }
    public void setNonce(String nonce) { this.nonce = nonce; }

    public String getIdempotencyKey() { return idempotencyKey; }
    public void setIdempotencyKey(String idempotencyKey) { this.idempotencyKey = idempotencyKey; }

    public int getKeyVersion() { return keyVersion; }
    public void setKeyVersion(int keyVersion) { this.keyVersion = keyVersion; }

    public String getSignature() { return signature; }
    public void setSignature(String signature) { this.signature = signature; }

}
