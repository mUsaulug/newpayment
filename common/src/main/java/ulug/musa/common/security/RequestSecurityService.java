package ulug.musa.common.security;

import ulug.musa.common.model.PaymentRequest;
import ulug.musa.common.util.TimeUtil;

import java.nio.charset.Charset;
import java.util.Map;
import java.util.function.LongSupplier;
import java.util.regex.Pattern;

/**
 * İmzalama/validasyon akışını her iki tarafta da aynı şekilde uygulayabilmek için
 * ortak bir servis. Burada yapılan kontroller:
 * <ul>
 *     <li>Gerekli alanların/header'ların varlığı</li>
 *     <li>Timestamp skew kontrolü</li>
 *     <li>Nonce formatı ve (gerekirse) tekrar kullanım kontrolü</li>
 *     <li>HMAC imza doğrulaması</li>
 * </ul>
 */
public class RequestSecurityService {

    private static final Pattern NONCE_PATTERN = Pattern.compile("^[A-Za-z0-9_-]{22}$");

    private final String hmacSecret;
    private final Charset hmacCharset;
    private final long allowedSkewSeconds;
    private final LongSupplier epochSecondsSupplier;

    public RequestSecurityService(String hmacSecret, Charset hmacCharset, long allowedSkewSeconds) {
        this(hmacSecret, hmacCharset, allowedSkewSeconds, null);
    }

    public RequestSecurityService(String hmacSecret, Charset hmacCharset, long allowedSkewSeconds, LongSupplier epochSecondsSupplier) {
        this.hmacSecret = hmacSecret;
        this.hmacCharset = hmacCharset;
        this.allowedSkewSeconds = allowedSkewSeconds;
        this.epochSecondsSupplier = epochSecondsSupplier;
    }

    public void validateHeaders(Map<String, String> headers,
                                String terminalIdHeader,
                                String nonceHeader,
                                String timestampHeader,
                                String signatureHeader,
                                String body) {
        String terminalId = mustGet(headers, terminalIdHeader);
        String nonce = mustGet(headers, nonceHeader);
        String tsStr = mustGet(headers, timestampHeader);
        String signature = mustGet(headers, signatureHeader);

        long timestamp = parseTimestamp(tsStr, timestampHeader);
        validateSkew(timestamp);
        validateNonceFormat(nonce);
        validateReplay(nonce);

        String dataToSign = CanonicalMessageBuilder.buildForHeaders(terminalId, nonce, timestamp, body);
        ensureSignatureValid(dataToSign, signature);
    }

    public void validatePaymentRequest(PaymentRequest request) {
        if (request == null) {
            handleError(ValidationErrorType.BAD_REQUEST, "İstek body boş");
            return;
        }

        requireNonBlank(request.getTerminalId(), "terminalId");
        requireNonBlank(request.getTraceId(), "traceId");
        if (request.getTxnType() == null) {
            handleError(ValidationErrorType.BAD_REQUEST, "txnType boş olamaz");
        }
        if (request.getAmount() == null) {
            handleError(ValidationErrorType.BAD_REQUEST, "amount boş olamaz");
        }
        requireNonBlank(request.getCurrency(), "currency");
        requireNonBlank(request.getPanToken(), "panToken");
        requireNonBlank(request.getNonce(), "nonce");
        requireNonBlank(request.getSignature(), "signature");
        if (request.getTimestamp() <= 0) {
            handleError(ValidationErrorType.BAD_REQUEST, "timestamp 0'dan büyük olmalı");
        }
        if (request.getKeyVersion() <= 0) {
            handleError(ValidationErrorType.BAD_REQUEST, "keyVersion 0'dan büyük olmalı");
        }

        validateSkew(request.getTimestamp());
        validateNonceFormat(request.getNonce());
        validateReplay(request.getNonce());

        String dataToSign = CanonicalMessageBuilder.buildForSignature(request);
        ensureSignatureValid(dataToSign, request.getSignature());
    }

    protected void validateReplay(String nonce) {
        if (!isNonceFresh(nonce)) {
            handleError(ValidationErrorType.CONFLICT, "Replay tespit edildi (nonce tekrar kullanıldı)");
        }
    }

    protected boolean isNonceFresh(String nonce) {
        return true;
    }

    protected void handleError(ValidationErrorType type, String message) {
        throw new IllegalArgumentException(message);
    }

    protected long getAllowedSkewSeconds() {
        return allowedSkewSeconds;
    }

    private void ensureSignatureValid(String dataToSign, String signature) {
        if (!HmacUtil.verify(hmacSecret, dataToSign, signature, hmacCharset)) {
            handleError(ValidationErrorType.UNAUTHORIZED, "HMAC imzası geçersiz");
        }
    }

    private void validateSkew(long requestTs) {
        long now = nowEpochSeconds();
        long diff = Math.abs(now - requestTs);
        if (diff > allowedSkewSeconds) {
            handleError(ValidationErrorType.UNAUTHORIZED, "Timestamp skew fazla: " + diff + "s");
        }
    }

    private void validateNonceFormat(String nonce) {
        if (!NONCE_PATTERN.matcher(nonce).matches()) {
            handleError(ValidationErrorType.BAD_REQUEST, "Nonce formatı geçersiz (NonceGenerator ile uyumlu olmalı)");
        }
    }

    private long parseTimestamp(String ts, String headerName) {
        try {
            return Long.parseLong(ts);
        } catch (NumberFormatException e) {
            handleError(ValidationErrorType.BAD_REQUEST, headerName + " sayısal olmalı (epoch seconds)");
            return 0L;
        }
    }

    private long nowEpochSeconds() {
        return epochSecondsSupplier == null ? TimeUtil.nowEpochSeconds() : epochSecondsSupplier.getAsLong();
    }

    private String mustGet(Map<String, String> headers, String key) {
        String v = headers.get(key);
        if (v == null || v.isBlank()) {
            handleError(ValidationErrorType.BAD_REQUEST, "Eksik header: " + key);
        }
        return v;
    }

    private void requireNonBlank(String value, String fieldName) {
        if (value == null || value.isBlank()) {
            handleError(ValidationErrorType.BAD_REQUEST, fieldName + " boş olamaz");
        }
    }

    public enum ValidationErrorType {
        BAD_REQUEST,
        UNAUTHORIZED,
        CONFLICT
    }
}
