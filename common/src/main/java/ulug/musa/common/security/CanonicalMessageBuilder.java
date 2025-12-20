package ulug.musa.common.security;

import ulug.musa.common.model.PaymentRequest;

import java.math.BigDecimal;

/**
 * HMAC imzası üretmeden önce isteği canonical (standart) bir metne çeviririz.
 * POS ve Acquirer aynı fonksiyonu kullanırsa "imza uyuşmazlığı" problemleri azalır.
 */
public final class CanonicalMessageBuilder {

    private CanonicalMessageBuilder() {}

    /**
     * İmzalanacak metin (sıra değişmeyecek):
     * terminalId|traceId|txnType|amount|currency|panToken|timestamp|nonce|idempotencyKey|keyVersion
     */
    public static String buildForSignature(PaymentRequest r) {
        return safe(r.getTerminalId()) + "|" +
                safe(r.getTraceId()) + "|" +
                (r.getTxnType() == null ? "" : r.getTxnType().name()) + "|" +
                amountToString(r.getAmount()) + "|" +
                safe(r.getCurrency()) + "|" +
                safe(r.getPanToken()) + "|" +
                r.getTimestamp() + "|" +
                safe(r.getNonce()) + "|" +
                safe(r.getIdempotencyKey()) + "|" +
                r.getKeyVersion();
    }

    private static String safe(String s) {
        return s == null ? "" : s.trim();
    }

    /**
     * BigDecimal imzada kritik: 150.0 vs 150.00 farkı sorun çıkarır.
     * Burada normalize ediyoruz.
     */
    private static String amountToString(BigDecimal amount) {
        if (amount == null) return "";
        return amount.stripTrailingZeros().toPlainString();
    }
}
