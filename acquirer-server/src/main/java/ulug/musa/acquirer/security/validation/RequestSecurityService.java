package ulug.musa.acquirer.security.validation;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import ulug.musa.acquirer.config.SecurityProperties;
import ulug.musa.acquirer.security.SecurityHeaders;
import ulug.musa.acquirer.security.hmac.HmacVerifier;
import ulug.musa.acquirer.security.replay.NonceStore;

import java.time.Instant;
import java.util.Map;

@Service
public class RequestSecurityService {

    private final HmacVerifier hmacVerifier;
    private final NonceStore nonceStore;
    private final long allowedSkewSeconds;

    public RequestSecurityService(SecurityProperties props, NonceStore nonceStore) {
        this.hmacVerifier = new HmacVerifier(props.hmac().secret());
        this.nonceStore = nonceStore;
        this.allowedSkewSeconds = props.replay().allowedSkewSeconds();
    }

    /**
     * İstek header'larını kontrol eder:
     * - terminalId var mı?
     * - nonce var mı ve daha önce kullanılmış mı?
     * - timestamp var mı ve zamanı makul mü?
     * - signature var mı ve HMAC doğru mu?
     *
     * signData formatı:
     * terminalId|nonce|timestamp|body
     */
    public void validate(Map<String, String> headers, String body) {
        String terminalId = mustGet(headers, SecurityHeaders.TERMINAL_ID);
        String nonce = mustGet(headers, SecurityHeaders.NONCE);
        String tsStr = mustGet(headers, SecurityHeaders.TIMESTAMP);
        String signature = mustGet(headers, SecurityHeaders.SIGNATURE);

        long timestamp = parseTimestamp(tsStr);
        validateSkew(timestamp);

        // Replay kontrol: nonce daha önce kullanıldı mı?
        boolean firstTime = nonceStore.storeIfAbsent(nonce, allowedSkewSeconds);
        if (!firstTime) {
            throw new SecurityValidationException(HttpStatus.CONFLICT, "Replay tespit edildi (nonce tekrar kullanıldı)");
        }

        String dataToSign = terminalId + "|" + nonce + "|" + timestamp + "|" + (body == null ? "" : body);

        if (!hmacVerifier.verify(dataToSign, signature)) {
            throw new SecurityValidationException(HttpStatus.UNAUTHORIZED, "HMAC imzası geçersiz");
        }
    }

    private String mustGet(Map<String, String> headers, String key) {
        String v = headers.get(key);
        if (v == null || v.isBlank()) {
            throw new SecurityValidationException(HttpStatus.BAD_REQUEST, "Eksik header: " + key);
        }
        return v;
    }

    private long parseTimestamp(String ts) {
        try {
            return Long.parseLong(ts);
        } catch (NumberFormatException e) {
            throw new SecurityValidationException(HttpStatus.BAD_REQUEST, "X-Timestamp sayısal olmalı (epoch seconds)");
        }
    }

    private void validateSkew(long requestTs) {
        long now = Instant.now().getEpochSecond();
        long diff = Math.abs(now - requestTs);
        if (diff > allowedSkewSeconds) {
            throw new SecurityValidationException(HttpStatus.UNAUTHORIZED, "Timestamp skew fazla: " + diff + "s");
        }
    }
}
