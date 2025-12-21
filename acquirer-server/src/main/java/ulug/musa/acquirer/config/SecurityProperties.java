package ulug.musa.acquirer.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import ulug.musa.common.security.SecurityHeaders;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

@ConfigurationProperties(prefix = "security")
public record SecurityProperties(
        Hmac hmac,
        Replay replay,
        Headers headers
) {
    public SecurityProperties {
        Objects.requireNonNull(hmac, "security.hmac ayarları zorunlu");
        if (replay == null) {
            replay = new Replay(120L, 60L);
        }
        if (headers == null) {
            headers = Headers.defaults();
        }
    }

    public record Hmac(String secret, Charset charset) {
        public Hmac {
            if (secret == null || secret.isBlank()) {
                throw new IllegalArgumentException("security.hmac.secret boş olamaz");
            }
            charset = charset == null ? StandardCharsets.UTF_8 : charset;
        }
    }

    public record Replay(Long allowedSkewSeconds, Long nonceCleanupIntervalSeconds) {
        public Replay {
            if (allowedSkewSeconds == null) {
                throw new IllegalArgumentException("security.replay.allowedSkewSeconds gerekli");
            }
            if (allowedSkewSeconds <= 0) {
                throw new IllegalArgumentException("security.replay.allowedSkewSeconds 0'dan büyük olmalı");
            }
            if (nonceCleanupIntervalSeconds == null || nonceCleanupIntervalSeconds <= 0) {
                nonceCleanupIntervalSeconds = 60L;
            }
        }
    }

    public record Headers(String terminalId, String nonce, String timestamp, String signature) {
        public Headers {
            terminalId = defaultIfBlank(terminalId, SecurityHeaders.TERMINAL_ID);
            nonce = defaultIfBlank(nonce, SecurityHeaders.NONCE);
            timestamp = defaultIfBlank(timestamp, SecurityHeaders.TIMESTAMP);
            signature = defaultIfBlank(signature, SecurityHeaders.SIGNATURE);
        }

        private static String defaultIfBlank(String value, String defaultValue) {
            return value == null || value.isBlank() ? defaultValue : value;
        }

        static Headers defaults() {
            return new Headers(
                    SecurityHeaders.TERMINAL_ID,
                    SecurityHeaders.NONCE,
                    SecurityHeaders.TIMESTAMP,
                    SecurityHeaders.SIGNATURE
            );
        }
    }
}
