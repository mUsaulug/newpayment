package ulug.musa.acquirer.config;

import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.stereotype.Component;
import ulug.musa.acquirer.security.replay.NonceStore;

import java.nio.charset.Charset;
import java.util.Objects;

@Component
public class SecurityHealthIndicator implements HealthIndicator {

    private final SecurityProperties properties;
    private final NonceStore nonceStore;

    public SecurityHealthIndicator(SecurityProperties properties, NonceStore nonceStore) {
        this.properties = Objects.requireNonNull(properties, "SecurityProperties eksik");
        this.nonceStore = Objects.requireNonNull(nonceStore, "NonceStore eksik");
    }

    @Override
    public Health health() {
        boolean secretMissing = isBlank(properties.hmac().secret());
        boolean skewInvalid = properties.replay().allowedSkewSeconds() == null
                || properties.replay().allowedSkewSeconds() <= 0;

        Health.Builder builder = secretMissing || skewInvalid ? Health.down() : Health.up();
        builder.withDetail("hmacCharset", charsetName(properties.hmac().charset()));
        builder.withDetail("allowedSkewSeconds", properties.replay().allowedSkewSeconds());
        builder.withDetail("nonceCleanupIntervalSeconds", properties.replay().nonceCleanupIntervalSeconds());
        builder.withDetail("nonceStore", nonceStore.getClass().getName());
        return builder.build();
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    private static String charsetName(Charset charset) {
        return charset == null ? "unknown" : charset.name();
    }
}
