package ulug.musa.acquirer.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "security")
public record SecurityProperties(
        Hmac hmac,
        Replay replay
) {
    public record Hmac(String secret) {}
    public record Replay(Long allowedSkewSeconds) {}
}
