package ulug.musa.posclient.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.nio.charset.Charset;

@ConfigurationProperties(prefix = "security")
public record SecurityProperties(Hmac hmac) {

    public record Hmac(String secret, Charset charset) {}
}
