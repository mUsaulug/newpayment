package ulug.musa.posclient.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "pos-client")
public record PosClientProperties(Acquirer acquirer, int keyVersion) {

    public record Acquirer(String baseUrl) {}
}
