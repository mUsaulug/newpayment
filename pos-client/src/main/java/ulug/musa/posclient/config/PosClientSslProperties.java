package ulug.musa.posclient.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "pos-client.ssl")
public record PosClientSslProperties(String keyStore,
                                     String keyStorePassword,
                                     String trustStore,
                                     String trustStorePassword) {
}
