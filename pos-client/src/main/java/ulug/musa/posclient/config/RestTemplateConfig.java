package ulug.musa.posclient.config;

import org.apache.hc.client5.http.impl.classic.CloseableHttpClient;
import org.apache.hc.client5.http.impl.classic.HttpClients;
import org.apache.hc.core5.ssl.SSLContextBuilder;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.http.client.HttpComponentsClientHttpRequestFactory;
import org.springframework.web.client.RestTemplate;

import javax.net.ssl.SSLContext;
import java.io.InputStream;
import java.security.KeyStore;

@Configuration
public class RestTemplateConfig {

    private static final String KEYSTORE_TYPE = "PKCS12";

    @Bean
    public RestTemplate restTemplate(RestTemplateBuilder restTemplateBuilder,
                                     PosClientSslProperties sslProperties,
                                     ResourceLoader resourceLoader) {
        SSLContext sslContext = buildSslContext(sslProperties, resourceLoader);
        CloseableHttpClient httpClient = HttpClients.custom()
                .setSSLContext(sslContext)
                .build();
        HttpComponentsClientHttpRequestFactory requestFactory =
                new HttpComponentsClientHttpRequestFactory(httpClient);
        return restTemplateBuilder.requestFactory(() -> requestFactory).build();
    }

    private SSLContext buildSslContext(PosClientSslProperties sslProperties, ResourceLoader resourceLoader) {
        try {
            KeyStore keyStore = loadStore(resourceLoader, sslProperties.keyStore(), sslProperties.keyStorePassword());
            KeyStore trustStore = loadStore(resourceLoader, sslProperties.trustStore(), sslProperties.trustStorePassword());

            return SSLContextBuilder.create()
                    .loadKeyMaterial(keyStore, sslProperties.keyStorePassword().toCharArray())
                    .loadTrustMaterial(trustStore, null)
                    .build();
        } catch (Exception e) {
            throw new IllegalStateException("Failed to initialize mTLS SSL context", e);
        }
    }

    private KeyStore loadStore(ResourceLoader resourceLoader, String location, String password) throws Exception {
        Resource resource = resourceLoader.getResource(location);
        if (!resource.exists()) {
            throw new IllegalArgumentException("Keystore resource not found: " + location);
        }
        try (InputStream inputStream = resource.getInputStream()) {
            KeyStore keyStore = KeyStore.getInstance(KEYSTORE_TYPE);
            keyStore.load(inputStream, password.toCharArray());
            return keyStore;
        }
    }
}
