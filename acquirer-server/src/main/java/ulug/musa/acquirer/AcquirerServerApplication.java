package ulug.musa.acquirer;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;
import org.springframework.beans.factory.SmartInitializingSingleton;
import org.springframework.context.annotation.Bean;
import ulug.musa.acquirer.config.SecurityProperties;

import java.nio.charset.Charset;
import java.util.Objects;

@SpringBootApplication
@ConfigurationPropertiesScan
public class AcquirerServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(AcquirerServerApplication.class, args);
    }

    @Bean
    public SmartInitializingSingleton securityPropertiesInitializer(SecurityProperties securityProperties) {
        return () -> validateSecurityProperties(securityProperties);
    }

    private static void validateSecurityProperties(SecurityProperties properties) {
        Objects.requireNonNull(properties, "SecurityProperties yüklenemedi");

        SecurityProperties.Hmac hmac = properties.hmac();
        if (hmac == null) {
            throw new IllegalStateException("security.hmac ayarları yüklenemedi");
        }
        if (hmac.secret() == null || hmac.secret().isBlank()) {
            throw new IllegalStateException("security.hmac.secret boş olamaz");
        }
        Charset charset = hmac.charset();
        if (charset == null) {
            throw new IllegalStateException("security.hmac.charset belirlenemedi");
        }

        SecurityProperties.Replay replay = properties.replay();
        if (replay == null) {
            throw new IllegalStateException("security.replay ayarları yüklenemedi");
        }
        if (replay.allowedSkewSeconds() == null || replay.allowedSkewSeconds() <= 0) {
            throw new IllegalStateException("security.replay.allowedSkewSeconds 0'dan büyük olmalı");
        }
        if (replay.nonceCleanupIntervalSeconds() == null || replay.nonceCleanupIntervalSeconds() <= 0) {
            throw new IllegalStateException("security.replay.nonceCleanupIntervalSeconds 0'dan büyük olmalı");
        }
    }
}
