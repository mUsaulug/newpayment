package ulug.musa.acquirer.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import ulug.musa.acquirer.security.replay.InMemoryNonceStore;
import ulug.musa.acquirer.security.replay.NonceStore;
import ulug.musa.common.util.NonceGenerator;
import ulug.musa.common.util.TimeUtil;

import java.util.function.LongSupplier;
import java.util.function.Supplier;

@Configuration
public class SecurityBeansConfig {

    @Bean
    public Supplier<String> nonceGenerator() {
        return NonceGenerator::generate;
    }

    @Bean
    public LongSupplier epochSecondsSupplier() {
        return TimeUtil::nowEpochSeconds;
    }

    @Bean
    public NonceStore nonceStore(LongSupplier epochSecondsSupplier, SecurityProperties securityProperties) {
        return new InMemoryNonceStore(epochSecondsSupplier, securityProperties.replay().nonceCleanupIntervalSeconds());
    }
}
