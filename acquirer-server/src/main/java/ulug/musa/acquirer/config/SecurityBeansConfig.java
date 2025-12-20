package ulug.musa.acquirer.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import ulug.musa.acquirer.security.replay.InMemoryNonceStore;
import ulug.musa.acquirer.security.replay.NonceStore;

@Configuration
public class SecurityBeansConfig {

    @Bean
    public NonceStore nonceStore() {
        return new InMemoryNonceStore();
    }
}
