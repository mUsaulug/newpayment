package ulug.musa.acquirer;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class AcquirerServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(AcquirerServerApplication.class, args);
    }
}
