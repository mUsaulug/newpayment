package ulug.musa.posclient;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class PosClientApplication {

	public static void main(String[] args) {
		SpringApplication.run(PosClientApplication.class, args);
	}

}
