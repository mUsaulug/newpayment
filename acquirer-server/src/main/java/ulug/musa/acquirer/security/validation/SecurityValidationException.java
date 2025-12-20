package ulug.musa.acquirer.security.validation;

import org.springframework.http.HttpStatus;

public class SecurityValidationException extends RuntimeException {

    private final HttpStatus status;

    public SecurityValidationException(HttpStatus status, String message) {
        super(message);
        this.status = status;
    }

    public HttpStatus status() {
        return status;
    }
}
