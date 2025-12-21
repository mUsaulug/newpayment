package ulug.musa.acquirer.security.validation;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import ulug.musa.common.model.ErrorResponse;

import java.io.IOException;
import java.time.Instant;

@RestControllerAdvice
public class SecurityExceptionHandler {

    public static final String ERROR_CODE = "SEC_VALIDATION_FAILED";

    private final ObjectMapper objectMapper;

    public SecurityExceptionHandler(ObjectMapper objectMapper) {
        this.objectMapper = objectMapper;
    }

    @ExceptionHandler(SecurityValidationException.class)
    public ResponseEntity<ErrorResponse> handleSecurityValidation(SecurityValidationException exception) {
        ErrorResponse body = buildBody(exception.status().value(), ERROR_CODE, exception.getMessage());
        return ResponseEntity.status(exception.status())
                .contentType(MediaType.APPLICATION_JSON)
                .body(body);
    }

    public void write(HttpServletResponse response, int status, String code, String message) throws IOException {
        if (response.isCommitted()) return;

        response.resetBuffer();
        response.setStatus(status);
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.setCharacterEncoding("UTF-8");

        response.getWriter().write(objectMapper.writeValueAsString(buildBody(status, code, message)));
        response.flushBuffer();
    }

    private ErrorResponse buildBody(int status, String code, String message) {
        return new ErrorResponse(Instant.now().toString(), status, code, message);
    }
}
