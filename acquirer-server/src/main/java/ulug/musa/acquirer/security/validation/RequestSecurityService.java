package ulug.musa.acquirer.security.validation;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import ulug.musa.acquirer.config.SecurityProperties;
import ulug.musa.acquirer.security.replay.NonceStore;
import ulug.musa.common.model.PaymentRequest;
import ulug.musa.common.security.RequestSecurityService.ValidationErrorType;

import java.util.Map;
import java.util.function.LongSupplier;

@Service
public class RequestSecurityService extends ulug.musa.common.security.RequestSecurityService {

    private final NonceStore nonceStore;
    private final long allowedSkewSeconds;
    private final SecurityProperties.Headers headers;

    public RequestSecurityService(SecurityProperties props, NonceStore nonceStore, LongSupplier epochSecondsSupplier) {
        super(props.hmac().secret(), props.hmac().charset(), props.replay().allowedSkewSeconds(), epochSecondsSupplier);
        this.nonceStore = nonceStore;
        this.allowedSkewSeconds = props.replay().allowedSkewSeconds();
        this.headers = props.headers();
    }

    public void validate(Map<String, String> headers, String body) {
        validateHeaders(headers,
                this.headers.terminalId(),
                this.headers.nonce(),
                this.headers.timestamp(),
                this.headers.signature(),
                body);
    }

    public void validate(PaymentRequest request) {
        validatePaymentRequest(request);
    }

    /**
     * Header doğrulaması filtrede yapılmışsa, nonce tekrar kullanım kontrolü bu adımda atlanmalıdır.
     */
    public void validateAfterHeaders(PaymentRequest request) {
        validatePaymentRequest(request, false);
    }

    @Override
    protected boolean isNonceFresh(String nonce) {
        return nonceStore.storeIfAbsent(nonce, allowedSkewSeconds);
    }

    @Override
    protected void handleError(ValidationErrorType type, String message) {
        throw new SecurityValidationException(map(type), message);
    }

    private HttpStatus map(ValidationErrorType type) {
        return switch (type) {
            case BAD_REQUEST -> HttpStatus.BAD_REQUEST;
            case UNAUTHORIZED -> HttpStatus.UNAUTHORIZED;
            case CONFLICT -> HttpStatus.CONFLICT;
        };
    }
}
