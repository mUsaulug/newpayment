package ulug.musa.acquirer.api;

import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import ulug.musa.acquirer.security.validation.RequestSecurityService;
import ulug.musa.common.model.PaymentRequest;
import ulug.musa.common.model.PaymentResponse;
import ulug.musa.common.model.TxnType;

import java.util.UUID;

@RestController
@RequestMapping(path = "/api/payments", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
public class PaymentController {

    private final RequestSecurityService requestSecurityService;

    public PaymentController(RequestSecurityService requestSecurityService) {
        this.requestSecurityService = requestSecurityService;
    }

    @PostMapping
    public PaymentResponse pay(@RequestBody PaymentRequest request) {
        requestSecurityService.validateAfterHeaders(request);

        return new PaymentResponse(
                request.getTraceId(),
                true,
                resolveResponseCode(request.getTxnType()),
                generateAuthCode(request.getTxnType()),
                UUID.randomUUID().toString(),
                "APPROVED"
        );
    }

    private String resolveResponseCode(TxnType txnType) {
        return switch (txnType) {
            case AUTH, CAPTURE -> "00";
            case VOID, REVERSAL -> "00";
        };
    }

    private String generateAuthCode(TxnType txnType) {
        if (txnType == TxnType.AUTH || txnType == TxnType.CAPTURE) {
            return String.format("%06d", Math.abs(UUID.randomUUID().hashCode()) % 1_000_000);
        }
        return null;
    }
}
