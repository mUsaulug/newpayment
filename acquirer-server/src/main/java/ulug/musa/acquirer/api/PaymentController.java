package ulug.musa.acquirer.api;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import ulug.musa.acquirer.fraud.FraudDecision;
import ulug.musa.acquirer.fraud.FraudDetectionService;
import ulug.musa.acquirer.security.validation.RequestSecurityService;
import ulug.musa.common.model.PaymentRequest;
import ulug.musa.common.model.PaymentResponse;
import ulug.musa.common.model.TxnType;

import java.math.BigDecimal;
import java.util.UUID;

/**
 * Payment Controller with integrated fraud detection.
 * 
 * Payment flow:
 * 1. Validate security headers (mTLS, HMAC)
 * 2. Run fraud detection (ML model + rules)
 * 3. Make decision (APPROVED/PENDING/DECLINED)
 * 4. Record transaction
 * 5. Return response
 */
@RestController
@RequestMapping(path = "/api/payments", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
public class PaymentController {

    private static final Logger log = LoggerFactory.getLogger(PaymentController.class);

    private final RequestSecurityService requestSecurityService;
    private final FraudDetectionService fraudDetectionService;

    public PaymentController(RequestSecurityService requestSecurityService,
            FraudDetectionService fraudDetectionService) {
        this.requestSecurityService = requestSecurityService;
        this.fraudDetectionService = fraudDetectionService;
    }

    @PostMapping
    public PaymentResponse pay(@RequestBody PaymentRequest request,
            @RequestHeader(value = "X-Merchant-Lat", required = false) BigDecimal merchantLat,
            @RequestHeader(value = "X-Merchant-Long", required = false) BigDecimal merchantLong,
            @RequestHeader(value = "X-Merchant-Category", required = false) String merchantCategory) {

        log.info("Payment request received: traceId={}, amount={}, panToken={}",
                request.getTraceId(), request.getAmount(), request.getPanToken());

        // 1. Security validation
        requestSecurityService.validateAfterHeaders(request);

        // 2. Fraud detection
        FraudDecision fraudDecision = fraudDetectionService.evaluate(
                request, merchantLat, merchantLong, merchantCategory);

        log.info("Fraud decision: traceId={}, score={}, decision={}",
                request.getTraceId(), fraudDecision.fraudScore(), fraudDecision.decision());

        // 3. Record transaction (async in production, sync for demo)
        fraudDetectionService.recordTransaction(
                request, fraudDecision, merchantLat, merchantLong, merchantCategory);

        // 4. Build response based on fraud decision
        return buildResponse(request, fraudDecision);
    }

    /**
     * Build payment response based on fraud decision.
     */
    private PaymentResponse buildResponse(PaymentRequest request, FraudDecision decision) {
        boolean approved = decision.isApproved();
        String responseCode = decision.getResponseCode();
        String authCode = approved ? generateAuthCode(request.getTxnType()) : null;
        String rrn = UUID.randomUUID().toString();
        String message = decision.getMessage();

        return new PaymentResponse(
                request.getTraceId(),
                approved,
                responseCode,
                authCode,
                rrn,
                message,
                decision.fraudScore(),
                decision.riskLevel(),
                decision.reasons());
    }

    /**
     * Generate 6-digit authorization code for approved transactions.
     */
    private String generateAuthCode(TxnType txnType) {
        if (txnType == TxnType.AUTH || txnType == TxnType.CAPTURE) {
            return String.format("%06d", Math.abs(UUID.randomUUID().hashCode()) % 1_000_000);
        }
        return null;
    }
}
