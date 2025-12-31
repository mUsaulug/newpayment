package ulug.musa.posclient.api;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import ulug.musa.common.model.PaymentResponse;
import ulug.musa.posclient.model.FrontendPaymentRequest;
import ulug.musa.posclient.service.PaymentProxyService;

@RestController
@RequestMapping("/api/pos/payments")
public class PaymentProxyController {

    private final PaymentProxyService paymentProxyService;

    public PaymentProxyController(PaymentProxyService paymentProxyService) {
        this.paymentProxyService = paymentProxyService;
    }

    @PostMapping
    public ResponseEntity<PaymentResponse> pay(@RequestBody FrontendPaymentRequest request) {
        return paymentProxyService.forwardPayment(request);
    }
}
