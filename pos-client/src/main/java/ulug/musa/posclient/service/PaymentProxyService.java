package ulug.musa.posclient.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import ulug.musa.common.model.PaymentRequest;
import ulug.musa.common.model.PaymentResponse;
import ulug.musa.common.security.CanonicalMessageBuilder;
import ulug.musa.common.security.HmacUtil;
import ulug.musa.common.security.SecurityHeaders;
import ulug.musa.common.util.NonceGenerator;
import ulug.musa.common.util.TimeUtil;
import ulug.musa.posclient.config.PosClientProperties;
import ulug.musa.posclient.config.SecurityProperties;
import ulug.musa.posclient.model.FrontendPaymentRequest;

import java.nio.charset.Charset;

@Service
public class PaymentProxyService {

    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    private final PosClientProperties posClientProperties;
    private final SecurityProperties securityProperties;

    public PaymentProxyService(RestTemplateBuilder restTemplateBuilder,
                               ObjectMapper objectMapper,
                               PosClientProperties posClientProperties,
                               SecurityProperties securityProperties) {
        this.restTemplate = restTemplateBuilder.build();
        this.objectMapper = objectMapper;
        this.posClientProperties = posClientProperties;
        this.securityProperties = securityProperties;
    }

    public ResponseEntity<PaymentResponse> forwardPayment(FrontendPaymentRequest frontendRequest) {
        PaymentRequest request = buildPaymentRequest(frontendRequest);
        String bodyJson = toJson(request);

        HttpHeaders headers = buildSecurityHeaders(request, bodyJson);
        headers.setContentType(MediaType.APPLICATION_JSON);

        String url = posClientProperties.acquirer().baseUrl() + "/api/payments";
        HttpEntity<String> entity = new HttpEntity<>(bodyJson, headers);
        ResponseEntity<PaymentResponse> response = restTemplate.exchange(url, HttpMethod.POST, entity, PaymentResponse.class);
        return ResponseEntity.status(response.getStatusCode()).body(response.getBody());
    }

    private PaymentRequest buildPaymentRequest(FrontendPaymentRequest frontendRequest) {
        PaymentRequest request = new PaymentRequest();
        request.setTerminalId(frontendRequest.getTerminalId());
        request.setTraceId(frontendRequest.getTraceId());
        request.setTxnType(frontendRequest.getTxnType());
        request.setAmount(frontendRequest.getAmount());
        request.setCurrency(frontendRequest.getCurrency());
        request.setPanToken(frontendRequest.getPanToken());

        long timestamp = TimeUtil.nowEpochSeconds();
        String nonce = NonceGenerator.generate();

        request.setTimestamp(timestamp);
        request.setNonce(nonce);
        request.setIdempotencyKey(frontendRequest.getIdempotencyKey());
        request.setKeyVersion(posClientProperties.keyVersion());

        String bodySignature = HmacUtil.sign(securityProperties.hmac().secret(),
                CanonicalMessageBuilder.buildForSignature(request),
                effectiveCharset());
        request.setSignature(bodySignature);

        return request;
    }

    private HttpHeaders buildSecurityHeaders(PaymentRequest request, String bodyJson) {
        HttpHeaders headers = new HttpHeaders();
        headers.set(SecurityHeaders.TERMINAL_ID, request.getTerminalId());
        headers.set(SecurityHeaders.NONCE, request.getNonce());
        headers.set(SecurityHeaders.TIMESTAMP, String.valueOf(request.getTimestamp()));

        String headerSignature = HmacUtil.sign(securityProperties.hmac().secret(),
                CanonicalMessageBuilder.buildForHeaders(request.getTerminalId(), request.getNonce(), request.getTimestamp(), bodyJson),
                effectiveCharset());
        headers.set(SecurityHeaders.SIGNATURE, headerSignature);

        return headers;
    }

    private Charset effectiveCharset() {
        Charset charset = securityProperties.hmac().charset();
        return charset == null ? Charset.defaultCharset() : charset;
    }

    private String toJson(PaymentRequest request) {
        try {
            return objectMapper.writeValueAsString(request);
        } catch (JsonProcessingException e) {
            throw new IllegalStateException("Payment request serialization failed", e);
        }
    }
}
