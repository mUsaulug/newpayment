package ulug.musa.common.model;

public class PaymentResponse
{
    // İstekten gelen traceId aynen geri döner (eşleştirme için)
    private String traceId;

    // Onay mı? (true=approved, false=declined)
    private boolean approved;

    // Örn: "00" (onay) / "05" (genel ret)
    private String responseCode;

    // Onaylandıysa authCode (örn 6 haneli)
    private String authCode;

    // Retrieval Reference Number (benzersiz referans)
    private String rrn;

    // İnsan okunur mesaj: APPROVED / DECLINED
    private String message;

    public PaymentResponse() {}

    public PaymentResponse(
            String traceId,
            boolean approved,
            String responseCode,
            String authCode,
            String rrn,
            String message
    ) {
        this.traceId = traceId;
        this.approved = approved;
        this.responseCode = responseCode;
        this.authCode = authCode;
        this.rrn = rrn;
        this.message = message;
    }

    public String getTraceId() { return traceId; }
    public void setTraceId(String traceId) { this.traceId = traceId; }

    public boolean isApproved() { return approved; }
    public void setApproved(boolean approved) { this.approved = approved; }

    public String getResponseCode() { return responseCode; }
    public void setResponseCode(String responseCode) { this.responseCode = responseCode; }

    public String getAuthCode() { return authCode; }
    public void setAuthCode(String authCode) { this.authCode = authCode; }

    public String getRrn() { return rrn; }
    public void setRrn(String rrn) { this.rrn = rrn; }

    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }
}
