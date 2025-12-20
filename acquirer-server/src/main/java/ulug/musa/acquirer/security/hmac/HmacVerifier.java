package ulug.musa.acquirer.security.hmac;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Base64;

public class HmacVerifier {

    private final byte[] secretBytes;

    public HmacVerifier(String secret) {
        if (secret == null || secret.isBlank()) {
            throw new IllegalArgumentException("HMAC secret boş olamaz");
        }
        this.secretBytes = secret.getBytes(StandardCharsets.UTF_8);
    }

    public String sign(String data) {
        try {
            Mac mac = Mac.getInstance("HmacSHA256");
            mac.init(new SecretKeySpec(secretBytes, "HmacSHA256"));
            byte[] raw = mac.doFinal(data.getBytes(StandardCharsets.UTF_8));
            return Base64.getUrlEncoder().withoutPadding().encodeToString(raw);
        } catch (Exception e) {
            throw new IllegalStateException("HMAC hesaplanamadı", e);
        }
    }

    public boolean verify(String data, String base64Signature) {
        if (base64Signature == null || base64Signature.isBlank()) return false;

        String expected = sign(data);

        // timing-attack'e daha dayanıklı karşılaştırma
        return MessageDigest.isEqual(
                expected.getBytes(StandardCharsets.UTF_8),
                base64Signature.getBytes(StandardCharsets.UTF_8)
        );
    }
}
