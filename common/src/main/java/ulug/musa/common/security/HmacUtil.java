package ulug.musa.common.security;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.util.Base64;

/**
 * HMAC-SHA256 ile imza üretme ve doğrulama.
 */
public final class HmacUtil {

    private static final String ALGO = "HmacSHA256";

    private HmacUtil() {}

    /**
     * sharedSecret: POS ve Acquirer arasında paylaşılan gizli anahtar (string)
     * message: canonical string
     *
     * çıktı: Base64 URL-safe imza (padding yok)
     */
    public static String sign(String sharedSecret, String message) {
        try {
            Mac mac = Mac.getInstance(ALGO);
            SecretKeySpec keySpec = new SecretKeySpec(sharedSecret.getBytes(StandardCharsets.UTF_8), ALGO);
            mac.init(keySpec);

            byte[] raw = mac.doFinal(message.getBytes(StandardCharsets.UTF_8));
            return Base64.getUrlEncoder().withoutPadding().encodeToString(raw);
        } catch (Exception e) {
            throw new IllegalStateException("HMAC signing failed", e);
        }
    }

    /**
     * constant-time compare: timing attack riskini azaltır
     */
    public static boolean verify(String sharedSecret, String message, String expectedSignature) {
        String actual = sign(sharedSecret, message);
        return constantTimeEquals(actual, expectedSignature);
    }

    private static boolean constantTimeEquals(String a, String b) {
        if (a == null || b == null) return false;
        if (a.length() != b.length()) return false;

        int result = 0;
        for (int i = 0; i < a.length(); i++) {
            result |= a.charAt(i) ^ b.charAt(i);
        }
        return result == 0;
    }
}
