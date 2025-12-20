package ulug.musa.common.util;

import java.security.SecureRandom;
import java.util.Base64;

/**
 * Nonce: Tek kullanımlık rastgele değer.
 * Replay attack önlemek için her istekte farklı olmalı.
 */
public final class NonceGenerator {

    private static final SecureRandom SECURE_RANDOM = new SecureRandom();

    private NonceGenerator() {}

    /**
     * 16 byte rastgele üretip Base64 URL-safe string döner.
     * Örn: "a9Kx..._-"
     */
    public static String generate() {
        byte[] bytes = new byte[16];
        SECURE_RANDOM.nextBytes(bytes);
        return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
    }
}
