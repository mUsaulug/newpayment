package ulug.musa.common.security;

/**
 * Bu projede kullandığımız güvenlik header isimlerini tek yerde topluyoruz.
 * Neden?
 * - Yazım hatası (typo) olmasın
 * - Her yerde aynı isim kullanılsın
 * - Değişiklik olursa tek yerden güncelleyelim
 */
public final class SecurityHeaders {

    private SecurityHeaders() {}

    public static final String TERMINAL_ID = "X-Terminal-Id";
    public static final String NONCE = "X-Nonce";
    public static final String TIMESTAMP = "X-Timestamp";
    public static final String SIGNATURE = "X-Signature";
}
