package ulug.musa.acquirer.security.replay;

public interface NonceStore {
    /**
     * Nonce daha önce görülmediyse verilen TTL süresi boyunca saklar ve true döner.
     * Daha önce görüldüyse false döner.
     * <p>
     * TTL değeri saniye cinsindendir ve {@link ulug.musa.common.util.TimeUtil#nowEpochSeconds()}
     * zamanı referans alınarak hesaplanır. Nonce formatı
     * {@link ulug.musa.common.util.NonceGenerator NonceGenerator} çıktısıyla uyumlu
     * olmalıdır (Base64 URL-safe, padding'siz 16 byte).
     */

    boolean storeIfAbsent(String nonce, long ttlSeconds);

    /**
     * Uygulamanın temizleme stratejisini kontrol edebilmek için çağrılır.
     * Çoklu instance dağıtımlarında merkezi saklama çözümleri bu metodu
     * no-op geçebilir; in-memory implementasyonlar için ise süresi dolan
     * nonce'ların silinmesi beklenir.
     *
     * @param referenceEpochSeconds temizleme zamanı (epoch seconds)
     */
    void cleanupExpired(long referenceEpochSeconds);
}
