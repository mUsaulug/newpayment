package ulug.musa.acquirer.security.replay;

public interface NonceStore {
    /**
     * Nonce daha önce görülmediyse kaydeder ve true döner.
     * Daha önce görüldüyse false döner.
     */
    boolean storeIfAbsent(String nonce, long ttlSeconds);
}
