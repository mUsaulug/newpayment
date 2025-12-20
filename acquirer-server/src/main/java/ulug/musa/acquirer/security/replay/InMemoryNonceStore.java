package ulug.musa.acquirer.security.replay;

import java.time.Instant;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class InMemoryNonceStore implements NonceStore {

    private final Map<String, Long> nonceExpiryEpochSeconds = new ConcurrentHashMap<>();

    @Override
    public boolean storeIfAbsent(String nonce, long ttlSeconds) {
        long now = Instant.now().getEpochSecond();
        long exp = now + ttlSeconds;

        // Basit temizlik: süresi dolan nonce'ları sil
        nonceExpiryEpochSeconds.entrySet().removeIf(e -> e.getValue() <= now);

        // putIfAbsent: nonce yoksa ekler, varsa dokunmaz
        Long existing = nonceExpiryEpochSeconds.putIfAbsent(nonce, exp);
        return existing == null;
    }
}
