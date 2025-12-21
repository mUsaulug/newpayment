package ulug.musa.acquirer.security.replay;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.LongSupplier;

public class InMemoryNonceStore implements NonceStore {

    private final Map<String, Long> nonceExpiryEpochSeconds = new ConcurrentHashMap<>();
    private final LongSupplier epochSecondsSupplier;

    public InMemoryNonceStore(LongSupplier epochSecondsSupplier) {
        this.epochSecondsSupplier = Objects.requireNonNull(epochSecondsSupplier, "epochSecondsSupplier gerekli");
    }

    @Override
    public boolean storeIfAbsent(String nonce, long ttlSeconds) {
        long now = epochSecondsSupplier.getAsLong();
        long exp = now + ttlSeconds;

        // Basit temizlik: süresi dolan nonce'ları sil
        nonceExpiryEpochSeconds.entrySet().removeIf(e -> e.getValue() <= now);

        // putIfAbsent: nonce yoksa ekler, varsa dokunmaz
        Long existing = nonceExpiryEpochSeconds.putIfAbsent(nonce, exp);
        return existing == null;
    }
}
