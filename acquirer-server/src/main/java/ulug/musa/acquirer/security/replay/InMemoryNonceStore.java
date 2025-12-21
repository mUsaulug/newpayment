package ulug.musa.acquirer.security.replay;

import ulug.musa.common.util.TimeUtil;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.LongSupplier;
import java.util.regex.Pattern;

public class InMemoryNonceStore implements NonceStore {

    private static final Pattern NONCE_PATTERN = Pattern.compile("^[A-Za-z0-9_-]{22}$");

    private final Map<String, Long> nonceExpiryEpochSeconds = new ConcurrentHashMap<>();
    private final LongSupplier epochSecondsSupplier;
    private final long cleanupIntervalSeconds;
    private volatile long lastCleanupEpochSeconds;

    public InMemoryNonceStore(LongSupplier epochSecondsSupplier, long cleanupIntervalSeconds) {
        this.epochSecondsSupplier = epochSecondsSupplier == null ? TimeUtil::nowEpochSeconds : epochSecondsSupplier;
        if (cleanupIntervalSeconds <= 0) {
            throw new IllegalArgumentException("cleanupIntervalSeconds 0'dan büyük olmalı");
        }
        this.cleanupIntervalSeconds = cleanupIntervalSeconds;
        this.lastCleanupEpochSeconds = nowEpochSeconds();
    }

    @Override
    public boolean storeIfAbsent(String nonce, long ttlSeconds) {
        Objects.requireNonNull(nonce, "nonce gerekli");
        if (!NONCE_PATTERN.matcher(nonce).matches()) {
            throw new IllegalArgumentException("Nonce formatı NonceGenerator ile uyumlu olmalı (Base64 URL-safe 16 byte)");
        }
        if (ttlSeconds <= 0) {
            throw new IllegalArgumentException("ttlSeconds 0'dan büyük olmalı");
        }

        long now = nowEpochSeconds();
        long exp = now + ttlSeconds;

        maybeCleanup(now);

        // putIfAbsent: nonce yoksa ekler, varsa dokunmaz
        Long existing = nonceExpiryEpochSeconds.putIfAbsent(nonce, exp);
        return existing == null;
    }

    @Override
    public void cleanupExpired(long referenceEpochSeconds) {
        nonceExpiryEpochSeconds.entrySet().removeIf(e -> e.getValue() <= referenceEpochSeconds);
    }

    private void maybeCleanup(long now) {
        if (now - lastCleanupEpochSeconds >= cleanupIntervalSeconds) {
            cleanupExpired(now);
            lastCleanupEpochSeconds = now;
        }
    }

    private long nowEpochSeconds() {
        return epochSecondsSupplier == null ? TimeUtil.nowEpochSeconds() : epochSecondsSupplier.getAsLong();
    }
}
