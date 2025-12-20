package ulug.musa.common.util;

import java.time.Instant;

/**
 * İsteklerde timestamp kullanacağız (epoch seconds).
 * Sunucu tarafında zaman penceresi kontrolü (örn 120 sn) yapılacak.
 */
public final class TimeUtil
{

    private TimeUtil() {}

    /** Şu anki epoch seconds */
    public static long nowEpochSeconds() {
        return Instant.now().getEpochSecond();
    }
}
