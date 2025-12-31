package ulug.musa.acquirer.fraud;

import java.util.Map;
import java.util.LinkedHashMap;

/**
 * DTO containing all 35 features required by the XGBoost fraud model.
 * Features are grouped by category for clarity.
 */
public record FraudFeatures(
        // === TIME FEATURES (8) ===
        int hour,
        int dayOfWeek,
        int day,
        int month,
        int year,
        int isWeekend, // 0 or 1
        int isNight, // 0 or 1
        double age, // User age in years

        // === AMOUNT FEATURES (3) ===
        double amt,
        double amtLog,
        double amtZscore,

        // === GEO FEATURES (4) ===
        double distanceKm,
        double distanceLog,
        double cityPopLog,
        double lat,
        double lng,
        double merchLat,
        double merchLong,
        double cityPop,

        // === BEHAVIORAL FEATURES (9) ===
        int cardTxCount,
        double timeSinceLastTx,
        double timeSinceLastTxLog,
        double cardAvgAmt,
        double amtDeviationFromCardAvg,
        int cardTxSequence,
        int isRecentActive,
        double amtRollingMean3,
        double amtRollingStd3,

        // === CATEGORICAL ENCODED (5) ===
        int categoryEncoded,
        int genderEncoded,
        int stateEncoded,
        int amtBucketEncoded,
        int distanceBucketEncoded,
        double merchantFreq) {
    /**
     * Convert to map for API serialization
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();

        // Time features
        map.put("hour", hour);
        map.put("dayofweek", dayOfWeek);
        map.put("day", day);
        map.put("month", month);
        map.put("year", year);
        map.put("is_weekend", isWeekend);
        map.put("is_night", isNight);
        map.put("age", age);

        // Amount features
        map.put("amt", amt);
        map.put("amt_log", amtLog);
        map.put("amt_zscore", amtZscore);

        // Geo features
        map.put("distance_km", distanceKm);
        map.put("distance_log", distanceLog);
        map.put("city_pop_log", cityPopLog);
        map.put("lat", lat);
        map.put("long", lng);
        map.put("merch_lat", merchLat);
        map.put("merch_long", merchLong);
        map.put("city_pop", cityPop);

        // Behavioral features
        map.put("card_tx_count", cardTxCount);
        map.put("time_since_last_tx", timeSinceLastTx);
        map.put("time_since_last_tx_log", timeSinceLastTxLog);
        map.put("card_avg_amt", cardAvgAmt);
        map.put("amt_deviation_from_card_avg", amtDeviationFromCardAvg);
        map.put("card_tx_sequence", cardTxSequence);
        map.put("is_recent_active", isRecentActive);
        map.put("amt_rolling_mean_3", amtRollingMean3);
        map.put("amt_rolling_std_3", amtRollingStd3);

        // Categorical encoded
        map.put("category_encoded", categoryEncoded);
        map.put("gender_encoded", genderEncoded);
        map.put("state_encoded", stateEncoded);
        map.put("amt_bucket_encoded", amtBucketEncoded);
        map.put("distance_bucket_encoded", distanceBucketEncoded);
        map.put("merchant_freq", merchantFreq);

        return map;
    }
}
