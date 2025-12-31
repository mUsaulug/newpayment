package ulug.musa.acquirer.fraud;

import org.springframework.stereotype.Service;
import ulug.musa.acquirer.domain.TransactionHistory;
import ulug.musa.acquirer.domain.UserProfile;
import ulug.musa.common.model.PaymentRequest;

import java.math.BigDecimal;
import java.time.*;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Map;

/**
 * Service for extracting fraud detection features from payment request + user
 * history.
 * Implements all 35 features required by the XGBoost model.
 */
@Service
public class FraudFeatureExtractor {

    // Global statistics (from training data - should be configurable)
    private static final double GLOBAL_AMT_MEAN = 70.35;
    private static final double GLOBAL_AMT_STD = 160.41;

    // Category encoding map (from LabelEncoder)
    private static final Map<String, Integer> CATEGORY_ENCODING = Map.ofEntries(
            Map.entry("entertainment", 0),
            Map.entry("food_dining", 1),
            Map.entry("gas_transport", 2),
            Map.entry("grocery_net", 3),
            Map.entry("grocery_pos", 4),
            Map.entry("health_fitness", 5),
            Map.entry("home", 6),
            Map.entry("kids_pets", 7),
            Map.entry("misc_net", 8),
            Map.entry("misc_pos", 9),
            Map.entry("personal_care", 10),
            Map.entry("shopping_net", 11),
            Map.entry("shopping_pos", 12),
            Map.entry("travel", 13),
            Map.entry("grocery", 14),
            Map.entry("restaurant", 15),
            Map.entry("cafe", 16),
            Map.entry("shopping", 17),
            Map.entry("electronics", 18));

    // State encoding (Turkish cities/regions)
    private static final Map<String, Integer> STATE_ENCODING = Map.ofEntries(
            Map.entry("Istanbul", 0),
            Map.entry("Ankara", 1),
            Map.entry("Izmir", 2),
            Map.entry("Bursa", 3),
            Map.entry("Antalya", 4));

    /**
     * Extract all features from payment request, user profile, and transaction
     * history.
     */
    public FraudFeatures extract(PaymentRequest request, UserProfile profile,
            List<TransactionHistory> recentTxns,
            BigDecimal merchantLat, BigDecimal merchantLong,
            String merchantCategory) {

        // Parse transaction time
        LocalDateTime txnTime = Instant.ofEpochSecond(request.getTimestamp())
                .atZone(ZoneId.of("Europe/Istanbul"))
                .toLocalDateTime();

        double amount = request.getAmount().doubleValue();

        // === TIME FEATURES ===
        int hour = txnTime.getHour();
        int dayOfWeek = txnTime.getDayOfWeek().getValue() % 7; // 0=Monday
        int day = txnTime.getDayOfMonth();
        int month = txnTime.getMonthValue();
        int year = txnTime.getYear();
        int isWeekend = (dayOfWeek >= 5) ? 1 : 0;
        int isNight = (hour >= 22 || hour <= 5) ? 1 : 0;
        double age = profile.getAge();

        // === AMOUNT FEATURES ===
        double amtLog = Math.log1p(amount);
        double amtZscore = (amount - GLOBAL_AMT_MEAN) / GLOBAL_AMT_STD;

        // === GEO FEATURES ===
        double userLat = profile.getHomeLat() != null ? profile.getHomeLat().doubleValue() : 41.0;
        double userLng = profile.getHomeLong() != null ? profile.getHomeLong().doubleValue() : 29.0;
        double mLat = merchantLat != null ? merchantLat.doubleValue() : userLat;
        double mLng = merchantLong != null ? merchantLong.doubleValue() : userLng;

        double distanceKm = haversineDistance(userLat, userLng, mLat, mLng);
        double distanceLog = Math.log1p(distanceKm);
        double cityPop = profile.getCityPop() != null ? profile.getCityPop() : 1000000;
        double cityPopLog = Math.log1p(cityPop);

        // === BEHAVIORAL FEATURES ===
        int cardTxCount = profile.getTransactionCount() != null ? profile.getTransactionCount() : 0;
        double cardAvgAmt = profile.getAvgAmount() != null ? profile.getAvgAmount().doubleValue() : amount;
        double amtDeviation = amount - cardAvgAmt;

        // Time since last transaction
        double timeSinceLastTx = calculateTimeSinceLastTx(recentTxns, txnTime);
        double timeSinceLastTxLog = Math.log1p(timeSinceLastTx);

        // Transaction sequence
        int cardTxSequence = cardTxCount + 1;
        int isRecentActive = (cardTxSequence > cardTxCount - 10) ? 1 : 0;

        // Rolling statistics from last 3 transactions
        double amtRollingMean3 = calculateRollingMean(recentTxns, amount);
        double amtRollingStd3 = calculateRollingStd(recentTxns, amount, amtRollingMean3);

        // === CATEGORICAL ENCODING ===
        int categoryEncoded = CATEGORY_ENCODING.getOrDefault(
                merchantCategory != null ? merchantCategory.toLowerCase() : "", 9);
        int genderEncoded = "F".equalsIgnoreCase(profile.getGender()) ? 0 : 1;
        int stateEncoded = STATE_ENCODING.getOrDefault(profile.getState(), -1);
        int amtBucketEncoded = encodeAmountBucket(amount);
        int distanceBucketEncoded = encodeDistanceBucket(distanceKm);
        double merchantFreq = 0.001; // Default low frequency for new merchants

        return new FraudFeatures(
                hour, dayOfWeek, day, month, year, isWeekend, isNight, age,
                amount, amtLog, amtZscore,
                distanceKm, distanceLog, cityPopLog, userLat, userLng, mLat, mLng, cityPop,
                cardTxCount, timeSinceLastTx, timeSinceLastTxLog, cardAvgAmt, amtDeviation,
                cardTxSequence, isRecentActive, amtRollingMean3, amtRollingStd3,
                categoryEncoded, genderEncoded, stateEncoded, amtBucketEncoded, distanceBucketEncoded, merchantFreq);
    }

    /**
     * Haversine formula: calculate distance between two lat/long points in km
     */
    public double haversineDistance(double lat1, double lon1, double lat2, double lon2) {
        final double R = 6371; // Earth radius in km

        double lat1Rad = Math.toRadians(lat1);
        double lat2Rad = Math.toRadians(lat2);
        double deltaLat = Math.toRadians(lat2 - lat1);
        double deltaLon = Math.toRadians(lon2 - lon1);

        double a = Math.sin(deltaLat / 2) * Math.sin(deltaLat / 2) +
                Math.cos(lat1Rad) * Math.cos(lat2Rad) *
                        Math.sin(deltaLon / 2) * Math.sin(deltaLon / 2);
        double c = 2 * Math.asin(Math.sqrt(a));

        return R * c;
    }

    /**
     * Calculate time since last transaction in seconds
     */
    private double calculateTimeSinceLastTx(List<TransactionHistory> recentTxns, LocalDateTime currentTime) {
        if (recentTxns.isEmpty()) {
            return 999999; // Large value for first transaction
        }
        LocalDateTime lastTxTime = recentTxns.get(0).getTransactionTime();
        return ChronoUnit.SECONDS.between(lastTxTime, currentTime);
    }

    /**
     * Calculate rolling mean of last 3 transactions (including current)
     */
    private double calculateRollingMean(List<TransactionHistory> recentTxns, double currentAmount) {
        if (recentTxns.isEmpty()) {
            return currentAmount;
        }

        double sum = currentAmount;
        int count = 1;

        for (int i = 0; i < Math.min(2, recentTxns.size()); i++) {
            sum += recentTxns.get(i).getAmount().doubleValue();
            count++;
        }

        return sum / count;
    }

    /**
     * Calculate rolling standard deviation of last 3 transactions
     */
    private double calculateRollingStd(List<TransactionHistory> recentTxns,
            double currentAmount, double mean) {
        if (recentTxns.isEmpty()) {
            return 0;
        }

        double sumSquares = Math.pow(currentAmount - mean, 2);
        int count = 1;

        for (int i = 0; i < Math.min(2, recentTxns.size()); i++) {
            double amt = recentTxns.get(i).getAmount().doubleValue();
            sumSquares += Math.pow(amt - mean, 2);
            count++;
        }

        return Math.sqrt(sumSquares / count);
    }

    /**
     * Encode amount into bucket: 0=very_low, 1=low, 2=medium, 3=high, 4=very_high
     */
    private int encodeAmountBucket(double amount) {
        if (amount <= 10)
            return 0;
        if (amount <= 50)
            return 1;
        if (amount <= 100)
            return 2;
        if (amount <= 500)
            return 3;
        return 4;
    }

    /**
     * Encode distance into bucket: 0=same_city, 1=nearby, 2=regional, 3=far,
     * 4=very_far
     */
    private int encodeDistanceBucket(double distanceKm) {
        if (distanceKm <= 1)
            return 0;
        if (distanceKm <= 10)
            return 1;
        if (distanceKm <= 50)
            return 2;
        if (distanceKm <= 200)
            return 3;
        return 4;
    }
}
