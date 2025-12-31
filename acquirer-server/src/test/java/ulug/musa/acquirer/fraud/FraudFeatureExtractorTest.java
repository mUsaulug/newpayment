package ulug.musa.acquirer.fraud;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import ulug.musa.acquirer.domain.TransactionHistory;
import ulug.musa.acquirer.domain.UserProfile;
import ulug.musa.common.model.PaymentRequest;
import ulug.musa.common.model.TxnType;

import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for FraudFeatureExtractor.
 */
class FraudFeatureExtractorTest {

    private final FraudFeatureExtractor extractor = new FraudFeatureExtractor();

    @Test
    @DisplayName("Should calculate Haversine distance correctly")
    void testHaversineDistance_knownCoordinates() {
        // Istanbul to Ankara (approx 350 km)
        double distance = extractor.haversineDistance(41.0082, 28.9784, 39.9334, 32.8597);

        assertTrue(distance > 300 && distance < 400,
                "Istanbul to Ankara should be ~350km, got: " + distance);
    }

    @Test
    @DisplayName("Should identify night transactions correctly")
    void testTimeFeatures_nightTransaction() {
        PaymentRequest request = createRequest(BigDecimal.valueOf(100),
                toEpoch(3)); // 03:00 AM
        UserProfile profile = createProfile();

        FraudFeatures features = extractor.extract(
                request, profile, Collections.emptyList(),
                BigDecimal.valueOf(41.0), BigDecimal.valueOf(29.0), "grocery");

        assertEquals(1, features.isNight(), "03:00 should be night");
        assertEquals(3, features.hour());
    }

    @Test
    @DisplayName("Should identify weekend transactions correctly")
    void testTimeFeatures_weekendTransaction() {
        // Epoch for a Saturday
        long saturdayEpoch = 1735401600L; // 2024-12-28 12:00 (Saturday)

        PaymentRequest request = new PaymentRequest();
        request.setAmount(BigDecimal.valueOf(100));
        request.setTimestamp(saturdayEpoch);
        request.setPanToken("tok_test");

        UserProfile profile = createProfile();

        FraudFeatures features = extractor.extract(
                request, profile, Collections.emptyList(),
                null, null, null);

        assertEquals(1, features.isWeekend(), "Saturday should be weekend");
    }

    @Test
    @DisplayName("Should calculate log amount correctly")
    void testAmountFeatures_logCalculation() {
        PaymentRequest request = createRequest(BigDecimal.valueOf(1000), toEpoch(12));
        UserProfile profile = createProfile();

        FraudFeatures features = extractor.extract(
                request, profile, Collections.emptyList(), null, null, null);

        assertEquals(1000.0, features.amt());
        assertEquals(Math.log1p(1000), features.amtLog(), 0.001);
    }

    @Test
    @DisplayName("Should calculate rolling mean for 3 transactions")
    void testRollingMean_with3Transactions() {
        PaymentRequest request = createRequest(BigDecimal.valueOf(400), toEpoch(12));
        UserProfile profile = createProfile();

        // Create past transactions
        List<TransactionHistory> recentTxns = List.of(
                createTxnHistory(100),
                createTxnHistory(200));

        FraudFeatures features = extractor.extract(
                request, profile, recentTxns, null, null, null);

        // Rolling mean of 400, 100, 200 = 233.33
        double expectedMean = (400 + 100 + 200) / 3.0;
        assertEquals(expectedMean, features.amtRollingMean3(), 0.1);
    }

    @Test
    @DisplayName("Should encode amount buckets correctly")
    void testAmountBucketEncoding() {
        // Very low: <= 10
        assertEquals(0, getAmtBucketEncoded(5));
        // Low: 10-50
        assertEquals(1, getAmtBucketEncoded(30));
        // Medium: 50-100
        assertEquals(2, getAmtBucketEncoded(75));
        // High: 100-500
        assertEquals(3, getAmtBucketEncoded(300));
        // Very high: > 500
        assertEquals(4, getAmtBucketEncoded(1000));
    }

    @Test
    @DisplayName("Should calculate user age from profile")
    void testUserAge_calculation() {
        UserProfile profile = new UserProfile("tok_test");
        profile.setDateOfBirth(LocalDate.of(1990, 1, 1));

        double age = profile.getAge();
        assertTrue(age >= 34 && age <= 36, "Age should be around 35, got: " + age);
    }

    // Helper methods
    private PaymentRequest createRequest(BigDecimal amount, long timestamp) {
        PaymentRequest request = new PaymentRequest();
        request.setAmount(amount);
        request.setTimestamp(timestamp);
        request.setPanToken("tok_test");
        request.setTerminalId("TERM001");
        request.setTraceId("trace-001");
        request.setTxnType(TxnType.AUTH);
        return request;
    }

    private UserProfile createProfile() {
        UserProfile profile = new UserProfile("tok_test");
        profile.setGender("M");
        profile.setDateOfBirth(LocalDate.of(1989, 5, 15));
        profile.setHomeLat(BigDecimal.valueOf(41.0));
        profile.setHomeLong(BigDecimal.valueOf(29.0));
        profile.setCity("Istanbul");
        profile.setState("Istanbul");
        profile.setCityPop(15000000);
        profile.setAvgAmount(BigDecimal.valueOf(200));
        profile.setTransactionCount(50);
        return profile;
    }

    private TransactionHistory createTxnHistory(double amount) {
        return TransactionHistory.from(
                "tok_test", "trace-old", BigDecimal.valueOf(amount),
                "TERM001", BigDecimal.valueOf(0.1), "LOW", "APPROVED",
                null, LocalDateTime.now().minusDays(1));
    }

    private long toEpoch(int hour) {
        // Returns epoch for today at specified hour (Turkey timezone)
        return LocalDateTime.now()
                .withHour(hour)
                .withMinute(0)
                .withSecond(0)
                .atZone(java.time.ZoneId.of("Europe/Istanbul"))
                .toEpochSecond();
    }

    private int getAmtBucketEncoded(double amount) {
        PaymentRequest request = createRequest(BigDecimal.valueOf(amount), toEpoch(12));
        UserProfile profile = createProfile();
        FraudFeatures features = extractor.extract(
                request, profile, Collections.emptyList(), null, null, null);
        return features.amtBucketEncoded();
    }
}
