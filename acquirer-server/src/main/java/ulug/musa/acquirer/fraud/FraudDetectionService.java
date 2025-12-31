package ulug.musa.acquirer.fraud;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import ulug.musa.acquirer.domain.TransactionHistory;
import ulug.musa.acquirer.domain.UserProfile;
import ulug.musa.acquirer.repository.TransactionHistoryRepository;
import ulug.musa.acquirer.repository.UserProfileRepository;
import ulug.musa.common.model.PaymentRequest;

import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.List;

/**
 * Core fraud detection orchestration service.
 * Coordinates feature extraction, model prediction, and decision making.
 */
@Service
public class FraudDetectionService {

    private static final Logger log = LoggerFactory.getLogger(FraudDetectionService.class);

    // Decision thresholds
    private static final double DECLINE_THRESHOLD = 0.85;
    private static final double PENDING_THRESHOLD = 0.65;

    private final FraudApiClient fraudApiClient;
    private final FraudFeatureExtractor featureExtractor;
    private final FraudExplanationService explanationService;
    private final UserProfileRepository userProfileRepo;
    private final TransactionHistoryRepository txnHistoryRepo;

    public FraudDetectionService(
            FraudApiClient fraudApiClient,
            FraudFeatureExtractor featureExtractor,
            FraudExplanationService explanationService,
            UserProfileRepository userProfileRepo,
            TransactionHistoryRepository txnHistoryRepo) {
        this.fraudApiClient = fraudApiClient;
        this.featureExtractor = featureExtractor;
        this.explanationService = explanationService;
        this.userProfileRepo = userProfileRepo;
        this.txnHistoryRepo = txnHistoryRepo;
    }

    /**
     * Evaluate a payment request for fraud.
     * 
     * @param request          Payment request to evaluate
     * @param merchantLat      Merchant latitude (optional)
     * @param merchantLong     Merchant longitude (optional)
     * @param merchantCategory Merchant category code (optional)
     * @return Fraud decision with score and explanation
     */
    public FraudDecision evaluate(PaymentRequest request,
            BigDecimal merchantLat,
            BigDecimal merchantLong,
            String merchantCategory) {

        String panToken = request.getPanToken();
        log.info("Evaluating fraud for panToken={}, amount={}", panToken, request.getAmount());

        // 1. Get or create user profile
        UserProfile profile = userProfileRepo.findByPanToken(panToken)
                .orElseGet(() -> createNewProfile(panToken));

        // 2. Get recent transactions for rolling stats
        List<TransactionHistory> recentTxns = txnHistoryRepo.findLast3Transactions(panToken);

        // 3. Extract features
        FraudFeatures features = featureExtractor.extract(
                request, profile, recentTxns, merchantLat, merchantLong, merchantCategory);

        // 4. Call fraud ML model
        FraudPrediction prediction = fraudApiClient.predict(features);

        // 5. Make decision based on score
        FraudDecision.Decision decision = makeDecision(prediction.probability());

        // 6. Generate explanation
        List<String> reasons = explanationService.explain(features, prediction);

        log.info("Fraud evaluation complete: score={}, decision={}",
                prediction.probability(), decision);

        return new FraudDecision(decision, prediction.probability(), prediction.riskLevel(), reasons);
    }

    /**
     * Create a new user profile for first-time users.
     */
    private UserProfile createNewProfile(String panToken) {
        log.info("Creating new user profile for panToken={}", panToken);
        UserProfile profile = new UserProfile(panToken);
        // Set defaults for new users
        profile.setGender("M"); // Default, should come from card issuer
        profile.setTransactionCount(0);
        profile.setAvgAmount(BigDecimal.ZERO);
        return userProfileRepo.save(profile);
    }

    /**
     * Make approval decision based on fraud probability.
     */
    private FraudDecision.Decision makeDecision(double probability) {
        if (probability >= DECLINE_THRESHOLD) {
            return FraudDecision.Decision.DECLINED;
        } else if (probability >= PENDING_THRESHOLD) {
            return FraudDecision.Decision.PENDING;
        } else {
            return FraudDecision.Decision.APPROVED;
        }
    }

    /**
     * Save transaction to history and update user profile.
     */
    public void recordTransaction(PaymentRequest request, FraudDecision decision,
            BigDecimal merchantLat, BigDecimal merchantLong,
            String merchantCategory) {

        LocalDateTime txnTime = Instant.ofEpochSecond(request.getTimestamp())
                .atZone(ZoneId.of("Europe/Istanbul"))
                .toLocalDateTime();

        // Save transaction
        TransactionHistory history = TransactionHistory.from(
                request.getPanToken(),
                request.getTraceId(),
                request.getAmount(),
                request.getTerminalId(),
                BigDecimal.valueOf(decision.fraudScore()),
                decision.riskLevel(),
                decision.decision().name(),
                String.join("; ", decision.reasons()),
                txnTime,
                request.getIdempotencyKey());
        history.setMerchantLat(merchantLat);
        history.setMerchantLong(merchantLong);
        history.setMerchantCategory(merchantCategory);
        txnHistoryRepo.save(history);

        // Update user profile stats
        if (decision.isApproved()) {
            updateUserProfile(request.getPanToken(), request.getAmount());
        }
    }

    /**
     * Update user profile with new transaction data.
     */
    private void updateUserProfile(String panToken, BigDecimal amount) {
        userProfileRepo.findByPanToken(panToken).ifPresent(profile -> {
            int newCount = profile.getTransactionCount() + 1;
            BigDecimal currentAvg = profile.getAvgAmount() != null ? profile.getAvgAmount() : BigDecimal.ZERO;

            // Calculate new running average
            BigDecimal newAvg = currentAvg
                    .multiply(BigDecimal.valueOf(profile.getTransactionCount()))
                    .add(amount)
                    .divide(BigDecimal.valueOf(newCount), 2, java.math.RoundingMode.HALF_UP);

            profile.setTransactionCount(newCount);
            profile.setAvgAmount(newAvg);
            userProfileRepo.save(profile);
        });
    }
}
