package ulug.musa.acquirer.fraud;

/**
 * Fraud prediction result from the ML model.
 */
public record FraudPrediction(
        double probability,
        String prediction, // "FRAUD" or "LEGITIMATE"
        String riskLevel, // CRITICAL, HIGH, MEDIUM, LOW, MINIMAL
        double threshold) {
    /**
     * Determine risk level from probability
     */
    public static String classifyRisk(double probability) {
        if (probability >= 0.8)
            return "CRITICAL";
        if (probability >= 0.6)
            return "HIGH";
        if (probability >= 0.4)
            return "MEDIUM";
        if (probability >= 0.2)
            return "LOW";
        return "MINIMAL";
    }
}
