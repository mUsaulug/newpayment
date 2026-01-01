package ulug.musa.acquirer.fraud;

import java.util.List;
import ulug.musa.common.model.FraudFeatureSnapshot;

/**
 * Final fraud decision with explanation.
 */
public record FraudDecision(
        Decision decision,
        double fraudScore,
        String riskLevel,
        List<String> reasons,
        FraudFeatureSnapshot features) {
    public enum Decision {
        APPROVED, // Score < 0.65
        PENDING, // Score 0.65 - 0.85 (manual review)
        DECLINED // Score >= 0.85
    }

    /**
     * Get response code for payment response
     */
    public String getResponseCode() {
        return switch (decision) {
            case APPROVED -> "00";
            case PENDING -> "01"; // Referral
            case DECLINED -> "05"; // Do not honor
        };
    }

    /**
     * Get human-readable message
     */
    public String getMessage() {
        return switch (decision) {
            case APPROVED -> "APPROVED";
            case PENDING -> "PENDING - Manuel onay gerekli";
            case DECLINED -> "DECLINED - Fraud riski tespit edildi";
        };
    }

    /**
     * Is this transaction approved?
     */
    public boolean isApproved() {
        return decision == Decision.APPROVED;
    }
}
