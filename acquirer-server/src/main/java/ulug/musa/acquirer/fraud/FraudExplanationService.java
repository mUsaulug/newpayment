package ulug.musa.acquirer.fraud;

import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Service for generating human-readable explanations of fraud decisions.
 * Uses rule-based logic to explain why a transaction was flagged.
 */
@Service
public class FraudExplanationService {

    /**
     * Generate explanation for fraud decision based on features and prediction.
     */
    public List<String> explain(FraudFeatures features, FraudPrediction prediction) {
        List<String> reasons = new ArrayList<>();

        // Only explain if there's some risk
        if (prediction.probability() < 0.2) {
            return reasons; // No explanation needed for very low risk
        }

        // === TIME-BASED EXPLANATIONS ===
        if (features.isNight() == 1) {
            reasons.add("Gece saatlerinde (22:00-05:00) yapılan işlem");
        }

        if (features.isWeekend() == 1 && features.amt() > 1000) {
            reasons.add("Hafta sonu yüksek tutarlı işlem");
        }

        // === AMOUNT-BASED EXPLANATIONS ===
        if (features.amtZscore() > 3) {
            reasons.add(String.format("Tutar, ortalamadan %.1f standart sapma yüksek", features.amtZscore()));
        } else if (features.amtZscore() > 2) {
            reasons.add("Tutar, normal işlem ortalamasının üzerinde");
        }

        if (features.amtDeviationFromCardAvg() > 0 && features.cardAvgAmt() > 0) {
            double ratio = features.amt() / features.cardAvgAmt();
            if (ratio > 10) {
                reasons.add(String.format("Tutar, kart ortalamasının %.0fx üzerinde (%.2f TL vs %.2f TL)",
                        ratio, features.amt(), features.cardAvgAmt()));
            } else if (ratio > 5) {
                reasons.add("Tutar, kart ortalamasının belirgin üzerinde");
            }
        }

        // === LOCATION-BASED EXPLANATIONS ===
        if (features.distanceKm() > 1000) {
            reasons.add(String.format("Çok uzak lokasyondan işlem (%.0f km)", features.distanceKm()));
        } else if (features.distanceKm() > 500) {
            reasons.add(String.format("Uzak lokasyondan işlem (%.0f km)", features.distanceKm()));
        } else if (features.distanceKm() > 100) {
            reasons.add("Farklı şehirden işlem");
        }

        // === BEHAVIORAL EXPLANATIONS ===
        if (features.timeSinceLastTx() < 60) {
            reasons.add("Son işlemden çok kısa süre sonra tekrar işlem (<1 dakika)");
        } else if (features.timeSinceLastTx() < 300) {
            reasons.add("Son işlemden kısa süre sonra tekrar işlem (<5 dakika)");
        }

        if (features.cardTxCount() == 0) {
            reasons.add("İlk kez işlem yapan kart");
        } else if (features.cardTxCount() < 3) {
            reasons.add("Çok az işlem geçmişi olan kart");
        }

        // Rolling volatility check
        if (features.amtRollingStd3() > 0) {
            double cv = features.amtRollingStd3() / features.amtRollingMean3();
            if (cv > 1.5) {
                reasons.add("Son işlemlerde yüksek tutar dalgalanması");
            }
        }

        // === VELOCITY CHECKS ===
        if (features.isRecentActive() == 1 && features.cardTxSequence() > 10) {
            double velocityScore = features.cardTxSequence() / Math.max(1, features.timeSinceLastTxLog());
            if (velocityScore > 5) {
                reasons.add("Yüksek işlem hızı tespit edildi");
            }
        }

        // Cap reasons at 5 for readability
        if (reasons.size() > 5) {
            reasons = reasons.subList(0, 5);
        }

        return reasons;
    }
}
