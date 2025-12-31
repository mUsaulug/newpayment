package ulug.musa.acquirer.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import ulug.musa.acquirer.domain.TransactionHistory;

import java.util.List;

/**
 * Repository for transaction history data access.
 * Provides methods for rolling statistics calculations.
 */
@Repository
public interface TransactionHistoryRepository extends JpaRepository<TransactionHistory, Long> {

    /**
     * Find last N transactions for a card (for rolling statistics)
     */
    @Query("SELECT t FROM TransactionHistory t WHERE t.panToken = :panToken ORDER BY t.transactionTime DESC LIMIT :limit")
    List<TransactionHistory> findLastNTransactions(@Param("panToken") String panToken, @Param("limit") int limit);

    /**
     * Find last 3 transactions (most common use case for rolling mean/std)
     */
    default List<TransactionHistory> findLast3Transactions(String panToken) {
        return findLastNTransactions(panToken, 3);
    }

    /**
     * Find by trace ID (for idempotency check)
     */
    boolean existsByTraceId(String traceId);

    /**
     * Count transactions for a card
     */
    long countByPanToken(String panToken);
}
