package ulug.musa.acquirer.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
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
    List<TransactionHistory> findByPanTokenOrderByTransactionTimeDesc(String panToken, Pageable pageable);

    /**
     * Find last 3 transactions (most common use case for rolling mean/std)
     */
    default List<TransactionHistory> findLast3Transactions(String panToken) {
        return findByPanTokenOrderByTransactionTimeDesc(panToken, PageRequest.of(0, 3));
    }

    /**
     * Find by trace ID (for idempotency check)
     */
    boolean existsByTraceId(String traceId);

    boolean existsByIdempotencyKey(String idempotencyKey);

    /**
     * Count transactions for a card
     */
    long countByPanToken(String panToken);
}
