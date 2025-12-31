package ulug.musa.acquirer.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import ulug.musa.acquirer.domain.UserProfile;

import java.util.Optional;

/**
 * Repository for user profile data access.
 */
@Repository
public interface UserProfileRepository extends JpaRepository<UserProfile, Long> {

    /**
     * Find user profile by PAN token
     */
    Optional<UserProfile> findByPanToken(String panToken);

    /**
     * Check if user exists
     */
    boolean existsByPanToken(String panToken);
}
