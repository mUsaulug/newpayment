package ulug.musa.acquirer.domain;

import jakarta.persistence.*;
import java.math.BigDecimal;
import java.time.LocalDate;
import java.time.LocalDateTime;

/**
 * User profile entity for fraud detection.
 * Stores card-holder behavioral patterns and demographic info.
 */
@Entity
@Table(name = "user_profiles")
public class UserProfile {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(name = "pan_token", unique = true, nullable = false, length = 64)
    private String panToken;

    @Column(length = 1)
    private String gender;

    @Column(name = "date_of_birth")
    private LocalDate dateOfBirth;

    @Column(name = "home_lat", precision = 10, scale = 6)
    private BigDecimal homeLat;

    @Column(name = "home_long", precision = 10, scale = 6)
    private BigDecimal homeLong;

    @Column(length = 100)
    private String city;

    @Column(length = 50)
    private String state;

    @Column(name = "city_pop")
    private Integer cityPop = 0;

    @Column(name = "avg_amount", precision = 12, scale = 2)
    private BigDecimal avgAmount = BigDecimal.ZERO;

    @Column(name = "transaction_count")
    private Integer transactionCount = 0;

    @Column(name = "created_at")
    private LocalDateTime createdAt;

    @Column(name = "updated_at")
    private LocalDateTime updatedAt;

    public UserProfile() {
    }

    public UserProfile(String panToken) {
        this.panToken = panToken;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = LocalDateTime.now();
    }

    @PrePersist
    protected void onCreate() {
        createdAt = LocalDateTime.now();
        updatedAt = LocalDateTime.now();
    }

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }

    /**
     * Calculate user age in years from date of birth
     */
    public double getAge() {
        if (dateOfBirth == null)
            return 35.0; // default
        return java.time.temporal.ChronoUnit.DAYS.between(dateOfBirth, LocalDate.now()) / 365.25;
    }

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getPanToken() {
        return panToken;
    }

    public void setPanToken(String panToken) {
        this.panToken = panToken;
    }

    public String getGender() {
        return gender;
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    public LocalDate getDateOfBirth() {
        return dateOfBirth;
    }

    public void setDateOfBirth(LocalDate dateOfBirth) {
        this.dateOfBirth = dateOfBirth;
    }

    public BigDecimal getHomeLat() {
        return homeLat;
    }

    public void setHomeLat(BigDecimal homeLat) {
        this.homeLat = homeLat;
    }

    public BigDecimal getHomeLong() {
        return homeLong;
    }

    public void setHomeLong(BigDecimal homeLong) {
        this.homeLong = homeLong;
    }

    public String getCity() {
        return city;
    }

    public void setCity(String city) {
        this.city = city;
    }

    public String getState() {
        return state;
    }

    public void setState(String state) {
        this.state = state;
    }

    public Integer getCityPop() {
        return cityPop;
    }

    public void setCityPop(Integer cityPop) {
        this.cityPop = cityPop;
    }

    public BigDecimal getAvgAmount() {
        return avgAmount;
    }

    public void setAvgAmount(BigDecimal avgAmount) {
        this.avgAmount = avgAmount;
    }

    public Integer getTransactionCount() {
        return transactionCount;
    }

    public void setTransactionCount(Integer transactionCount) {
        this.transactionCount = transactionCount;
    }

    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }
}
