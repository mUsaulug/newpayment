package ulug.musa.common.model;

public class FraudFeatureSnapshot {
    private int hour;
    private double distanceKm;
    private double timeSinceLastTx;
    private int isNight;
    private double amtZscore;
    private double cardAvgAmt;

    public FraudFeatureSnapshot() {
    }

    public FraudFeatureSnapshot(int hour, double distanceKm, double timeSinceLastTx, int isNight, double amtZscore,
            double cardAvgAmt) {
        this.hour = hour;
        this.distanceKm = distanceKm;
        this.timeSinceLastTx = timeSinceLastTx;
        this.isNight = isNight;
        this.amtZscore = amtZscore;
        this.cardAvgAmt = cardAvgAmt;
    }

    public int getHour() {
        return hour;
    }

    public void setHour(int hour) {
        this.hour = hour;
    }

    public double getDistanceKm() {
        return distanceKm;
    }

    public void setDistanceKm(double distanceKm) {
        this.distanceKm = distanceKm;
    }

    public double getTimeSinceLastTx() {
        return timeSinceLastTx;
    }

    public void setTimeSinceLastTx(double timeSinceLastTx) {
        this.timeSinceLastTx = timeSinceLastTx;
    }

    public int getIsNight() {
        return isNight;
    }

    public void setIsNight(int isNight) {
        this.isNight = isNight;
    }

    public double getAmtZscore() {
        return amtZscore;
    }

    public void setAmtZscore(double amtZscore) {
        this.amtZscore = amtZscore;
    }

    public double getCardAvgAmt() {
        return cardAvgAmt;
    }

    public void setCardAvgAmt(double cardAvgAmt) {
        this.cardAvgAmt = cardAvgAmt;
    }
}
