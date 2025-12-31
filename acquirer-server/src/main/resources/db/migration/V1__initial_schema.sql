-- Fraud Detection Integration - Initial Schema
-- V1: User profiles and transaction history tables

-- User profiles (card-based identity)
CREATE TABLE user_profiles (
    id BIGSERIAL PRIMARY KEY,
    pan_token VARCHAR(64) UNIQUE NOT NULL,
    gender VARCHAR(1),  -- 'M', 'F'
    date_of_birth DATE,
    home_lat DECIMAL(10,6),
    home_long DECIMAL(10,6),
    city VARCHAR(100),
    state VARCHAR(50),
    city_pop INTEGER DEFAULT 0,
    avg_amount DECIMAL(12,2) DEFAULT 0,
    transaction_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Transaction history (for rolling stats and audit)
CREATE TABLE transaction_history (
    id BIGSERIAL PRIMARY KEY,
    pan_token VARCHAR(64) NOT NULL,
    trace_id VARCHAR(64) NOT NULL,
    amount DECIMAL(12,2) NOT NULL,
    terminal_id VARCHAR(32),
    merchant_lat DECIMAL(10,6),
    merchant_long DECIMAL(10,6),
    merchant_category VARCHAR(50),
    fraud_score DECIMAL(5,4),
    risk_level VARCHAR(20),
    decision VARCHAR(20) NOT NULL,  -- APPROVED, DECLINED, PENDING
    fraud_reasons TEXT,
    transaction_time TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_user_pan_token ON user_profiles(pan_token);
CREATE INDEX idx_txn_pan_time ON transaction_history(pan_token, transaction_time DESC);
CREATE INDEX idx_txn_trace_id ON transaction_history(trace_id);
