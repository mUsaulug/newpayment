-- Demo users for testing fraud detection scenarios

-- User 1: Ahmet (Normal user, Istanbul, moderate spender)
INSERT INTO user_profiles (pan_token, gender, date_of_birth, home_lat, home_long, city, state, city_pop, avg_amount, transaction_count)
VALUES ('tok_ahmet_001', 'M', '1989-05-15', 40.9886, 29.0271, 'Kadıköy', 'Istanbul', 15000000, 450.00, 127);

-- User 2: Ayşe (Normal user, Ankara, low spender)
INSERT INTO user_profiles (pan_token, gender, date_of_birth, home_lat, home_long, city, state, city_pop, avg_amount, transaction_count)
VALUES ('tok_ayse_002', 'F', '1996-08-22', 39.9334, 32.8597, 'Çankaya', 'Ankara', 5500000, 200.00, 45);

-- User 3: Mehmet (Recently relocated user, Izmir -> Ankara)
INSERT INTO user_profiles (pan_token, gender, date_of_birth, home_lat, home_long, city, state, city_pop, avg_amount, transaction_count)
VALUES ('tok_mehmet_003', 'M', '1984-11-30', 38.4237, 27.1428, 'Konak', 'Izmir', 4400000, 1200.00, 89);

-- Sample transaction history for Ahmet (normal pattern)
INSERT INTO transaction_history (pan_token, trace_id, amount, terminal_id, merchant_lat, merchant_long, merchant_category, fraud_score, risk_level, decision, transaction_time) VALUES
('tok_ahmet_001', 'hist-001', 150.00, 'TERM001', 40.9912, 29.0228, 'grocery', 0.05, 'MINIMAL', 'APPROVED', NOW() - INTERVAL '7 days'),
('tok_ahmet_001', 'hist-002', 520.00, 'TERM002', 41.0082, 28.9784, 'restaurant', 0.08, 'MINIMAL', 'APPROVED', NOW() - INTERVAL '5 days'),
('tok_ahmet_001', 'hist-003', 380.00, 'TERM001', 40.9886, 29.0271, 'grocery', 0.04, 'MINIMAL', 'APPROVED', NOW() - INTERVAL '2 days');

-- Sample transaction history for Ayşe (normal pattern)
INSERT INTO transaction_history (pan_token, trace_id, amount, terminal_id, merchant_lat, merchant_long, merchant_category, fraud_score, risk_level, decision, transaction_time) VALUES
('tok_ayse_002', 'hist-011', 85.00, 'TERM010', 39.9400, 32.8600, 'cafe', 0.03, 'MINIMAL', 'APPROVED', NOW() - INTERVAL '10 days'),
('tok_ayse_002', 'hist-012', 220.00, 'TERM011', 39.9350, 32.8550, 'shopping', 0.06, 'MINIMAL', 'APPROVED', NOW() - INTERVAL '3 days');

-- Sample transaction history for Mehmet (higher amounts)
INSERT INTO transaction_history (pan_token, trace_id, amount, terminal_id, merchant_lat, merchant_long, merchant_category, fraud_score, risk_level, decision, transaction_time) VALUES
('tok_mehmet_003', 'hist-021', 1500.00, 'TERM020', 38.4200, 27.1400, 'electronics', 0.12, 'LOW', 'APPROVED', NOW() - INTERVAL '14 days'),
('tok_mehmet_003', 'hist-022', 950.00, 'TERM021', 38.4250, 27.1450, 'shopping', 0.09, 'MINIMAL', 'APPROVED', NOW() - INTERVAL '6 days');
