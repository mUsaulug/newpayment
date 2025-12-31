-- Add idempotency key support for transaction history

ALTER TABLE transaction_history
    ADD COLUMN idempotency_key VARCHAR(64);

CREATE INDEX idx_txn_idempotency_key ON transaction_history(idempotency_key);
