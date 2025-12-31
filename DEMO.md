# Fraud Detection Integration - Demo Script

## Prerequisites

1. **PostgreSQL Database**
   ```powershell
   # Start PostgreSQL with Docker
   docker run -d --name pos-postgres -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=pos_payment postgres:15
   ```

2. **Python Environment**
   ```powershell
   cd fraudAgent
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. **Start Fraud API**
   ```powershell
   cd fraudAgent
   python fraud_api.py
   # API runs on http://localhost:8000
   ```

4. **Start Payment Server**
   ```powershell
   cd acquirer-server
   .\gradlew bootRun
   # Server runs on https://localhost:8443
   ```

## Demo Scenarios

### Senaryo 1: Normal Transaction (Ahmet - Istanbul)
Expected: **APPROVED** (score < 0.3)

```powershell
curl -k -X POST https://localhost:8443/api/payments `
  -H "Content-Type: application/json" `
  -H "X-Merchant-Lat: 40.9912" `
  -H "X-Merchant-Long: 29.0228" `
  -H "X-Merchant-Category: grocery" `
  -d '{
    "terminalId": "TERM001",
    "traceId": "demo-normal-001",
    "txnType": "AUTH",
    "amount": 150.00,
    "currency": "TRY",
    "panToken": "tok_ahmet_001",
    "timestamp": 1735660200,
    "nonce": "abc123",
    "idempotencyKey": "idem-001",
    "keyVersion": 1,
    "signature": "demo"
  }'
```

### Senaryo 2: Fraud Transaction (Ayşe - Londra'dan gece işlemi)
Expected: **DECLINED** (score > 0.85)

```powershell
curl -k -X POST https://localhost:8443/api/payments `
  -H "Content-Type: application/json" `
  -H "X-Merchant-Lat: 51.5074" `
  -H "X-Merchant-Long: -0.1278" `
  -H "X-Merchant-Category: electronics" `
  -d '{
    "terminalId": "TERM002",
    "traceId": "demo-fraud-001",
    "txnType": "AUTH",
    "amount": 50000.00,
    "currency": "TRY",
    "panToken": "tok_ayse_002",
    "timestamp": 1735614000,
    "nonce": "def456",
    "idempotencyKey": "idem-002",
    "keyVersion": 1,
    "signature": "demo"
  }'
```

### Senaryo 3: Edge Case (Mehmet - Taşınmış kullanıcı)
Expected: **PENDING** (score ~0.65)

```powershell
curl -k -X POST https://localhost:8443/api/payments `
  -H "Content-Type: application/json" `
  -H "X-Merchant-Lat: 39.9334" `
  -H "X-Merchant-Long: 32.8597" `
  -H "X-Merchant-Category: shopping" `
  -d '{
    "terminalId": "TERM003",
    "traceId": "demo-edge-001",
    "txnType": "AUTH",
    "amount": 5000.00,
    "currency": "TRY",
    "panToken": "tok_mehmet_003",
    "timestamp": 1735657200,
    "nonce": "ghi789",
    "idempotencyKey": "idem-003",
    "keyVersion": 1,
    "signature": "demo"
  }'
```

## Expected Response Format

```json
{
  "traceId": "demo-normal-001",
  "approved": true,
  "responseCode": "00",
  "authCode": "123456",
  "rrn": "uuid-here",
  "message": "APPROVED",
  "fraudScore": 0.05,
  "riskLevel": "MINIMAL",
  "fraudReasons": []
}
```

## Health Check Endpoints

```powershell
# Fraud API Health
curl http://localhost:8000/health

# Payment Server Health
curl -k https://localhost:8443/actuator/health
```
