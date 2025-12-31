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

5. **mTLS Client Certificate**
   The server requires client cert authentication (`server.ssl.client-auth: need`).
   Use the bundled client certificate when calling the API:
   ```
   scripts/certs/output/pos-client.p12
   ```
   Replace `<CLIENT_P12_PASSWORD>` below with the actual keystore password.

6. **HMAC Secret**
   The request signatures must use the same HMAC secret as the server:
   ```
   acquirer-server/src/main/resources/application.yml -> security.hmac.secret
   ```

## Demo Scenarios

> **Important:** `/api/*` requires:
> - `X-Terminal-Id`, `X-Nonce`, `X-Timestamp`, `X-Signature` headers
> - A body `signature` field calculated from the canonical request
> - mTLS client certificate

### Helper: Generate Signed Request + Headers
Run this Python snippet to generate a signed request and a ready-to-use `curl` command.
Update `SCENARIO` and `MERCHANT_*` variables for each demo.

```powershell
python - <<'PY'
import base64
import hmac
import hashlib
import json
import time
import os
import secrets

# === Config ===
HMAC_SECRET = "CHANGE_ME_DEV_SECRET_32+CHARS"
CLIENT_P12 = "scripts/certs/output/pos-client.p12"
CLIENT_P12_PASSWORD = "<CLIENT_P12_PASSWORD>"

# === Scenario data ===
SCENARIO = {
    "terminalId": "TERM001",
    "traceId": "demo-normal-001",
    "txnType": "AUTH",
    "amount": 150.00,
    "currency": "TRY",
    "panToken": "tok_ahmet_001",
    "timestamp": int(time.time()),
    "nonce": base64.urlsafe_b64encode(secrets.token_bytes(16)).decode().rstrip("="),
    "idempotencyKey": "idem-001",
    "keyVersion": 1,
}

MERCHANT_LAT = "40.9912"
MERCHANT_LONG = "29.0228"
MERCHANT_CATEGORY = "grocery"

# === Build canonical strings ===
def amount_to_string(v):
    return ("%.10f" % v).rstrip("0").rstrip(".")

canonical_body = (
    f"{SCENARIO['terminalId']}|{SCENARIO['traceId']}|{SCENARIO['txnType']}|"
    f"{amount_to_string(SCENARIO['amount'])}|{SCENARIO['currency']}|{SCENARIO['panToken']}|"
    f"{SCENARIO['timestamp']}|{SCENARIO['nonce']}|{SCENARIO['idempotencyKey']}|{SCENARIO['keyVersion']}"
)

body_sig = base64.urlsafe_b64encode(
    hmac.new(HMAC_SECRET.encode(), canonical_body.encode(), hashlib.sha256).digest()
).decode().rstrip("=")

SCENARIO["signature"] = body_sig
body_json = json.dumps(SCENARIO, separators=(",", ":"))

canonical_headers = f"{SCENARIO['terminalId']}|{SCENARIO['nonce']}|{SCENARIO['timestamp']}|{body_json}"
header_sig = base64.urlsafe_b64encode(
    hmac.new(HMAC_SECRET.encode(), canonical_headers.encode(), hashlib.sha256).digest()
).decode().rstrip("=")

curl_cmd = f"""
curl -k --cert-type P12 --cert {CLIENT_P12}:{CLIENT_P12_PASSWORD} \\
  -X POST https://localhost:8443/api/payments \\
  -H "Content-Type: application/json" \\
  -H "X-Terminal-Id: {SCENARIO['terminalId']}" \\
  -H "X-Nonce: {SCENARIO['nonce']}" \\
  -H "X-Timestamp: {SCENARIO['timestamp']}" \\
  -H "X-Signature: {header_sig}" \\
  -H "X-Merchant-Lat: {MERCHANT_LAT}" \\
  -H "X-Merchant-Long: {MERCHANT_LONG}" \\
  -H "X-Merchant-Category: {MERCHANT_CATEGORY}" \\
  -d '{body_json}'
"""

print(curl_cmd.strip())
PY
```

### Senaryo 1: Normal Transaction (Ahmet - Istanbul)
Expected: **APPROVED** (score < 0.3)

Use the helper snippet above with:
```
traceId: demo-normal-001
panToken: tok_ahmet_001
merchant: grocery @ (40.9912, 29.0228)
```

### Senaryo 2: Fraud Transaction (Ayşe - Londra'dan gece işlemi)
Expected: **DECLINED** (score > 0.85)

Use the helper snippet above with:
```
traceId: demo-fraud-001
panToken: tok_ayse_002
merchant: electronics @ (51.5074, -0.1278)
amount: 50000.00
```

### Senaryo 3: Edge Case (Mehmet - Taşınmış kullanıcı)
Expected: **PENDING** (score ~0.65)

Use the helper snippet above with:
```
traceId: demo-edge-001
panToken: tok_mehmet_003
merchant: shopping @ (39.9334, 32.8597)
amount: 5000.00
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
