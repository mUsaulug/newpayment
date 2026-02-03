# Güvenli POS Ödeme Kanalı + Sahtecilik (Fraud) Tespit Sistemi

Bu depo; POS tarafındaki isteklerin güvenli şekilde imzalanıp acquirer sunucusuna gönderildiği, sahtecilik (fraud) skorlaması ile karar üretildiği ve sonuçların hem UI hem de veri tabanına yansıtıldığı uçtan uca bir demo sistemini içerir. Sistem 4 ana bileşenden oluşur: **frontend**, **pos-client**, **acquirer-server** ve **fraudAgent**. Ayrıca ortak DTO’lar ve güvenlik yardımcıları **common** modülünde paylaşılır.

## Mimari Özet

```
[Frontend (Vite/React)]
        |
        |  /api/pos/payments  (HTTP)
        v
[POS Client (Spring Boot)] --mTLS + HMAC--> [Acquirer Server (Spring Boot)]
        |                                         |
        |  /pos-client/stream (SSE)               |  /api/payments
        v                                         v
  UI canlı akış                            [Fraud Agent (FastAPI)]
                                                |
                                                v
                                          XGBoost Model
```

## Bileşenler

### 1) `acquirer-server`
* `/api/payments` endpoint’i ödeme taleplerini alır, güvenlik doğrulamalarını uygular ve fraud skorlaması ile karar üretir. `traceId` veya `idempotencyKey` çakışmasında **409** döner. Ayrıca `/api/echo` test endpoint’i ve `/ping` sağlık endpoint’i vardır.
* mTLS ve HMAC kontrollü bir güvenlik katmanı kullanır. Header doğrulamaları filtre seviyesinde yapılır.
* Fraud kararlarını `FraudDetectionService` üzerinden üretir ve veritabanına kaydeder.

### 2) `pos-client`
* Frontend’den gelen ödeme isteklerini alır, HMAC imzalarını hesaplayıp acquirer’a iletir.
* Canlı senaryo akışı için SSE endpoint’i sağlar.
* mTLS için client sertifikasını ve truststore’u yükleyerek acquirer’a güvenli bağlantı kurar.

### 3) `fraudAgent`
* FastAPI üzerinde çalışan bir ML servisidir. `/predict` endpoint’inde XGBoost modelini çağırır; model yoksa kurallı fallback uygular.
* Model dosyası `fraudAgent/models/` altında bulunur ve uygulama başlangıcında yüklenir.

### 4) `frontend`
* React + Vite tabanlı UI, POS simülasyonunu ve fraud karar akışını görselleştirir. Ödemeler `pos-client` üzerinden yapılır ve SSE ile akış izlenir.

### 5) `common`
* Ortak DTO’lar (`PaymentRequest`, `PaymentResponse`) ve HMAC/nonce yardımcıları bu modülde bulunur.

## Güvenlik Katmanı (Özet)

* **mTLS:** POS client → acquirer arasındaki bağlantı karşılıklı sertifika doğrulaması ile korunur. Acquirer HTTPS üzerinde client-auth zorunlu kılar.
* **HMAC İmzaları:** `PaymentRequest` gövdesi ve header’lar ayrı canonical formatlarda imzalanır. İmzalama ve doğrulama logic’i ortak modülde tutulur.
* **Replay Koruması:** `nonce` formatı doğrulanır ve tekrar kullanım engellenir.

## Veri Modeli (Özet)

* `user_profiles` kullanıcı davranış metriklerini saklar.
* `transaction_history` karar ve fraud skorlarını saklar; `idempotency_key` desteği mevcuttur.

## Kurulum (Özet)

> Aşağıdaki adımlar geliştirme ortamı içindir. `DEMO.md` dosyasında ayrıntılı demo akışı bulunur.

### 1) PostgreSQL
```bash
docker run -d --name pos-postgres -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=pos_payment postgres:15
```

### 2) Fraud API
```bash
cd fraudAgent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python fraud_api.py
```

### 3) Acquirer Server
```bash
cd acquirer-server
./gradlew bootRun
```

### 4) POS Client
```bash
cd pos-client
./gradlew bootRun
```

### 5) Frontend
```bash
cd frontend
npm install
npm run dev
```

## Notlar

* Uygulama, HMAC secret’ının acquirer ve pos-client tarafında aynı olmasını bekler.
* Demo senaryoları ve UI akışı için `demo_scenarios.json` dosyasını inceleyebilirsiniz.

---

Detaylı inceleme ve mimari rapor için: **PROJE_RAPORU.md**
