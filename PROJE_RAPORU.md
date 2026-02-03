# Proje Raporu: Güvenli POS Ödeme Kanalı ve Fraud Tespit Sistemi

**Kapsam:** Bu rapor, `/workspace/newpayment` reposundaki tüm bileşenleri uçtan uca inceler; mimari akış, güvenlik, veri modeli, fraud algoritması ve operasyonel çalıştırma adımlarını detaylandırır.

## 1) Yüksek Seviye Özet

Bu proje, POS tarafında oluşturulan ödeme isteklerinin güvenli şekilde imzalanıp acquirer sunucusuna iletildiği, fraud modelinden skor alındığı ve kararın hem UI hem de veritabanına işlendiği bir demo altyapısıdır. Temel bileşenler:

* **Frontend (React/Vite):** Ödeme senaryolarını, güvenlik aşamalarını ve fraud kararlarını görselleştirir.
* **POS Client (Spring Boot):** Frontend’den gelen istekleri HMAC imzası ile acquirer’a iletir ve canlı akış yayınlar.
* **Acquirer Server (Spring Boot):** Ödeme doğrulaması, fraud değerlendirmesi ve veritabanı kayıtlarını yönetir.
* **Fraud Agent (FastAPI):** XGBoost modeline dayalı fraud skoru üretir; model yoksa kurallı fallback uygular.
* **Common Modül:** Ortak DTO’lar ve güvenlik yardımcıları.

## 2) Depo Yapısı ve Modüller

```
acquirer-server/     # Spring Boot acquirer + fraud orchestration + DB
pos-client/          # POS proxy + mTLS + SSE stream
fraudAgent/          # FastAPI + XGBoost model servisi
frontend/            # Vite/React UI
common/              # Ortak DTO + güvenlik yardımcıları
scripts/certs/       # mTLS sertifikaları (p12, crt, key)
```

## 3) Uçtan Uca İş Akışı

Aşağıdaki akış ödeme isteğinin sistem içinde nasıl işlendiğini gösterir:

1. **Frontend → POS Client:** UI `/api/pos/payments` endpoint’ine istek gönderir (frontend doğrudan acquirer’a çağrı yapmaz).
2. **POS Client → Acquirer:** POS client, `PaymentRequest` oluşturur; gövde ve header imzalarını üretir, mTLS ile acquirer’a gönderir.
3. **Acquirer Güvenlik Katmanı:** Security filter header’ları doğrular; `RequestSecurityService` ise gövde imzası, timestamp ve nonce kontrollerini uygular.
4. **Fraud Değerlendirme:** `FraudDetectionService` profil + işlem geçmişi + merchant bilgilerini alır, feature’ları çıkarır ve Fraud Agent’a gönderir.
5. **Karar ve Kayıt:** Skor threshold’larına göre APPROVED/PENDING/DECLINED kararı çıkar; `transaction_history` tablosuna kayıt atılır, gerekiyorsa `user_profiles` güncellenir.
6. **POS Client → Frontend:** Yanıt UI’ya iletilir, ayrıca SSE ile canlı senaryo akışı yayınlanır.

## 4) Güvenlik Mimarisinin Detayları

### 4.1 mTLS (Karşılıklı Sertifika)

* Acquirer server HTTPS üzerinde çalışır ve client-auth zorunludur.
* POS client, `pos-client.p12` ile acquirer’a bağlanır ve truststore üzerinden doğrulama yapar.

### 4.2 HMAC İmza Yapısı

* Gövde imzası: `terminalId|traceId|txnType|amount|currency|panToken|timestamp|nonce|idempotencyKey|keyVersion` formatında üretilir.
* Header imzası: `terminalId|nonce|timestamp|body` formatında üretilir.
* İmzalar `HMAC-SHA256` ile Base64 URL-safe formatında üretilir ve doğrulanır.

### 4.3 Replay / Timestamp Kontrolleri

* `nonce` formatı regex ile doğrulanır; timestamp ile skew kontrolü yapılır.
* Acquirer tarafında nonce tekrar kullanımına karşı `NonceStore` kullanılır.

## 5) Fraud Algoritması ve Feature Engineering

### 5.1 Feature Üretimi

`FraudFeatureExtractor` toplamda 35 feature üretir. Başlıca gruplar:

* **Zaman feature’ları:** `hour`, `dayOfWeek`, `isNight`, `isWeekend`.
* **Tutar feature’ları:** `amt`, `amtLog`, `amtZscore`.
* **Lokasyon feature’ları:** `distanceKm`, `distanceLog`, `cityPopLog`.
* **Davranış feature’ları:** `cardTxCount`, `timeSinceLastTx`, `amtRollingMean3` vb.
* **Kategori encoding:** `categoryEncoded`, `genderEncoded`, `stateEncoded`.

### 5.2 Fraud Karar Eşikleri

* **DECLINED:** skor ≥ 0.85
* **PENDING:** skor ≥ 0.65
* **APPROVED:** diğer durumlar

Karar mantığı `FraudDetectionService` içinde uygulanır.

### 5.3 Fraud Agent (FastAPI + XGBoost)

* `/predict` endpoint’inde model `xgboost_fraud_model_latest.pkl` ile skor üretilir.
* Model yüklenemezse rule-based fallback devreye girer.

## 6) Veri Modeli (PostgreSQL)

### 6.1 `user_profiles`
* `pan_token`: kart kimliği
* `avg_amount`, `transaction_count`: davranış istatistikleri
* `home_lat`, `home_long`: kullanıcı konumu

### 6.2 `transaction_history`
* Fraud skoru, risk seviyesi, karar ve merchant bilgisi saklanır.
* `idempotency_key` ile tekrarlı istekler tespit edilir.

### 6.3 Demo Veri

`V2__demo_data.sql` içinde örnek kullanıcılar ve işlem geçmişi bulunur.

## 7) API Envanteri

### 7.1 Acquirer Server
| Endpoint | Method | Açıklama |
| --- | --- | --- |
| `/api/payments` | POST | Ödeme isteğini alır, fraud değerlendirme yapar. | 
| `/api/echo` | POST | Text echo; güvenlik filtrelerinden geçer. |
| `/ping` | GET | Basit sağlık endpoint’i. |

### 7.2 POS Client
| Endpoint | Method | Açıklama |
| --- | --- | --- |
| `/api/pos/payments` | POST | Frontend’den gelen isteği acquirer’a iletir. |
| `/pos-client/stream` | GET | SSE üzerinden canlı senaryo akışı. |

### 7.3 Fraud Agent
| Endpoint | Method | Açıklama |
| --- | --- | --- |
| `/predict` | POST | Fraud skoru üretir. |
| `/health` | GET | Model durumunu döndürür. |

## 8) Konfigürasyonlar

* **Acquirer:** `server.port=8443`, mTLS zorunlu, HMAC secret `security.hmac.secret`.
* **POS Client:** acquirer base URL `https://localhost:8443`, HMAC secret `security.hmac.secret`.
* **Fraud Agent:** Varsayılan `http://localhost:8000`.

## 9) Çalıştırma Senaryosu (Özet)

Detaylı demo akışı `DEMO.md` dosyasında yer alır.

1. PostgreSQL başlatılır.
2. Fraud Agent `python fraud_api.py` ile ayağa kaldırılır.
3. Acquirer `./gradlew bootRun` ile çalıştırılır.
4. POS client `./gradlew bootRun` ile çalıştırılır.
5. Frontend `npm run dev` ile açılır.

## 10) UI Senaryo Akışı

`demo_scenarios.json` dosyasında UI için önceden tanımlanmış senaryolar ve state machine bilgisi bulunur.

---

Bu rapor; kod tabanı, konfigürasyonlar ve demo verilerinin tamamı üzerinden oluşturulmuştur.
