# Proje Raporu: Güvenli POS Ödeme ve Sahtecilik Tespit Sistemi

**Tarih:** 31 Aralık 2024
**Hazırlayan:** Antigravity (AI Assistant)

## İçindekiler
1. [Proje Özeti](#1-proje-özeti)
2. [Sistem Mimarisi](#2-sistem-mimarisi)
3. [Teknoloji Yığını](#3-teknoloji-yığını)
4. [Modüllerin Detaylı Analizi](#4-modüllerin-detaylı-analizi)
5. [Veri Modeli ve Veritabanı](#5-veri-modeli-ve-veritabanı)
6. [İş Akışları (Workflows)](#6-iş-akışları)
7. [Sahtecilik Tespit Sistemi (Fraud Detection)](#7-sahtecilik-tespit-sistemi)
8. [Kurulum ve Çalıştırma](#8-kurulum-ve-çalıştırma)

---

## 1. Proje Özeti

Bu proje, güvenli bir POS (Point of Sale) ödeme kanalı simülasyonu ve entegre bir yapay zeka tabanlı sahtecilik tespit (fraud detection) sistemini içeren kapsamlı bir finansal teknoloji çözümüdür.

Sistem, POS terminallerinden gelen ödeme isteklerini karşılar, bu istekleri işler ve gerçek zamanlı olarak bir sahtecilik risk analizine tabi tutar. Analiz sonucunda işlem ya onaylanır (APPROVED), reddedilir (DECLINED) ya da incelemeye alınır. Amaç, modern ödeme sistemlerinin güvenliğini artırmak ve yapay zeka desteğiyle finansal kayıpları minimize etmektir.

---

## 2. Sistem Mimarisi

Proje, mikroservis mimarisine benzer modüler bir yapıda tasarlanmıştır. Temel olarak üç ana bileşenden oluşur:

1.  **Acquirer Server (Ödeme Sunucusu):** Sistemin beyni olan Java tabanlı backend uygulamasıdır. Ödeme isteklerini karşılar, veritabanı işlemlerini yönetir ve Fraud Agent ile iletişim kurar.
2.  **Fraud Agent (Sahtecilik Ajanı):** Python ile geliştirilmiş, makine öğrenmesi (ML) tabanlı risk analiz servisidir. Gelen işlem verilerini analiz ederek bir risk skoru üretir.
3.  **Veritabanı (PostgreSQL):** Kullanıcı profilleri, işlem geçmişi ve sistem loglarının tutulduğu ilişkisel veritabanıdır.

### Mimari Diyagramı (Kavramsal)

```mermaid
graph TD
    POS[POS Client / İstemci] -->|HTTPS POST /api/payments| AS[Acquirer Server (Java)]
    AS -->|Geçmiş Veri Sorgusu| DB[(PostgreSQL)]
    AS -->|Risk Analiz İsteği| FA[Fraud Agent (Python)]
    FA -->|ML Model Tahmini| FA
    FA -->|Risk Skoru & Karar| AS
    AS -->|Sonuç (Onay/Ret)| POS
```

---

## 3. Teknoloji Yığını

Projede endüstri standardı, güvenilir ve modern teknolojiler kullanılmıştır:

### Backend (Acquirer Server)
*   **Dil:** Java 17+
*   **Framework:** Spring Boot 3.5.9 (Web, Data JPA, Validation, Actuator)
*   **Build Tool:** Gradle (Kotlin DSL veya Groovy)
*   **Veritabanı Migrasyonu:** Flyway
*   **Testing:** JUnit 5

### Fraud Detection (Fraud Agent)
*   **Dil:** Python 3.10+
*   **Web Framework:** (Muhtemelen FastAPI veya Flask - `fraud_api.py` üzerinden çalışır)
*   **ML Kütüphaneleri:** XGBoost (Tahminleme), Pandas (Veri İşleme), Scikit-learn
*   **Model:** Eğitilmiş XGBoost sınıflandırma modeli

### Veri ve Altyapı
*   **Veritabanı:** PostgreSQL 15 (Docker üzerinde çalışır)
*   **Konteynerizasyon:** Docker (Veritabanı servisi için)

---

## 4. Modüllerin Detaylı Analizi

### 4.1. Acquirer Server (`acquirer-server`)
Ödeme ekosisteminin merkezidir.
*   **PaymentController:** `/api/payments` endpoint'i üzerinden gelen JSON formatındaki ödeme isteklerini karşılar.
*   **Domain Modelleri:** `UserProfile` (Kullanıcı demografik verileri) ve `TransactionHistory` (Geçmiş işlemler) varlıklarını yönetir.
*   **Entegrasyon:** `FraudServiceClient` (varsayımsal isimlendirme) aracılığıyla Python servisine HTTP çağrıları yapar.

### 4.2. Fraud Agent (`fraudAgent`)
Analitik zekayı barındırır.
*   **API:** HTTP üzerinden veri alır.
*   **Özellik Çıkarımı (Feature Engineering):** Ham işlem verisinden (tutar, zaman, konum) modelin anlayacağı öznitelikleri (örneğin: "son 1 saatteki işlem sayısı", "ortalama tutardan sapma oranı") türetir.
*   **Karar Mekanizması:** 0 ile 1 arasında bir skor üretir. (Örn: >0.85 ise Ret).

### 4.3. Common (`common`)
*   Projeler arası paylaşılan veri transfer objeleri (DTO), sabitler ve yardımcı sınıfları içerir. Bu sayede kod tekrarı önlenir.

---

## 5. Veri Modeli ve Veritabanı

Veritabanı şeması, sahtecilik tespiti için kritik olan tarihsel veriyi saklayacak şekilde optimize edilmiştir.

### Tablo: `user_profiles`
Kullanıcıların davranışsal profillerini saklar.
*   `id`: Benzersiz kayıt ID'si.
*   `pan_token`: Kredi kartı numarasının tokenize edilmiş hali (Güvenlik için).
*   `avg_amount`: Kullanıcının ortalama harcama tutarı.
*   `access_locations`: Sık kullanılan lokasyonlar.

### Tablo: `transaction_history`
Yapılan her işlemin kaydını tutar. ML modelinin "geçmiş davranışları" öğrenmesi için bu tablo kritiktir.
*   `amount`: İşlem tutarı.
*   `merchant_category`: Harcama yapılan kategori (Market, Elektronik vb.).
*   `is_fraud`: İşlemin fraud olup olmadığı (Eğitim verisi için etiket).
*   `home_lat`, `home_long`: Kullanıcının ev adresi koordinatları.

---

## 6. İş Akışları

### 6.1. Ödeme İşlem Akışı (Happy Path)
1.  **İstek Başlatma:** POS terminali veya istemci, şifreli bir ödeme isteği oluşturur.
2.  **Doğrulama:** Acquirer Server, isteğin imzasını (signature), zaman damgasını ve formatını doğrular.
3.  **Profil Yükleme:** İşlemi yapan kartın (`panToken`) geçmiş profili veritabanından çekilir.
4.  **Risk Analizi:**
    *   Mevcut işlem verisi + Kullanıcı Profili verisi paketlenir.
    *   Python Fraud API'ye gönderilir.
5.  **Karar:**
    *   ML modeli bir skor (örn: 0.05) döndürür.
    *   Skor eşik değerin altındaysa (örn: < 0.30), işlem **ONAYLANIR**.
6.  **Kayıt:** İşlem sonucu veritabanına kaydedilir ve kullanıcının profili (ortalama harcama vb.) güncellenir.
7.  **Yanıt:** İstemciye "APPROVED" yanıtı dönülür.

### 6.2. Sahtecilik Yakalama Senaryosu
*   Eğer kullanıcı normalde İstanbul'da harcama yapıyorken, aniden Londra'dan yüksek tutarlı bir elektronik harcaması gelirse:
    1.  Konum farkı (Distance feature) yüksek çıkar.
    2.  Tutar sapması (Amount deviation) yüksek çıkar.
    3.  ML modeli yüksek bir risk skoru (örn: 0.95) üretir.
    4.  Sistem işlemi otomatik olarak **REDDEDER (DECLINED)**.

---

## 7. Sahtecilik Tespit Sistemi (Fraud Detection)

Bu modül, basit kurallar (if-else) yerine istatistiksel modeller kullanır.

*   **Algoritma:** XGBoost (Extreme Gradient Boosting). Hızlı ve yüksek başarımlı bir ağaç tabanlı modeldir.
*   **Girdiler (Features):**
    *   İşlem Tutarı
    *   Kullanıcı Yaşı
    *   Mesafe (Kullanıcı evi ile mağaza arası)
    *   Zaman (Gece yarısı yapılan işlemler daha risklidir)
    *   Kategori Risk Faktörü

---

## 8. Kurulum ve Çalıştırma

Sistemi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyin.

### 8.1. Ön Gereksinimler
*   Java 17 veya üzeri JDK
*   Python 3.10+
*   Docker Desktop (PostgreSQL için)
*   Git

### 8.2. Adım Adım Kurulum

**1. Veritabanını Başlatın:**
```bash
docker run -d --name pos-postgres -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=pos_payment postgres:15
```

**2. Fraud Agent'ı (Python) Çalıştırın:**
```bash
cd fraudAgent
python -m venv .venv
.\.venv\Scripts\Activate  # Windows için
pip install -r requirements.txt
python fraud_api.py
```
*(Bu servis 8000 portunda çalışacaktır)*

**3. Acquirer Server'ı (Java) Çalıştırın:**
Yeni bir terminal açın ve:
```bash
cd acquirer-server
.\gradlew bootRun
```
*(Bu servis 8443 portunda çalışacaktır)*

**4. Test Etme:**
`DEMO.md` dosyasındaki örnek `curl` komutlarını kullanarak sisteme istek gönderebilir ve sonuçları gözlemleyebilirsiniz.

---

## Sonuç

Secure POS Payment Channel projesi, güvenli ve zeki ödeme sistemlerinin nasıl kurgulanabileceğine dair modern bir örnektir. Mikroservis mimarisi, veri odaklı karar verme mekanizmaları ve güçlü teknoloji altyapısı ile ölçeklenebilir bir temel sunar.
