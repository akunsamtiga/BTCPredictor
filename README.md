# 🪙 Bitcoin Price Predictor - Automated VPS Version

Sistem prediksi harga Bitcoin otomatis dengan Machine Learning (LSTM + Random Forest + Gradient Boosting) yang terintegrasi dengan Firebase untuk deployment di VPS.

## 🌟 Fitur Utama

- ✅ **Prediksi Multi-Timeframe**: 15 menit, 30 menit, 1 jam, 4 jam, 12 jam, 24 jam
- ✅ **Machine Learning**: Ensemble LSTM, Random Forest, Gradient Boosting
- ✅ **Real-time Data**: Pengambilan data Bitcoin secara kontinyu
- ✅ **Firebase Integration**: Penyimpanan prediksi dan validasi hasil
- ✅ **Automated Validation**: Tracking akurasi (WIN/LOSE) otomatis
- ✅ **Statistics Dashboard**: Win rate dan performa model
- ✅ **Auto-Retraining**: Model dilatih ulang secara otomatis
- ✅ **VPS Ready**: Siap di-deploy di VPS dengan systemd

## 📁 Struktur File

```
btc-predictor/
├── config.py                      # Konfigurasi sistem
├── firebase_manager.py            # Firebase operations
├── btc_predictor_automated.py     # ML Predictor (Part 1 & 2)
├── scheduler.py                   # Automation scheduler
├── requirements.txt               # Dependencies
├── firebase-credentials.json      # Firebase credentials (tidak di-commit)
├── models/                        # Saved ML models
│   ├── lstm_model.keras
│   ├── rf_model.pkl
│   ├── gb_model.pkl
│   └── scalers.pkl
└── logs/
    └── btc_predictor_automation.log
```

## 🚀 Setup Instructions

### 1. Persiapan VPS

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+
sudo apt install python3 python3-pip python3-venv -y

# Install dependencies sistem
sudo apt install build-essential libssl-dev libffi-dev python3-dev -y
```

### 2. Clone & Setup Project

```bash
# Buat direktori project
mkdir ~/btc-predictor
cd ~/btc-predictor

# Buat virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Setup Firebase

#### a. Buat Firebase Project
1. Pergi ke [Firebase Console](https://console.firebase.google.com/)
2. Buat project baru
3. Enable **Firestore Database**
4. Enable **Realtime Database** (optional)

#### b. Download Credentials
1. Project Settings → Service Accounts
2. Generate new private key
3. Download JSON file
4. Rename menjadi `firebase-credentials.json`
5. Upload ke VPS: `~/btc-predictor/firebase-credentials.json`

#### c. Update config.py
```python
FIREBASE_CONFIG = {
    'credentials_path': 'firebase-credentials.json',
    'database_url': 'https://YOUR-PROJECT-ID.firebaseio.com'  # Ganti dengan project ID Anda
}
```

### 4. Konfigurasi (Optional)

Edit `config.py` untuk menyesuaikan:

```python
# Timeframe prediksi
PREDICTION_CONFIG = {
    'timeframes': [15, 30, 60, 240, 720, 1440],  # dalam menit
    'prediction_interval': 300,  # Prediksi setiap 5 menit
    'validation_check_interval': 60,  # Validasi setiap 1 menit
}

# Model configuration
MODEL_CONFIG = {
    'auto_retrain_interval': 86400,  # Retrain setiap 24 jam
}
```

## 🎯 Menjalankan Predictor

### Mode Manual (Testing)

```bash
# Aktifkan virtual environment
source ~/btc-predictor/venv/bin/activate

# Jalankan scheduler
python scheduler.py
```

### Mode Otomatis dengan systemd (Recommended)

#### 1. Buat Service File

```bash
sudo nano /etc/systemd/system/btc-predictor.service
```

#### 2. Paste Konfigurasi Berikut:

```ini
[Unit]
Description=Bitcoin Price Predictor Automation
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/btc-predictor
Environment="PATH=/home/YOUR_USERNAME/btc-predictor/venv/bin"
ExecStart=/home/YOUR_USERNAME/btc-predictor/venv/bin/python scheduler.py
Restart=always
RestartSec=10
StandardOutput=append:/home/YOUR_USERNAME/btc-predictor/logs/output.log
StandardError=append:/home/YOUR_USERNAME/btc-predictor/logs/error.log

[Install]
WantedBy=multi-user.target
```

**Ganti:**
- `YOUR_USERNAME` dengan username VPS Anda
- Path sesuai lokasi instalasi

#### 3. Enable & Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (auto-start on boot)
sudo systemctl enable btc-predictor

# Start service
sudo systemctl start btc-predictor

# Check status
sudo systemctl status btc-predictor
```

#### 4. Command Berguna

```bash
# Stop service
sudo systemctl stop btc-predictor

# Restart service
sudo systemctl restart btc-predictor

# View logs
tail -f ~/btc-predictor/logs/btc_predictor_automation.log

# View systemd logs
sudo journalctl -u btc-predictor -f
```

## 📊 Monitoring & Logs

### Check Logs

```bash
# Real-time log monitoring
tail -f ~/btc-predictor/logs/btc_predictor_automation.log

# View last 100 lines
tail -n 100 ~/btc-predictor/logs/btc_predictor_automation.log

# Search for errors
grep "ERROR" ~/btc-predictor/logs/btc_predictor_automation.log
```

### Firebase Console

1. **Predictions Collection**: Lihat semua prediksi
2. **Validation Collection**: Hasil validasi (WIN/LOSE)
3. **Statistics Collection**: Win rate dan performa
4. **Model Performance**: Metrics model ML

## 🔍 Cara Kerja Sistem

### 1. Data Fetching (Real-time)
- Mengambil data Bitcoin dari CryptoCompare API
- Update setiap 60 detik
- Cache data untuk efisiensi

### 2. Prediction Cycle (Setiap 5 Menit)
```
Fetch Data → Add Indicators → ML Prediction → Save to Firebase
```

Untuk setiap timeframe:
- **15 min, 30 min, 1 jam, 4 jam, 12 jam, 24 jam**
- Menggunakan ensemble 3 model (LSTM, RF, GB)
- Confidence score berdasarkan agreement model

### 3. Validation Cycle (Setiap 1 Menit)
```
Get Unvalidated Predictions → Check Target Time → Compare with Actual Price → WIN/LOSE
```

Win Condition:
- Jika prediksi CALL (Bullish) dan harga naik = **WIN** ✅
- Jika prediksi PUT (Bearish) dan harga turun = **WIN** ✅
- Selain itu = **LOSE** ❌

### 4. Auto-Retraining (Setiap 24 Jam)
- Model dilatih ulang dengan data terbaru
- Metrics disimpan ke Firebase
- Model lama di-backup otomatis

## 📈 Struktur Data di Firebase

### Predictions Collection
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "timeframe_minutes": 60,
  "current_price": 45000.00,
  "predicted_price": 45500.00,
  "price_change": 500.00,
  "price_change_pct": 1.11,
  "trend": "CALL (Bullish)",
  "confidence": 75.5,
  "target_time": "2024-01-15T11:30:00",
  "validated": false,
  "validation_result": null
}
```

### Validation Collection (Setelah Validasi)
```json
{
  "timestamp": "2024-01-15T11:30:00",
  "prediction_id": "doc_id_xyz",
  "result": "WIN",
  "predicted_price": 45500.00,
  "actual_price": 45800.00,
  "error": 300.00,
  "error_pct": 0.66
}
```

### Statistics Collection
```json
{
  "timeframe_minutes": 60,
  "period_days": 7,
  "total_predictions": 100,
  "wins": 65,
  "losses": 35,
  "win_rate": 65.0,
  "avg_error": 250.50,
  "avg_error_pct": 0.55,
  "last_updated": "2024-01-15T12:00:00"
}
```

## 🛠️ Troubleshooting

### Problem: Service tidak start

```bash
# Check logs
sudo journalctl -u btc-predictor -n 50

# Check permissions
ls -la ~/btc-predictor/firebase-credentials.json

# Test manual
source ~/btc-predictor/venv/bin/activate
python scheduler.py
```

### Problem: Firebase connection error

```bash
# Verify credentials
cat ~/btc-predictor/firebase-credentials.json

# Test connection
python -c "import firebase_admin; print('Firebase OK')"
```

### Problem: Model training gagal

```bash
# Check disk space
df -h

# Check memory
free -h

# Check TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Problem: API rate limit

- Daftar CryptoCompare API key gratis
- Update di `config.py`:
```python
DATA_CONFIG = {
    'cryptocompare_api_key': 'YOUR_API_KEY_HERE',
}
```

## 🔄 Update & Maintenance

### Update Code

```bash
cd ~/btc-predictor
source venv/bin/activate

# Pull updates (if using git)
git pull

# Restart service
sudo systemctl restart btc-predictor
```

### Manual Retraining

```bash
cd ~/btc-predictor
source venv/bin/activate

python -c "
from scheduler import PredictionScheduler
scheduler = PredictionScheduler()
scheduler.train_models()
"
```

### Cleanup Old Data

```bash
python -c "
from firebase_manager import FirebaseManager
firebase = FirebaseManager()
firebase.cleanup_old_data(days=30)
"
```

## 📊 Viewing Results

### Via Firebase Console
1. Buka [Firebase Console](https://console.firebase.google.com/)
2. Pilih project Anda
3. Firestore Database → Collections

### Via Python Script

```python
from firebase_manager import FirebaseManager

firebase = FirebaseManager()

# Get statistics
stats = firebase.get_statistics(days=7)
print(f"Win Rate: {stats['win_rate']}%")
print(f"Total: {stats['total_predictions']}")
print(f"Wins: {stats['wins']}, Losses: {stats['losses']}")
```

## ⚠️ Important Notes

1. **API Rate Limits**: CryptoCompare free tier memiliki batasan. Pertimbangkan upgrade atau gunakan API key.

2. **VPS Resources**: 
   - Minimum: 2GB RAM, 1 vCPU
   - Recommended: 4GB RAM, 2 vCPU

3. **Firebase Limits**: 
   - Free tier: 20K writes/day, 50K reads/day
   - Monitor usage di Firebase Console

4. **Backup Models**:
   ```bash
   # Backup models sebelum retrain
   cp -r models/ models_backup_$(date +%Y%m%d)/
   ```

5. **Monitor Disk Space**:
   ```bash
   # Setup log rotation
   sudo nano /etc/logrotate.d/btc-predictor
   ```

## 📞 Support

Jika ada masalah:
1. Check logs terlebih dahulu
2. Verify Firebase credentials
3. Test API connectivity
4. Check system resources

## 📝 License

Private use only. Gunakan dengan bijak dan sesuai regulasi trading di wilayah Anda.

---

**⚠️ DISCLAIMER**: Prediksi ini untuk referensi saja. Cryptocurrency trading berisiko tinggi. Selalu lakukan riset sendiri dan gunakan risk management yang baik. Tidak ada jaminan profit dari sistem ini.