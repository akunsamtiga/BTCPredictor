# 🚀 Quick Start Guide - Bitcoin Predictor VPS

Panduan cepat untuk menjalankan Bitcoin Predictor di VPS Anda dalam 10 menit!

## ✅ Prerequisites

Sebelum mulai, pastikan Anda memiliki:
- ✓ VPS dengan Ubuntu 20.04+ atau Debian 11+ (minimal 2GB RAM)
- ✓ SSH access ke VPS
- ✓ Firebase project sudah dibuat

## 📋 Step-by-Step Installation

### Step 1: Login ke VPS

```bash
ssh username@your-vps-ip
```

### Step 2: Download Project Files

```bash
# Buat direktori
mkdir -p ~/btc-predictor
cd ~/btc-predictor

# Upload semua file Python ke direktori ini:
# - config.py
# - firebase_manager.py
# - btc_predictor_automated.py (Part 1 & 2 digabung)
# - scheduler.py
# - requirements.txt
# - setup.sh
```

**Upload cara termudah:**
```bash
# Dari komputer lokal
scp *.py requirements.txt setup.sh username@your-vps-ip:~/btc-predictor/
```

### Step 3: Setup Firebase

#### A. Download Firebase Credentials

1. Buka [Firebase Console](https://console.firebase.google.com/)
2. Pilih/buat project Anda
3. ⚙️ Project Settings → Service Accounts
4. Click **Generate new private key**
5. Save file sebagai `firebase-credentials.json`

#### B. Upload ke VPS

```bash
# Dari komputer lokal
scp firebase-credentials.json username@your-vps-ip:~/btc-predictor/
```

#### C. Update config.py

```bash
nano ~/btc-predictor/config.py
```

Ubah baris ini:
```python
FIREBASE_CONFIG = {
    'credentials_path': 'firebase-credentials.json',
    'database_url': 'https://YOUR-PROJECT-ID.firebaseio.com'  # ← Ganti ini!
}
```

Ganti `YOUR-PROJECT-ID` dengan ID project Firebase Anda.

**Cara cek Project ID:**
- Di Firebase Console, lihat di bagian atas atau Project Settings

### Step 4: Gabungkan File btc_predictor_automated.py

File predictor ada 2 part. Gabungkan jadi 1 file:

```bash
cd ~/btc-predictor

# Buat file baru
cat > btc_predictor_automated.py << 'EOF'
# Copy SEMUA isi dari Part 1
# Kemudian lanjut dengan SEMUA isi dari Part 2 (tanpa komentar pemisah)
EOF
```

Atau edit manual:
```bash
nano btc_predictor_automated.py
```

Paste Part 1, lalu Part 2 (hapus line pemisah "# ... Continued in Part 2")

### Step 5: Run Setup Script

```bash
cd ~/btc-predictor
chmod +x setup.sh
./setup.sh
```

Setup script akan:
- ✓ Install dependencies
- ✓ Create virtual environment
- ✓ Setup directories
- ✓ Verify Firebase credentials
- ✓ Test installation

**Follow the prompts!**

### Step 6: Enable Firestore Database

⚠️ **PENTING!** Aktifkan Firestore di Firebase Console:

1. Firebase Console → Build → Firestore Database
2. Click **Create database**
3. Pilih **Production mode**
4. Pilih location (asia-southeast2 untuk Indonesia)
5. Click **Enable**

### Step 7: Test Run

```bash
cd ~/btc-predictor
source venv/bin/activate
python scheduler.py
```

Anda akan melihat:
```
================================================================================
🚀 STARTING BITCOIN PREDICTOR AUTOMATION
================================================================================
🔧 Initializing models...
📚 No existing models found, training new ones...
...
✅ Training completed successfully!
...
🎯 Running initial predictions...
💰 Current BTC Price: $45,123.45
...
✅ Prediction saved: doc_xyz123
```

**Biarkan berjalan 1-2 menit untuk training awal.**

Press `Ctrl+C` untuk stop.

### Step 8: Setup Auto-Start (Systemd)

```bash
# Enable service
sudo systemctl enable btc-predictor

# Start service
sudo systemctl start btc-predictor

# Check status
sudo systemctl status btc-predictor
```

Jika berhasil, Anda akan melihat:
```
● btc-predictor.service - Bitcoin Price Predictor
   Active: active (running)
```

## 📊 Monitoring

### Check Logs Real-time

```bash
# Application logs
tail -f ~/btc-predictor/logs/btc_predictor_automation.log

# System logs
sudo journalctl -u btc-predictor -f
```

### Check Firebase

1. Buka [Firebase Console](https://console.firebase.google.com/)
2. Pilih project Anda
3. Firestore Database

Anda akan melihat collections:
- **bitcoin_predictions** - Semua prediksi
- **prediction_validation** - Hasil validasi
- **prediction_statistics** - Win rate & statistik
- **bitcoin_data** - Data Bitcoin terbaru

## 🎯 Expected Output

Setelah 15-30 menit, Anda akan melihat:

### Di Logs:
```
🔮 RUNNING PREDICTIONS - 2024-01-15 10:30:00
💰 Current BTC Price: $45,000.00

⏱️  Predicting for 15 minutes...
   🟢 ↗️ $45,250.00 (+0.56%) - Confidence: 72.3%
✅ Prediction saved: abc123

⏱️  Predicting for 60 minutes...
   🔴 ↘️ $44,800.00 (-0.44%) - Confidence: 68.5%
✅ Prediction saved: def456
...
```

### Di Firebase:
- Predictions dengan status `validated: false`
- Setelah waktu target tercapai → validasi otomatis
- Result: `WIN` atau `LOSE`

## 🔧 Common Commands

```bash
# Start service
sudo systemctl start btc-predictor

# Stop service
sudo systemctl stop btc-predictor

# Restart service
sudo systemctl restart btc-predictor

# Check status
sudo systemctl status btc-predictor

# View logs
tail -f ~/btc-predictor/logs/btc_predictor_automation.log

# Check statistics
cd ~/btc-predictor
source venv/bin/activate
python -c "
from firebase_manager import FirebaseManager
fb = FirebaseManager()
stats = fb.get_statistics(days=7)
print(f'Win Rate: {stats[\"win_rate\"]}%')
print(f'Total: {stats[\"total_predictions\"]}')
"
```

## ⚠️ Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'tensorflow'"

```bash
cd ~/btc-predictor
source venv/bin/activate
pip install -r requirements.txt
```

### Problem: "Firebase credentials not found"

```bash
# Check file exists
ls -la ~/btc-predictor/firebase-credentials.json

# Re-upload if missing
# From local machine:
scp firebase-credentials.json username@vps-ip:~/btc-predictor/
```

### Problem: Service won't start

```bash
# Check logs
sudo journalctl -u btc-predictor -n 50

# Test manually
cd ~/btc-predictor
source venv/bin/activate
python scheduler.py
```

### Problem: "Rate limit exceeded"

Ini normal untuk free tier CryptoCompare API.

**Solusi:**
1. Daftar API key gratis: https://www.cryptocompare.com/cryptopian/api-keys
2. Update `config.py`:
```python
DATA_CONFIG = {
    'cryptocompare_api_key': 'YOUR_API_KEY_HERE',
}
```

## 📈 Viewing Results

### Method 1: Firebase Console (Easiest)

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Your Project → Firestore Database
3. Browse collections: `bitcoin_predictions`, `prediction_statistics`

### Method 2: Python Script

```bash
cd ~/btc-predictor
source venv/bin/activate

python << 'EOF'
from firebase_manager import FirebaseManager

fb = FirebaseManager()

# Get overall stats
stats = fb.get_statistics(days=7)
print("\n📊 Overall Statistics (Last 7 Days)")
print(f"Total Predictions: {stats['total_predictions']}")
print(f"Wins: {stats['wins']}")
print(f"Losses: {stats['losses']}")
print(f"Win Rate: {stats['win_rate']}%")
print(f"Avg Error: ${stats['avg_error']:.2f}")

# Get per-timeframe stats
for tf in [15, 30, 60, 240, 720, 1440]:
    tf_stats = fb.get_statistics(timeframe_minutes=tf, days=7)
    if tf_stats['total_predictions'] > 0:
        print(f"\n⏱️  {tf} minutes:")
        print(f"   Win Rate: {tf_stats['win_rate']:.1f}% ({tf_stats['wins']}/{tf_stats['total_predictions']})")
EOF
```

## 🎉 Success Checklist

- ✅ Service running: `sudo systemctl status btc-predictor` shows "active (running)"
- ✅ Logs updating: `tail -f ~/btc-predictor/logs/btc_predictor_automation.log` shows activity
- ✅ Firebase populated: Firestore has data in collections
- ✅ Predictions being made: New documents appearing every 5 minutes
- ✅ Validations working: Predictions getting validated with WIN/LOSE

## 📞 Need Help?

Common issues:
1. ❌ **Import errors** → Run `pip install -r requirements.txt` in venv
2. ❌ **Firebase errors** → Check credentials file and database URL
3. ❌ **API errors** → Check internet connection, consider API key
4. ❌ **Memory errors** → Upgrade VPS to 4GB RAM

## 🎯 Next Steps

Once everything is running:

1. **Monitor for 24 hours** to ensure stability
2. **Check win rate** after 48-72 hours (need enough data)
3. **Adjust timeframes** in `config.py` if needed
4. **Set up alerts** (optional - use Firebase Cloud Functions)
5. **Create dashboard** (optional - use Firebase + React)

---

**🎊 Congratulations!** Your Bitcoin Predictor is now running 24/7 on your VPS!

The system will:
- 🔄 Make predictions every 5 minutes
- ✅ Validate results automatically
- 📊 Track win rate and accuracy
- 🔧 Retrain models every 24 hours
- 💾 Store everything in Firebase

**Remember:** This is for reference only. Always do your own research and use proper risk management when trading cryptocurrency! 🚨