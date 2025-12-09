# Bitcoin Predictor - Full Clean VPS Deployment Guide

## 📋 Prerequisites

Sebelum mulai, pastikan Anda memiliki:
- ✅ VPS dengan Ubuntu 20.04/22.04 LTS
- ✅ SSH access ke VPS
- ✅ CryptoCompare API key
- ✅ Firebase service account JSON file
- ✅ Minimal 2GB RAM, 20GB disk space

---

## 🧹 STEP 1: Stop & Cleanup (Jika Ada Service Lama)

```bash
# Connect ke VPS
ssh stcautotrade@your-vps-ip

# Stop service lama jika ada
sudo systemctl stop btc-predictor
sudo systemctl disable btc-predictor

# Backup data penting (optional)
cd /home/stcautotrade
mkdir -p ~/backup-$(date +%Y%m%d)
cp -r btc-predictor/logs ~/backup-$(date +%Y%m%d)/ 2>/dev/null || true
cp -r btc-predictor/models ~/backup-$(date +%Y%m%d)/ 2>/dev/null || true

# Hapus direktori lama
rm -rf btc-predictor

# Hapus service file lama
sudo rm -f /etc/systemd/system/btc-predictor.service
sudo systemctl daemon-reload
```

---

## 🔧 STEP 2: Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+
sudo apt install -y python3 python3-pip python3-venv python3-dev

# Install system dependencies
sudo apt install -y build-essential libssl-dev libffi-dev \
    git curl wget redis-server

# Install TensorFlow dependencies
sudo apt install -y libhdf5-dev libc-ares-dev libeigen3-dev \
    libatlas-base-dev libopenblas-dev

# Verify Python version (harus 3.8+)
python3 --version

# Start & enable Redis
sudo systemctl start redis-server
sudo systemctl enable redis-server
sudo systemctl status redis-server
```

---

## 📁 STEP 3: Create Project Directory

```bash
# Buat direktori project
cd /home/stcautotrade
mkdir -p btc-predictor
cd btc-predictor

# Buat struktur folder
mkdir -p logs models models_backup
```

---

## 📤 STEP 4: Upload Files ke VPS

**Option A: Via SCP (dari komputer lokal)**

```bash
# Dari komputer lokal (bukan di VPS)
# Ganti path sesuai lokasi project Anda

# Upload semua file Python
scp *.py stcautotrade@your-vps-ip:/home/stcautotrade/btc-predictor/

# Upload requirements.txt
scp requirements.txt stcautotrade@your-vps-ip:/home/stcautotrade/btc-predictor/

# Upload service file
scp btc-predictor.service stcautotrade@your-vps-ip:/home/stcautotrade/btc-predictor/

# Upload .env (PENTING!)
scp .env stcautotrade@your-vps-ip:/home/stcautotrade/btc-predictor/

# Upload service-account.json
scp service-account.json stcautotrade@your-vps-ip:/home/stcautotrade/btc-predictor/
```

**Option B: Via Git (recommended)**

```bash
# Di VPS
cd /home/stcautotrade/btc-predictor

# Clone dari repository
git clone https://github.com/your-username/btc-predictor.git .

# Atau pull jika sudah ada
git pull origin main
```

**Option C: Via SFTP GUI**
- Gunakan FileZilla atau WinSCP
- Connect ke VPS
- Upload semua file ke `/home/stcautotrade/btc-predictor/`

---

## 🔐 STEP 5: Setup Environment Files

```bash
cd /home/stcautotrade/btc-predictor

# Create .env file
nano .env
```

**Paste dan edit ini:**

```ini
# API Keys
CRYPTOCOMPARE_API_KEY=your_api_key_here

# Firebase
FIREBASE_CREDENTIALS_PATH=/home/stcautotrade/btc-predictor/service-account.json
FIREBASE_DATABASE_URL=https://your-project.firebaseio.com

# Environment
ENVIRONMENT=production
TRADING_MODE=paper

# System Limits
MAX_MEMORY_MB=2048
MAX_CPU_PERCENT=90
LOG_LEVEL=INFO

# Confidence Thresholds (OPTIMIZED)
MIN_CONFIDENCE_ULTRA_SHORT=40
MIN_CONFIDENCE_SHORT=38
MIN_CONFIDENCE_MEDIUM=35
MIN_CONFIDENCE_LONG=32

# Alerts (Optional)
ENABLE_ALERTS=false
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Redis
ENABLE_REDIS=true
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
```

**Save:** `Ctrl+X`, `Y`, `Enter`

```bash
# Verify .env exists
cat .env

# Setup Firebase service account
nano service-account.json
```

**Paste JSON dari Firebase Console:**

```json
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  ...
}
```

**Save:** `Ctrl+X`, `Y`, `Enter`

```bash
# Set permissions
chmod 600 .env
chmod 600 service-account.json
chmod +x *.py
```

---

## 🐍 STEP 6: Setup Python Virtual Environment

```bash
cd /home/stcautotrade/btc-predictor

# Create virtual environment
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies (ini bisa lama 10-20 menit)
pip install -r requirements.txt

# Verify installations
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import pandas as pd; print(f'Pandas: {pd.__version__}')"
python -c "import firebase_admin; print('Firebase: OK')"

# Deactivate venv
deactivate
```

---

## ✅ STEP 7: Test System

```bash
cd /home/stcautotrade/btc-predictor
source venv/bin/activate

# Test 1: Environment
python3 test_system.py --quick

# Test 2: Redis
python3 test_redis_connection.py

# Test 3: Firebase
python3 -c "from firebase_manager import FirebaseManager; fb = FirebaseManager(); print('Firebase OK')"

# Test 4: API
python3 -c "from btc_predictor_automated import get_current_btc_price; print(f'BTC: ${get_current_btc_price()}')"

deactivate
```

**Semua test harus PASS!** ✅

---

## 🚀 STEP 8: Setup Systemd Service

```bash
# Copy service file ke systemd
sudo cp /home/stcautotrade/btc-predictor/btc-predictor.service \
    /etc/systemd/system/btc-predictor.service

# Edit jika perlu
sudo nano /etc/systemd/system/btc-predictor.service

# Reload systemd
sudo systemctl daemon-reload

# Enable service (auto-start on boot)
sudo systemctl enable btc-predictor

# Start service
sudo systemctl start btc-predictor

# Check status
sudo systemctl status btc-predictor
```

**Status harus "active (running)"** 🟢

---

## 📊 STEP 9: Monitor Logs

```bash
# Follow live logs
sudo journalctl -u btc-predictor -f

# View last 100 lines
sudo journalctl -u btc-predictor -n 100

# View logs from today
sudo journalctl -u btc-predictor --since today

# View error logs only
sudo journalctl -u btc-predictor -p err
```

**Yang harus Anda lihat:**
- ✅ Firebase connected
- ✅ Models loaded atau training started
- ✅ System started
- ✅ Predictions being made

---

## 🔍 STEP 10: Verify Everything Works

```bash
cd /home/stcautotrade/btc-predictor
source venv/bin/activate

# Check system health
python3 monitor.py health

# Check recent predictions
python3 monitor.py recent

# Full dashboard
python3 monitor.py full

deactivate
```

---

## 🛠️ Useful Commands

### Service Management
```bash
# Stop service
sudo systemctl stop btc-predictor

# Start service
sudo systemctl start btc-predictor

# Restart service
sudo systemctl restart btc-predictor

# Check status
sudo systemctl status btc-predictor

# Disable auto-start
sudo systemctl disable btc-predictor
```

### Logs
```bash
# Live logs
sudo journalctl -u btc-predictor -f

# Last 50 lines
sudo journalctl -u btc-predictor -n 50

# Errors only
sudo journalctl -u btc-predictor -p err

# Clear old logs
sudo journalctl --vacuum-time=7d
```

### Maintenance
```bash
cd /home/stcautotrade/btc-predictor
source venv/bin/activate

# Force retrain models
python3 maintenance.py retrain

# Validate pending predictions
python3 maintenance.py validate

# Cleanup old data (30 days)
python3 maintenance.py cleanup 30

# Export predictions CSV
python3 maintenance.py export 7

deactivate
```

### System Info
```bash
# Memory usage
free -h

# Disk usage
df -h

# CPU usage
top -bn1 | head -20

# Redis status
redis-cli ping

# Check processes
ps aux | grep python
```

---

## 🔧 Troubleshooting

### Problem: Service tidak start

```bash
# Check logs
sudo journalctl -u btc-predictor -n 50

# Common issues:
# 1. .env tidak ada
ls -la /home/stcautotrade/btc-predictor/.env

# 2. venv tidak ada
ls -la /home/stcautotrade/btc-predictor/venv/

# 3. Permissions
sudo chown -R stcautotrade:stcautotrade /home/stcautotrade/btc-predictor
chmod +x /home/stcautotrade/btc-predictor/*.py
```

### Problem: High memory

```bash
# Check memory
free -h

# Restart service
sudo systemctl restart btc-predictor

# If persistent, adjust MAX_MEMORY_MB in .env
nano /home/stcautotrade/btc-predictor/.env
# Change: MAX_MEMORY_MB=1536
sudo systemctl restart btc-predictor
```

### Problem: Firebase tidak connect

```bash
# Check credentials
cat /home/stcautotrade/btc-predictor/service-account.json

# Test Firebase
cd /home/stcautotrade/btc-predictor
source venv/bin/activate
python3 -c "from firebase_manager import FirebaseManager; FirebaseManager()"
```

### Problem: No predictions

```bash
# Check logs
sudo journalctl -u btc-predictor -f

# Check if models exist
ls -lh /home/stcautotrade/btc-predictor/models/

# If no models, retrain
cd /home/stcautotrade/btc-predictor
source venv/bin/activate
python3 maintenance.py retrain
```

### Problem: Redis not working

```bash
# Check Redis status
sudo systemctl status redis-server

# Restart Redis
sudo systemctl restart redis-server

# Test connection
redis-cli ping

# Disable Redis if problematic
nano /home/stcautotrade/btc-predictor/.env
# Change: ENABLE_REDIS=false
sudo systemctl restart btc-predictor
```

---

## 📈 Post-Deployment Checklist

- [ ] Service running (`sudo systemctl status btc-predictor`)
- [ ] Logs showing predictions (`sudo journalctl -u btc-predictor -f`)
- [ ] Firebase receiving data (check Firebase Console)
- [ ] No error messages in logs
- [ ] Memory usage under limit (`free -h`)
- [ ] Redis working (`redis-cli ping`)
- [ ] Models loaded or training
- [ ] Predictions appearing in monitor

---

## 🎯 Next Steps

### 1. Monitor First Hour
```bash
# Watch logs
sudo journalctl -u btc-predictor -f
```

### 2. Check After 1 Hour
```bash
cd /home/stcautotrade/btc-predictor
source venv/bin/activate
python3 monitor.py full
```

### 3. Setup Monitoring (Optional)
```bash
# Add to crontab for daily health check
crontab -e

# Add this line:
0 */6 * * * cd /home/stcautotrade/btc-predictor && /home/stcautotrade/btc-predictor/venv/bin/python3 monitor.py health >> logs/health.log 2>&1
```

---

## 📞 Support

Jika ada masalah:

1. **Check logs first:**
   ```bash
   sudo journalctl -u btc-predictor -n 100
   ```

2. **Check system resources:**
   ```bash
   free -h
   df -h
   ```

3. **Run test suite:**
   ```bash
   cd /home/stcautotrade/btc-predictor
   source venv/bin/activate
   python3 test_system.py --full
   ```

4. **Restart service:**
   ```bash
   sudo systemctl restart btc-predictor
   ```

---

## 🎉 Success Indicators

Anda tahu deployment berhasil jika:

✅ Service status = **active (running)**  
✅ Logs show predictions every minute  
✅ Firebase Console menunjukkan data baru  
✅ No error messages  
✅ Memory stable < 2GB  
✅ `python3 monitor.py full` shows stats  

**Selamat! Bitcoin Predictor Anda sudah running! 🚀**

---

## 📝 Quick Reference

```bash
# Status check
sudo systemctl status btc-predictor

# Live logs
sudo journalctl -u btc-predictor -f

# Monitor dashboard
cd /home/stcautotrade/btc-predictor && source venv/bin/activate && python3 monitor.py full

# Restart service
sudo systemctl restart btc-predictor

# Stop service
sudo systemctl stop btc-predictor

# Memory check
free -h

# Disk check
df -h
```