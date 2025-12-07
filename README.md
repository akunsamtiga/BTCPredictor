# 🔄 VPS Complete Restart Guide - Bitcoin Predictor

## 📋 Daftar Isi
1. [Stop & Kill Semua Proses](#1-stop--kill-semua-proses)
2. [Backup Data Penting](#2-backup-data-penting)
3. [Clean Installation](#3-clean-installation)
4. [Verifikasi & Testing](#4-verifikasi--testing)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Stop & Kill Semua Proses

### A. Stop Systemd Service
```bash
# Stop service
sudo systemctl stop btc-predictor

# Disable auto-start
sudo systemctl disable btc-predictor

# Verify stopped
sudo systemctl status btc-predictor
```

### B. Kill Manual Processes
```bash
# Cari semua proses Python yang terkait
ps aux | grep -E "(scheduler|btc_predictor|python3)"

# Kill berdasarkan nama
pkill -9 -f scheduler
pkill -9 -f btc_predictor
pkill -9 -f "python3.*scheduler"

# Alternative: Kill by user (jika diperlukan)
pkill -9 -u stcautotrade

# Verify tidak ada proses yang tersisa
ps aux | grep scheduler
```

### C. Cek & Kill Port yang Digunakan
```bash
# Cek port yang digunakan (biasanya 8000 untuk Prometheus)
sudo lsof -i :8000
sudo netstat -tulpn | grep 8000

# Kill process di port tersebut
sudo fuser -k 8000/tcp

# Atau kill by PID
sudo kill -9 <PID>
```

### D. Clean Zombie Processes
```bash
# Cek zombie processes
ps aux | grep 'Z'

# Clean systemd
sudo systemctl daemon-reload
sudo systemctl reset-failed
```

---

## 2. Backup Data Penting

### A. Backup Models
```bash
# Masuk ke directory project
cd /home/stcautotrade/btc-predictor

# Backup models
mkdir -p ~/backups/models_$(date +%Y%m%d_%H%M%S)
cp -r models/* ~/backups/models_$(date +%Y%m%d_%H%M%S)/

# Verify backup
ls -lh ~/backups/
```

### B. Backup Logs
```bash
# Backup logs
mkdir -p ~/backups/logs_$(date +%Y%m%d_%H%M%S)
cp -r logs/* ~/backups/logs_$(date +%Y%m%d_%H%M%S)/

# Optional: compress logs
tar -czf ~/backups/logs_$(date +%Y%m%d).tar.gz logs/
```

### C. Backup Configuration
```bash
# Backup .env file
cp .env ~/backups/.env.backup

# Backup service file
sudo cp /etc/systemd/system/btc-predictor.service ~/backups/
```

---

## 3. Clean Installation

### A. Clean Virtual Environment
```bash
cd /home/stcautotrade/btc-predictor

# Deactivate jika aktif
deactivate 2>/dev/null || true

# Remove old venv
rm -rf venv

# Create new venv
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### B. Install Dependencies
```bash
# Install dari requirements.txt
pip install -r requirements.txt

# Verify installations
pip list | grep -E "(tensorflow|pandas|firebase|schedule)"
```

### C. Clean Logs & Temp Files
```bash
# Clean old logs (keep backup)
rm -f logs/*.log
rm -f logs/*.txt

# Recreate logs directory
mkdir -p logs
touch logs/.gitkeep

# Clean Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
```

### D. Environment Variables Check
```bash
# Check .env file
cat .env

# Verify critical variables
echo "Checking environment variables..."
grep -E "(API_KEY|FIREBASE|ENVIRONMENT|TRADING_MODE)" .env

# Set permissions (tidak perlu)
chmod 600 .env
```

### E. Test Import Dependencies
```bash
# Quick test Python imports
python3 << EOF
try:
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import firebase_admin
    import schedule
    print("✅ All critical imports successful")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Pandas: {pd.__version__}")
except ImportError as e:
    print(f"❌ Import error: {e}")
EOF
```

---

## 4. Verifikasi & Testing

### A. Run System Tests
```bash
# Activate venv jika belum
source venv/bin/activate

# Run quick test
python3 test_system.py --quick

# Run full test (optional)
python3 test_system.py --full
```

### B. Test Firebase Connection
```bash
python3 << EOF
from firebase_manager import FirebaseManager

try:
    fb = FirebaseManager()
    if fb.connected:
        print("✅ Firebase connected successfully")
    else:
        print("❌ Firebase connection failed")
except Exception as e:
    print(f"❌ Error: {e}")
EOF
```

### C. Test API Connection
```bash
python3 << EOF
from btc_predictor_automated import get_current_btc_price

try:
    price = get_current_btc_price()
    if price:
        print(f"✅ API working - Current BTC: ${price:,.2f}")
    else:
        print("❌ API failed")
except Exception as e:
    print(f"❌ Error: {e}")
EOF
```

### D. Manual Test Run
```bash
# Run scheduler manually (test mode)
python3 scheduler.py

# Biarkan jalan 1-2 menit, lalu Ctrl+C
# Check logs
tail -f logs/scheduler.log
```

---

## 5. Start Fresh Service

### A. Update Service File (Optional)
```bash
# Edit service file jika perlu
sudo nano /etc/systemd/system/btc-predictor.service

# Reload systemd
sudo systemctl daemon-reload
```

### B. Start Service Fresh
```bash
# Enable service
sudo systemctl enable btc-predictor

# Start service
sudo systemctl start btc-predictor

# Check status immediately
sudo systemctl status btc-predictor

# Check logs real-time
sudo journalctl -u btc-predictor -f
```

### C. Monitor First 5 Minutes
```bash
# Watch logs continuously
tail -f logs/scheduler.log

# In another terminal, check status
watch -n 5 'sudo systemctl status btc-predictor'

# Check memory usage
watch -n 5 'free -h'

# Check CPU usage
top -p $(pgrep -f scheduler)
```

### D. Verify System Health
```bash
# After 5 minutes, check health
python3 << EOF
from system_health import monitor_health

report = monitor_health()
print(f"Status: {report['overall_status']}")
print(f"Memory: {report['memory']['process_memory_mb']:.0f}MB")
print(f"CPU: {report['cpu']['cpu_percent']:.1f}%")
EOF
```

---

## 6. Troubleshooting Common Issues

### A. Service Won't Start
```bash
# Check detailed error
sudo journalctl -u btc-predictor -n 50 --no-pager

# Check permissions
ls -la /home/stcautotrade/btc-predictor/
ls -la ~/.config/gcloud/  # For Firebase

# Check Python path
which python3
python3 --version

# Test direct run
cd /home/stcautotrade/btc-predictor
source venv/bin/activate
python3 scheduler.py
```

### B. Import Errors
```bash
# Reinstall problematic package
pip uninstall tensorflow
pip install tensorflow==2.18.0

# Or reinstall all
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### C. Firebase Connection Issues
```bash
# Check credentials file
ls -la service-account.json

# Check permissions
chmod 600 service-account.json

# Test connection
python3 -c "from firebase_manager import FirebaseManager; fb = FirebaseManager(); print('Connected' if fb.connected else 'Failed')"
```

### D. Memory Issues
```bash
# Check available memory
free -h

# Clear cache
sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches

# Restart with clean state
sudo systemctl restart btc-predictor
```

### E. Port Already in Use
```bash
# Find and kill process using port
sudo lsof -ti:8000 | xargs kill -9

# Or change port in config
# Edit .env:
# PROMETHEUS_PORT=8001
```

---

## 7. Post-Restart Checklist

### ✅ Immediate Checks (First 5 minutes)
- [ ] Service status shows "active (running)"
- [ ] No critical errors in logs
- [ ] Memory usage < 500MB initially
- [ ] CPU usage stabilizes
- [ ] No crash/restart loops

### ✅ Short-term Checks (First hour)
- [ ] First prediction created successfully
- [ ] Firebase writes successful
- [ ] Memory stable (not growing)
- [ ] No repeated errors
- [ ] Heartbeat updates in Firebase

### ✅ Long-term Monitoring (First 24h)
- [ ] Multiple predictions completed
- [ ] Validations working
- [ ] Statistics updating
- [ ] Memory stays under 2GB
- [ ] No unexpected restarts

---

## 8. Quick Reference Commands

```bash
# Stop everything
sudo systemctl stop btc-predictor
pkill -9 -f scheduler

# Clean restart
sudo systemctl daemon-reload
sudo systemctl start btc-predictor
sudo systemctl status btc-predictor

# Monitor
sudo journalctl -u btc-predictor -f
tail -f logs/scheduler.log

# Check health
python3 -c "from system_health import monitor_health; monitor_health()"

# Emergency stop
sudo systemctl stop btc-predictor
sudo kill -9 $(pgrep -f scheduler)
```

---

## 9. Prevention Tips

### A. Before Next Restart
```bash
# Export current state
python3 monitor.py export

# Backup models
cp -r models models_backup_$(date +%Y%m%d)

# Save logs
tar -czf logs_backup.tar.gz logs/
```

### B. Set Up Monitoring
```bash
# Add to crontab for health check
crontab -e

# Add line:
# */5 * * * * /home/stcautotrade/btc-predictor/venv/bin/python3 /home/stcautotrade/btc-predictor/monitor.py health >> /home/stcautotrade/health.log 2>&1
```

### C. Auto-cleanup Script
Create `cleanup.sh`:
```bash
#!/bin/bash
cd /home/stcautotrade/btc-predictor

# Clean old logs (keep 7 days)
find logs/ -name "*.log" -mtime +7 -delete

# Clean old backups
find ~/backups/ -mtime +30 -delete

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

echo "Cleanup completed: $(date)"
```

Make executable:
```bash
chmod +x cleanup.sh
```

---

## 🆘 Emergency Contact Checklist

Jika masih bermasalah setelah restart:

1. **Screenshot error logs**:
   ```bash
   sudo journalctl -u btc-predictor -n 100 > error_log.txt
   ```

2. **System info**:
   ```bash
   free -h > system_info.txt
   df -h >> system_info.txt
   top -bn1 | head -20 >> system_info.txt
   ```

3. **Environment info**:
   ```bash
   cat .env | grep -v "PASSWORD\|TOKEN\|KEY" > env_info.txt
   ```

4. **Package versions**:
   ```bash
   pip list > packages.txt
   ```

---

## 📝 Notes

- **Backup**: Selalu backup sebelum restart total
- **Testing**: Test manual dulu sebelum enable service
- **Monitoring**: Monitor setidaknya 1 jam setelah restart
- **Logs**: Periksa logs secara berkala
- **Memory**: Watch memory usage, restart jika > 2.5GB

---

**Good luck! 🚀**

Jika ada error specific, share error message untuk troubleshooting lebih detail.