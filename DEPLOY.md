# 🚀 Panduan Complete: Push ke GitHub → Pull & Deploy di VPS

## 📤 PART 1: Push Update dari Local ke GitHub

### **Step 1: Edit & Test di Local**

```bash
# Di komputer local
cd /path/to/btc-predictor

# Edit file yang mau diubah
nano config.py
# atau
nano scheduler.py

# Test perubahan
python3 scheduler.py
```

### **Step 2: Commit & Push ke GitHub**

```bash
# Check status
git status

# Add files yang diubah
git add config.py
git add scheduler.py
# atau add semua
git add .

# Commit dengan message yang jelas
git commit -m "Update: Tambah fitur X dan fix bug Y"

# Push ke GitHub
git push origin main

# Verify di GitHub
# Buka: https://github.com/akunsamtiga/BTCPredictor
# Pastikan commit sudah masuk
```

---

## 📥 PART 2: Pull & Deploy Update di VPS

### **Method A: Quick Update (No Breaking Changes)**

Gunakan ini jika update tidak mengubah dependencies atau struktur database:

```bash
# 1. Stop service
sudo systemctl stop btc-predictor

# 2. Navigate to directory
cd /home/stcautotrade/btc-predictor

# 3. Pull latest changes
git pull origin main

# 4. Restart service
sudo systemctl start btc-predictor

# 5. Monitor logs
sudo journalctl -u btc-predictor -f
```

### **Method B: Safe Update dengan Backup**

Gunakan ini untuk update yang lebih besar:

```bash
# 1. Stop service
sudo systemctl stop btc-predictor

# 2. Navigate & check current status
cd /home/stcautotrade/btc-predictor
git status
git log -1  # Lihat commit terakhir

# 3. Backup file penting
cp config.py config.py.backup.$(date +%Y%m%d_%H%M%S)
cp service-account.json service-account.json.backup
cp -r models models_backup_$(date +%Y%m%d)
cp -r logs logs_backup_$(date +%Y%m%d)

# 4. Stash local changes (jika ada)
git stash

# 5. Pull latest
git pull origin main

# 6. Restore service account jika tertimpa
if [ ! -f service-account.json ]; then
    cp service-account.json.backup service-account.json
    chmod 600 service-account.json
fi

# 7. Update dependencies (jika requirements.txt berubah)
source venv/bin/activate
pip install --upgrade -r requirements.txt

# 8. Restart service
sudo systemctl start btc-predictor

# 9. Monitor
sudo journalctl -u btc-predictor -f
```

### **Method C: Hard Reset & Deploy (Clean Slate)**

Gunakan ini jika ada conflict atau mau clean install:

```bash
# 1. Stop service
sudo systemctl stop btc-predictor

# 2. Full backup
cd /home/stcautotrade/btc-predictor
BACKUP_DIR=~/btc-full-backup-$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR
cp -r models $BACKUP_DIR/
cp -r logs $BACKUP_DIR/
cp config.py $BACKUP_DIR/ 2>/dev/null
cp service-account.json $BACKUP_DIR/
echo "✅ Backup saved to: $BACKUP_DIR"

# 3. Hard reset dari GitHub
git fetch origin
git reset --hard origin/main
git clean -fd  # Hapus untracked files

# 4. Restore file critical
cp $BACKUP_DIR/service-account.json .
chmod 600 service-account.json

# Optional: restore models (jika tidak ingin retrain)
# cp -r $BACKUP_DIR/models .

# 5. Update dependencies
source venv/bin/activate
pip install --upgrade -r requirements.txt

# 6. Restart
sudo systemctl start btc-predictor

# 7. Monitor
sudo journalctl -u btc-predictor -f
```

---

## 🤖 Automated Update Script

Buat script untuk otomasi proses update:

```bash
nano ~/pull-and-deploy.sh
```

Paste script ini:

```bash
#!/bin/bash

#==============================================================================
# Bitcoin Predictor - Pull & Deploy Script
# Usage: ~/pull-and-deploy.sh [soft|hard]
#==============================================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="/home/stcautotrade/btc-predictor"
BACKUP_DIR="$HOME/btc-backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check mode
MODE=${1:-soft}

echo "=========================================="
echo "Bitcoin Predictor - Pull & Deploy"
echo "Mode: $MODE"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="

# Stop service
log_info "Stopping service..."
sudo systemctl stop btc-predictor

if [ $? -ne 0 ]; then
    log_error "Failed to stop service"
    exit 1
fi

# Navigate to project
cd $PROJECT_DIR

if [ $? -ne 0 ]; then
    log_error "Project directory not found: $PROJECT_DIR"
    exit 1
fi

# Create backup
log_info "Creating backup..."
mkdir -p $BACKUP_DIR/backup_$TIMESTAMP

cp config.py $BACKUP_DIR/backup_$TIMESTAMP/ 2>/dev/null
cp service-account.json $BACKUP_DIR/backup_$TIMESTAMP/
cp -r models $BACKUP_DIR/backup_$TIMESTAMP/ 2>/dev/null
cp -r logs $BACKUP_DIR/backup_$TIMESTAMP/ 2>/dev/null

log_info "Backup saved to: $BACKUP_DIR/backup_$TIMESTAMP"

# Git operations
log_info "Fetching from GitHub..."
git fetch origin

if [ "$MODE" == "hard" ]; then
    log_warn "Performing HARD reset..."
    
    # Stash any local changes
    git stash push -m "Auto-stash before hard reset $TIMESTAMP"
    
    # Hard reset
    git reset --hard origin/main
    
    # Clean untracked files
    git clean -fd
    
    log_info "Hard reset completed"
else
    log_info "Performing soft pull..."
    
    # Stash local changes
    git stash
    
    # Pull updates
    git pull origin main
    
    if [ $? -ne 0 ]; then
        log_error "Git pull failed. Try with 'hard' mode."
        log_warn "Restoring service..."
        sudo systemctl start btc-predictor
        exit 1
    fi
    
    log_info "Pull completed"
fi

# Show what changed
log_info "Recent commits:"
git log -3 --oneline --decorate

# Restore critical files
log_info "Restoring critical files..."

if [ ! -f service-account.json ]; then
    log_warn "service-account.json missing, restoring from backup..."
    cp $BACKUP_DIR/backup_$TIMESTAMP/service-account.json .
    chmod 600 service-account.json
fi

# Update dependencies if requirements changed
if git diff HEAD@{1} HEAD -- requirements.txt | grep -q '^[+-]'; then
    log_info "requirements.txt changed, updating dependencies..."
    source venv/bin/activate
    pip install --upgrade -r requirements.txt
    
    if [ $? -ne 0 ]; then
        log_error "Failed to update dependencies"
        log_warn "Continuing anyway..."
    fi
else
    log_info "No dependency changes detected"
fi

# Restart service
log_info "Starting service..."
sudo systemctl start btc-predictor

if [ $? -ne 0 ]; then
    log_error "Failed to start service"
    log_error "Check logs: sudo journalctl -u btc-predictor -n 50"
    exit 1
fi

# Wait a bit
sleep 5

# Check status
log_info "Checking service status..."
sudo systemctl status btc-predictor --no-pager -l

# Show recent logs
log_info "Recent logs:"
sudo journalctl -u btc-predictor -n 10 --no-pager

echo ""
echo "=========================================="
echo "✅ Deployment completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  • Monitor logs: sudo journalctl -u btc-predictor -f"
echo "  • Check status: sudo systemctl status btc-predictor"
echo "  • View backup: ls -la $BACKUP_DIR/backup_$TIMESTAMP"
echo "  • Rollback: cd $PROJECT_DIR && git reset --hard HEAD~1 && sudo systemctl restart btc-predictor"
echo ""
```

Save dan buat executable:

```bash
chmod +x ~/pull-and-deploy.sh
```

### **Cara Menggunakan Script:**

```bash
# Soft update (recommended untuk update biasa)
~/pull-and-deploy.sh soft

# Hard reset (untuk major changes atau ada conflict)
~/pull-and-deploy.sh hard

# Default = soft
~/pull-and-deploy.sh
```

---

## 🔄 Complete Workflow: Local → GitHub → VPS

### **1. Di Local Machine (Development)**

```bash
# Edit code
nano config.py

# Test
python3 scheduler.py

# Add & commit
git add .
git commit -m "Update: Improve prediction accuracy"

# Push to GitHub
git push origin main
```

### **2. Di VPS (Production)**

```bash
# Option A: Menggunakan script automated
~/pull-and-deploy.sh soft

# Option B: Manual step-by-step
sudo systemctl stop btc-predictor
cd /home/stcautotrade/btc-predictor
git pull origin main
sudo systemctl start btc-predictor
sudo journalctl -u btc-predictor -f
```

---

## 📊 Update Tracking & Verification

### **Setelah Deploy, Verify:**

```bash
# 1. Check commit hash
cd /home/stcautotrade/btc-predictor
git log -1 --oneline

# 2. Check service status
sudo systemctl status btc-predictor

# 3. Check logs untuk errors
sudo journalctl -u btc-predictor -n 50 | grep -i error

# 4. Test prediction masih jalan
sudo journalctl -u btc-predictor -n 20 | grep "Prediction cycle"

# 5. Check Firebase masih connected
sudo journalctl -u btc-predictor -n 20 | grep Firebase
```

---

## 🆘 Rollback Jika Ada Masalah

```bash
# 1. Stop service
sudo systemctl stop btc-predictor

# 2. Rollback git
cd /home/stcautotrade/btc-predictor
git log --oneline -5  # Lihat 5 commit terakhir
git reset --hard <commit-hash-yang-aman>

# 3. Atau rollback ke commit sebelumnya
git reset --hard HEAD~1

# 4. Restore dari backup
LATEST_BACKUP=$(ls -td ~/btc-backups/* | head -1)
cp $LATEST_BACKUP/service-account.json .
cp -r $LATEST_BACKUP/models . 2>/dev/null

# 5. Restart
sudo systemctl start btc-predictor
sudo journalctl -u btc-predictor -f
```

---

## 🎯 Best Practices Workflow

1. **Always test locally first**
   ```bash
   python3 scheduler.py  # Di local machine
   ```

2. **Write clear commit messages**
   ```bash
   git commit -m "Fix: Resolve Firebase timeout issue"
   git commit -m "Feature: Add 2-hour timeframe prediction"
   git commit -m "Update: Improve LSTM model accuracy"
   ```

3. **Deploy during low traffic**
   - Idealnya saat prediction cycle selesai
   - Avoid deploy saat tengah training

4. **Monitor after deploy**
   ```bash
   # Watch logs selama 5-10 menit
   sudo journalctl -u btc-predictor -f
   ```

5. **Keep backups**
   ```bash
   # Auto-backup dengan script
   ~/pull-and-deploy.sh  # Sudah include backup
   ```

---

## 📋 Quick Reference

| Task | Command |
|------|---------|
| **Soft update** | `~/pull-and-deploy.sh soft` |
| **Hard update** | `~/pull-and-deploy.sh hard` |
| **Manual pull** | `cd ~/btc-predictor && git pull origin main` |
| **Check updates available** | `cd ~/btc-predictor && git fetch && git status` |
| **View commit log** | `cd ~/btc-predictor && git log --oneline -10` |
| **Rollback 1 commit** | `cd ~/btc-predictor && git reset --hard HEAD~1` |
| **Restart after update** | `sudo systemctl restart btc-predictor` |
| **Monitor logs** | `sudo journalctl -u btc-predictor -f` |

---

Sekarang Anda punya workflow lengkap dari development → GitHub → production deployment! 🚀

Ada yang ingin ditanyakan tentang CI/CD atau automation lainnya? 😊