#!/bin/bash

# Bitcoin Predictor VPS Setup Script
# Enhanced with error handling and validation

set -e  # Exit on error

echo "================================================================================"
echo "  Bitcoin Predictor - VPS Setup"
echo "================================================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Variables
USER="stcautotrade"
PROJECT_DIR="/home/$USER/btc-predictor"
VENV_DIR="$PROJECT_DIR/venv"
SERVICE_NAME="btc-predictor"

# Functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

check_system() {
    print_warning "Checking system requirements..."
    
    # Check Ubuntu/Debian
    if ! command -v apt-get &> /dev/null; then
        print_error "This script is for Ubuntu/Debian systems only"
        exit 1
    fi
    
    # Check RAM (minimum 2GB)
    total_ram=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_ram" -lt 2 ]; then
        print_warning "System has less than 2GB RAM. Performance may be limited."
    fi
    
    # Check disk space (minimum 5GB)
    free_space=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$free_space" -lt 5 ]; then
        print_error "Insufficient disk space. At least 5GB required."
        exit 1
    fi
    
    print_success "System check passed"
}

create_user() {
    if id "$USER" &>/dev/null; then
        print_warning "User $USER already exists"
    else
        print_warning "Creating user $USER..."
        useradd -m -s /bin/bash "$USER"
        print_success "User created"
    fi
}

install_dependencies() {
    print_warning "Installing system dependencies..."
    
    apt-get update
    apt-get install -y \
        python3.12 \
        python3.12-venv \
        python3-pip \
        git \
        curl \
        build-essential \
        python3-dev \
        libssl-dev \
        libffi-dev \
        htop \
        screen \
        fail2ban \
        ufw
    
    print_success "System dependencies installed"
}

setup_firewall() {
    print_warning "Configuring firewall..."
    
    # Enable UFW
    ufw --force enable
    
    # Allow SSH
    ufw allow ssh
    ufw allow 22/tcp
    
    # Allow HTTP/HTTPS (if needed)
    # ufw allow 80/tcp
    # ufw allow 443/tcp
    
    print_success "Firewall configured"
}

setup_project() {
    print_warning "Setting up project directory..."
    
    # Create directories
    mkdir -p "$PROJECT_DIR"
    mkdir -p "$PROJECT_DIR/logs"
    mkdir -p "$PROJECT_DIR/models"
    mkdir -p "$PROJECT_DIR/models_backup"
    
    # Set permissions
    chown -R $USER:$USER "$PROJECT_DIR"
    chmod -R 755 "$PROJECT_DIR"
    
    print_success "Project directory created"
}

setup_python_env() {
    print_warning "Setting up Python virtual environment..."
    
    # Create virtual environment
    sudo -u $USER python3.12 -m venv "$VENV_DIR"
    
    # Upgrade pip
    sudo -u $USER "$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel
    
    print_success "Virtual environment created"
}

install_python_packages() {
    print_warning "Installing Python packages..."
    
    # Copy requirements.txt if exists
    if [ -f "requirements.txt" ]; then
        cp requirements.txt "$PROJECT_DIR/"
        chown $USER:$USER "$PROJECT_DIR/requirements.txt"
        
        # Install packages
        sudo -u $USER "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
        
        print_success "Python packages installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

copy_project_files() {
    print_warning "Copying project files..."
    
    # List of files to copy
    files=(
        "config.py"
        "firebase_manager.py"
        "btc_predictor_automated.py"
        "scheduler.py"
        "system_health.py"
        "monitor.py"
        "maintenance.py"
    )
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            cp "$file" "$PROJECT_DIR/"
            chown $USER:$USER "$PROJECT_DIR/$file"
            print_success "Copied $file"
        else
            print_warning "File not found: $file"
        fi
    done
}

setup_firebase() {
    print_warning "Setting up Firebase credentials..."
    
    if [ -f "service-account.json" ]; then
        cp service-account.json "$PROJECT_DIR/"
        chown $USER:$USER "$PROJECT_DIR/service-account.json"
        chmod 600 "$PROJECT_DIR/service-account.json"
        print_success "Firebase credentials copied"
    else
        print_error "service-account.json not found!"
        print_warning "Please copy your Firebase service account JSON file to:"
        print_warning "$PROJECT_DIR/service-account.json"
    fi
}

setup_systemd_service() {
    print_warning "Setting up systemd service..."
    
    # Create service file
    cat > /etc/systemd/system/$SERVICE_NAME.service << EOF
[Unit]
Description=Bitcoin Price Predictor Automation Service
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"

ExecStart=$VENV_DIR/bin/python3 scheduler.py

Restart=always
RestartSec=30
StartLimitBurst=5
StartLimitIntervalSec=600

TimeoutStartSec=300
TimeoutStopSec=60

MemoryMax=2G
MemoryHigh=1.5G
CPUQuota=80%

StandardOutput=append:$PROJECT_DIR/logs/service_output.log
StandardError=append:$PROJECT_DIR/logs/service_error.log
SyslogIdentifier=$SERVICE_NAME

NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=$PROJECT_DIR

WatchdogSec=600
NotifyAccess=all

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd
    systemctl daemon-reload
    
    print_success "Systemd service created"
}

setup_log_rotation() {
    print_warning "Setting up log rotation..."
    
    cat > /etc/logrotate.d/$SERVICE_NAME << EOF
$PROJECT_DIR/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0644 $USER $USER
    sharedscripts
    postrotate
        systemctl reload $SERVICE_NAME > /dev/null 2>&1 || true
    endscript
}
EOF
    
    print_success "Log rotation configured"
}

setup_monitoring() {
    print_warning "Setting up monitoring cron jobs..."
    
    # Add cron job for health monitoring
    (sudo -u $USER crontab -l 2>/dev/null; echo "*/15 * * * * cd $PROJECT_DIR && $VENV_DIR/bin/python monitor.py health >> $PROJECT_DIR/logs/monitor.log 2>&1") | sudo -u $USER crontab -
    
    print_success "Monitoring cron jobs added"
}

create_scripts() {
    print_warning "Creating management scripts..."
    
    # Start script
    cat > "$PROJECT_DIR/start.sh" << EOF
#!/bin/bash
sudo systemctl start $SERVICE_NAME
sudo systemctl status $SERVICE_NAME
EOF
    
    # Stop script
    cat > "$PROJECT_DIR/stop.sh" << EOF
#!/bin/bash
sudo systemctl stop $SERVICE_NAME
EOF
    
    # Restart script
    cat > "$PROJECT_DIR/restart.sh" << EOF
#!/bin/bash
sudo systemctl restart $SERVICE_NAME
sudo systemctl status $SERVICE_NAME
EOF
    
    # Status script
    cat > "$PROJECT_DIR/status.sh" << EOF
#!/bin/bash
sudo systemctl status $SERVICE_NAME
EOF
    
    # Logs script
    cat > "$PROJECT_DIR/logs.sh" << EOF
#!/bin/bash
sudo journalctl -u $SERVICE_NAME -f
EOF
    
    # Make executable
    chmod +x "$PROJECT_DIR"/*.sh
    chown $USER:$USER "$PROJECT_DIR"/*.sh
    
    print_success "Management scripts created"
}

display_summary() {
    echo ""
    echo "================================================================================"
    echo "  Setup Complete!"
    echo "================================================================================"
    echo ""
    echo "Project Directory: $PROJECT_DIR"
    echo "Service Name: $SERVICE_NAME"
    echo ""
    echo "Management Commands:"
    echo "  Start:   sudo systemctl start $SERVICE_NAME"
    echo "  Stop:    sudo systemctl stop $SERVICE_NAME"
    echo "  Restart: sudo systemctl restart $SERVICE_NAME"
    echo "  Status:  sudo systemctl status $SERVICE_NAME"
    echo "  Logs:    sudo journalctl -u $SERVICE_NAME -f"
    echo ""
    echo "Or use the convenience scripts:"
    echo "  $PROJECT_DIR/start.sh"
    echo "  $PROJECT_DIR/stop.sh"
    echo "  $PROJECT_DIR/restart.sh"
    echo "  $PROJECT_DIR/status.sh"
    echo "  $PROJECT_DIR/logs.sh"
    echo ""
    echo "Monitoring:"
    echo "  cd $PROJECT_DIR && $VENV_DIR/bin/python monitor.py"
    echo ""
    echo "Maintenance:"
    echo "  cd $PROJECT_DIR && $VENV_DIR/bin/python maintenance.py"
    echo ""
    echo "Next Steps:"
    echo "  1. Verify Firebase credentials: $PROJECT_DIR/service-account.json"
    echo "  2. Check configuration: $PROJECT_DIR/config.py"
    echo "  3. Enable service: sudo systemctl enable $SERVICE_NAME"
    echo "  4. Start service: sudo systemctl start $SERVICE_NAME"
    echo ""
    echo "================================================================================"
}

# Main execution
main() {
    check_root
    check_system
    create_user
    install_dependencies
    setup_firewall
    setup_project
    setup_python_env
    install_python_packages
    copy_project_files
    setup_firebase
    setup_systemd_service
    setup_log_rotation
    setup_monitoring
    create_scripts
    display_summary
}

# Run main
main