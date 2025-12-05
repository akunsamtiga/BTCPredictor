#!/bin/bash

# Bitcoin Predictor - Quick Setup Script for VPS
# This script automates the installation process

set -e  # Exit on error

echo "=================================================="
echo "🪙 Bitcoin Predictor - VPS Setup"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Variables
INSTALL_DIR="$HOME/btc-predictor"
VENV_DIR="$INSTALL_DIR/venv"
LOG_DIR="$INSTALL_DIR/logs"
MODEL_DIR="$INSTALL_DIR/models"

# Functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}➜${NC} $1"
}

check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION detected"
        return 0
    else
        print_error "Python 3 not found"
        return 1
    fi
}

install_system_deps() {
    print_info "Installing system dependencies..."
    
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-venv build-essential \
            libssl-dev libffi-dev python3-dev
        print_success "System dependencies installed"
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3 python3-pip python3-devel gcc gcc-c++ make \
            openssl-devel libffi-devel
        print_success "System dependencies installed"
    else
        print_error "Unsupported package manager. Please install dependencies manually."
        exit 1
    fi
}

setup_directories() {
    print_info "Creating directories..."
    
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$MODEL_DIR"
    
    print_success "Directories created"
}

setup_venv() {
    print_info "Setting up virtual environment..."
    
    cd "$INSTALL_DIR"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    print_success "Virtual environment created"
}

install_python_deps() {
    print_info "Installing Python dependencies..."
    
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    
    if [ -f "$INSTALL_DIR/requirements.txt" ]; then
        pip install -r "$INSTALL_DIR/requirements.txt"
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
}

check_firebase_credentials() {
    print_info "Checking Firebase credentials..."
    
    if [ ! -f "$INSTALL_DIR/firebase-credentials.json" ]; then
        print_error "firebase-credentials.json not found!"
        echo ""
        echo "Please follow these steps:"
        echo "1. Go to Firebase Console: https://console.firebase.google.com/"
        echo "2. Select your project"
        echo "3. Project Settings → Service Accounts"
        echo "4. Generate new private key"
        echo "5. Save as firebase-credentials.json"
        echo "6. Upload to: $INSTALL_DIR/firebase-credentials.json"
        echo ""
        read -p "Press Enter after uploading firebase-credentials.json..."
        
        if [ ! -f "$INSTALL_DIR/firebase-credentials.json" ]; then
            print_error "firebase-credentials.json still not found. Exiting."
            exit 1
        fi
    fi
    
    print_success "Firebase credentials found"
}

configure_config() {
    print_info "Configuring application..."
    
    if [ ! -f "$INSTALL_DIR/config.py" ]; then
        print_error "config.py not found!"
        exit 1
    fi
    
    # Check if firebase URL is configured
    if grep -q "your-project.firebaseio.com" "$INSTALL_DIR/config.py"; then
        print_error "Please update Firebase URL in config.py"
        echo ""
        echo "Edit $INSTALL_DIR/config.py and update:"
        echo "  'database_url': 'https://YOUR-PROJECT-ID.firebaseio.com'"
        echo ""
        read -p "Press Enter after updating config.py..."
    fi
    
    print_success "Configuration checked"
}

setup_systemd() {
    print_info "Setting up systemd service..."
    
    # Create service file from template
    SERVICE_FILE="/tmp/btc-predictor.service"
    
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Bitcoin Price Predictor with ML and Firebase
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=$VENV_DIR/bin/python3 scheduler.py
Restart=always
RestartSec=10
StandardOutput=append:$LOG_DIR/output.log
StandardError=append:$LOG_DIR/error.log

[Install]
WantedBy=multi-user.target
EOF
    
    sudo cp "$SERVICE_FILE" /etc/systemd/system/btc-predictor.service
    sudo systemctl daemon-reload
    
    print_success "Systemd service created"
}

test_installation() {
    print_info "Testing installation..."
    
    cd "$INSTALL_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Test imports
    python3 << EOF
import sys
try:
    import tensorflow
    import pandas
    import numpy
    import firebase_admin
    import schedule
    print("✓ All dependencies imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
    else
        print_error "Installation test failed"
        exit 1
    fi
}

# Main Installation Process
main() {
    echo ""
    print_info "Starting installation process..."
    echo ""
    
    # Check if already installed
    if [ -d "$VENV_DIR" ]; then
        print_info "Installation directory already exists"
        read -p "Do you want to reinstall? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Installation cancelled"
            exit 0
        fi
    fi
    
    # Installation steps
    check_python || install_system_deps
    setup_directories
    setup_venv
    install_python_deps
    check_firebase_credentials
    configure_config
    test_installation
    
    # Optional systemd setup
    echo ""
    print_info "Installation completed!"
    echo ""
    read -p "Do you want to setup systemd service for auto-start? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_systemd
        
        echo ""
        print_success "Setup completed successfully!"
        echo ""
        echo "To start the service:"
        echo "  sudo systemctl start btc-predictor"
        echo ""
        echo "To enable auto-start on boot:"
        echo "  sudo systemctl enable btc-predictor"
        echo ""
        echo "To check status:"
        echo "  sudo systemctl status btc-predictor"
        echo ""
        echo "To view logs:"
        echo "  tail -f $LOG_DIR/btc_predictor_automation.log"
        echo ""
    else
        echo ""
        print_success "Setup completed successfully!"
        echo ""
        echo "To run manually:"
        echo "  cd $INSTALL_DIR"
        echo "  source venv/bin/activate"
        echo "  python scheduler.py"
        echo ""
    fi
    
    echo "=================================================="
    echo "🎉 Installation Complete!"
    echo "=================================================="
}

# Run main installation
main