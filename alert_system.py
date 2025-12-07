"""
Alert System for Bitcoin Predictor
Sends notifications via Telegram, Email, and logs to Firebase
"""

import logging
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, Dict
from enum import Enum

from config import ALERT_CONFIG, FIREBASE_COLLECTIONS
from timezone_utils import now_iso_wib

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self, firebase_manager=None):
        self.firebase = firebase_manager
        self.enabled = ALERT_CONFIG['enable_alerts']
        self.alerts_sent = 0
        self.last_alert_time = {}
        
        # Rate limiting (max 1 alert per type per 5 minutes)
        self.rate_limit_seconds = 300
        
        if self.enabled:
            logger.info("✅ Alert system initialized")
        else:
            logger.info("ℹ️ Alert system disabled")
    
    def send_alert(self, title: str, message: str, 
                   severity: AlertSeverity = AlertSeverity.INFO,
                   alert_type: str = "general") -> bool:
        """
        Send alert via configured channels
        
        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level
            alert_type: Type of alert for rate limiting
        """
        if not self.enabled:
            logger.debug(f"Alert disabled: {title}")
            return False
        
        # Rate limiting
        if not self._check_rate_limit(alert_type):
            logger.debug(f"Alert rate limited: {alert_type}")
            return False
        
        success = False
        
        try:
            # Format message with emoji
            emoji = self._get_emoji(severity)
            formatted_title = f"{emoji} {title}"
            
            # Log to Firebase
            if self.firebase:
                self._log_to_firebase(formatted_title, message, severity)
            
            # Send to Telegram
            if ALERT_CONFIG.get('telegram_bot_token'):
                if self._send_telegram(formatted_title, message, severity):
                    success = True
            
            # Send email for critical alerts
            if severity == AlertSeverity.CRITICAL and ALERT_CONFIG.get('alert_email'):
                if self._send_email(formatted_title, message):
                    success = True
            
            # Log locally
            if severity == AlertSeverity.CRITICAL:
                logger.critical(f"{formatted_title}: {message}")
            elif severity == AlertSeverity.WARNING:
                logger.warning(f"{formatted_title}: {message}")
            else:
                logger.info(f"{formatted_title}: {message}")
            
            if success:
                self.alerts_sent += 1
                self.last_alert_time[alert_type] = datetime.now()
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Alert failed: {e}")
            return False
    
    def _check_rate_limit(self, alert_type: str) -> bool:
        """Check if alert is rate limited"""
        if alert_type not in self.last_alert_time:
            return True
        
        elapsed = (datetime.now() - self.last_alert_time[alert_type]).total_seconds()
        return elapsed >= self.rate_limit_seconds
    
    def _get_emoji(self, severity: AlertSeverity) -> str:
        """Get emoji for severity"""
        emoji_map = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨"
        }
        return emoji_map.get(severity, "📢")
    
    def _send_telegram(self, title: str, message: str, 
                       severity: AlertSeverity) -> bool:
        """Send alert via Telegram"""
        try:
            bot_token = ALERT_CONFIG.get('telegram_bot_token')
            chat_id = ALERT_CONFIG.get('telegram_chat_id')
            
            if not bot_token or not chat_id:
                return False
            
            # Format message
            text = f"<b>{title}</b>\n\n{message}\n\n<i>{now_iso_wib()}</i>"
            
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.debug("✅ Telegram alert sent")
                return True
            else:
                logger.warning(f"⚠️ Telegram failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")
            return False
    
    def _send_email(self, title: str, message: str) -> bool:
        """Send alert via email"""
        try:
            email = ALERT_CONFIG.get('alert_email')
            smtp_server = ALERT_CONFIG.get('smtp_server')
            smtp_port = ALERT_CONFIG.get('smtp_port')
            username = ALERT_CONFIG.get('smtp_username')
            password = ALERT_CONFIG.get('smtp_password')
            
            if not all([email, smtp_server, username, password]):
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = email
            msg['Subject'] = f"Bitcoin Predictor Alert: {title}"
            
            body = f"""
            {title}
            
            {message}
            
            Time: {now_iso_wib()}
            
            ---
            This is an automated alert from Bitcoin Predictor
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            logger.debug("✅ Email alert sent")
            return True
            
        except Exception as e:
            logger.error(f"❌ Email error: {e}")
            return False
    
    def _log_to_firebase(self, title: str, message: str, 
                         severity: AlertSeverity):
        """Log alert to Firebase"""
        try:
            if not self.firebase or not self.firebase.connected:
                return
            
            collection = self.firebase.firestore_db.collection(
                FIREBASE_COLLECTIONS['alerts']
            )
            
            alert_data = {
                'timestamp': now_iso_wib(),
                'title': title,
                'message': message,
                'severity': severity.value,
                'alerts_sent_count': self.alerts_sent
            }
            
            collection.add(alert_data)
            
        except Exception as e:
            logger.debug(f"Firebase alert log failed: {e}")
    
    # Convenience methods for common alerts
    
    def alert_low_winrate(self, winrate: float, timeframe: str):
        """Alert on low win rate"""
        if not ALERT_CONFIG.get('alert_on_low_winrate'):
            return
        
        threshold = ALERT_CONFIG.get('min_winrate_alert', 45.0)
        if winrate < threshold:
            self.send_alert(
                "Low Win Rate Detected",
                f"Win rate for {timeframe} dropped to {winrate:.1f}% (threshold: {threshold}%)\n"
                f"Consider reviewing strategy or pausing predictions.",
                AlertSeverity.WARNING,
                f"low_winrate_{timeframe}"
            )
    
    def alert_high_memory(self, memory_mb: float, max_mb: float):
        """Alert on high memory usage"""
        if not ALERT_CONFIG.get('alert_on_high_memory'):
            return
        
        self.send_alert(
            "High Memory Usage",
            f"Memory usage: {memory_mb:.0f}MB / {max_mb:.0f}MB\n"
            f"System may restart to free memory.",
            AlertSeverity.WARNING,
            "high_memory"
        )
    
    def alert_consecutive_failures(self, count: int):
        """Alert on consecutive prediction failures"""
        if not ALERT_CONFIG.get('alert_on_consecutive_failures'):
            return
        
        max_failures = ALERT_CONFIG.get('max_consecutive_failures', 3)
        if count >= max_failures:
            self.send_alert(
                "Multiple Prediction Failures",
                f"{count} consecutive prediction failures detected.\n"
                f"System health check recommended.",
                AlertSeverity.CRITICAL,
                "consecutive_failures"
            )
    
    def alert_system_startup(self):
        """Alert on system startup"""
        self.send_alert(
            "System Started",
            "Bitcoin Predictor system has started successfully.",
            AlertSeverity.INFO,
            "startup"
        )
    
    def alert_system_shutdown(self, reason: str = "normal"):
        """Alert on system shutdown"""
        self.send_alert(
            "System Shutdown",
            f"Bitcoin Predictor is shutting down.\nReason: {reason}",
            AlertSeverity.WARNING if reason != "normal" else AlertSeverity.INFO,
            "shutdown"
        )
    
    def alert_model_retrain(self, success: bool, metrics: Optional[Dict] = None):
        """Alert on model retraining"""
        if success:
            msg = "Model retraining completed successfully."
            if metrics:
                msg += f"\n\nMetrics:\n"
                for model, model_metrics in metrics.items():
                    msg += f"\n{model.upper()}:\n"
                    for metric, value in model_metrics.items():
                        msg += f"  {metric}: {value:.4f}\n"
            
            self.send_alert(
                "Model Retrained",
                msg,
                AlertSeverity.INFO,
                "model_retrain"
            )
        else:
            self.send_alert(
                "Model Retraining Failed",
                "Failed to retrain models. System may use outdated models.",
                AlertSeverity.CRITICAL,
                "model_retrain_fail"
            )
    
    def alert_api_failure(self, api_name: str, error: str):
        """Alert on API failure"""
        self.send_alert(
            f"{api_name} API Failure",
            f"Failed to connect to {api_name}.\nError: {error}\n"
            f"Predictions may be affected.",
            AlertSeverity.WARNING,
            f"api_failure_{api_name}"
        )
    
    def alert_firebase_disconnected(self):
        """Alert on Firebase disconnection"""
        self.send_alert(
            "Firebase Disconnected",
            "Lost connection to Firebase. Data may not be saved.",
            AlertSeverity.CRITICAL,
            "firebase_disconnected"
        )
    
    def alert_disk_space_low(self, free_gb: float):
        """Alert on low disk space"""
        self.send_alert(
            "Low Disk Space",
            f"Only {free_gb:.2f}GB free disk space remaining.\n"
            f"Consider cleaning up old data.",
            AlertSeverity.WARNING,
            "disk_space_low"
        )
    
    def alert_backtest_failed(self, timeframe: str, winrate: float, threshold: float):
        """Alert on failed backtest"""
        self.send_alert(
            "Backtest Failed",
            f"Backtest for {timeframe} failed validation.\n"
            f"Win rate: {winrate:.1f}% (required: {threshold:.1f}%)\n"
            f"Predictions for this timeframe will be disabled.",
            AlertSeverity.WARNING,
            f"backtest_failed_{timeframe}"
        )
    
    def alert_good_performance(self, timeframe: str, winrate: float):
        """Alert on good performance"""
        if winrate >= 65.0:  # Only alert on exceptional performance
            self.send_alert(
                "Excellent Performance",
                f"🎉 {timeframe} achieved {winrate:.1f}% win rate!",
                AlertSeverity.INFO,
                f"good_performance_{timeframe}"
            )
    
    def get_alert_summary(self) -> Dict:
        """Get alert system summary"""
        return {
            'enabled': self.enabled,
            'total_alerts_sent': self.alerts_sent,
            'telegram_configured': bool(ALERT_CONFIG.get('telegram_bot_token')),
            'email_configured': bool(ALERT_CONFIG.get('alert_email')),
        }


# Convenience function
def get_alert_manager(firebase_manager=None) -> AlertManager:
    """Get or create alert manager instance"""
    return AlertManager(firebase_manager)