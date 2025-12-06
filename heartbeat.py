"""
Heartbeat System for Bitcoin Predictor
Maintains system status in Firebase for web dashboard monitoring
"""

import logging
import time
from datetime import datetime
from typing import Optional
import psutil
import os

logger = logging.getLogger(__name__)


class HeartbeatManager:
    """Manages system heartbeat and status updates to Firebase"""
    
    def __init__(self, firebase_manager, update_interval: int = 30):
        """
        Initialize heartbeat manager
        
        Args:
            firebase_manager: FirebaseManager instance
            update_interval: Seconds between heartbeat updates (default: 30)
        """
        self.firebase = firebase_manager
        self.update_interval = update_interval
        self.start_time = datetime.now()
        self.last_heartbeat = None
        self.heartbeat_count = 0
        
    def send_heartbeat(self, additional_data: Optional[dict] = None):
        """
        Send heartbeat status to Firebase
        
        Args:
            additional_data: Optional dict with additional status info
        """
        try:
            # Get system metrics
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            cpu_percent = process.cpu_percent(interval=0.1)
            
            # Calculate uptime
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            uptime_hours = uptime_seconds / 3600
            
            # Prepare heartbeat data
            heartbeat_data = {
                'status': 'online',
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': int(uptime_seconds),
                'uptime_hours': round(uptime_hours, 2),
                'memory_mb': round(memory_info.rss / 1024 / 1024, 2),
                'cpu_percent': round(cpu_percent, 2),
                'process_id': process.pid,
                'heartbeat_count': self.heartbeat_count,
                'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None
            }
            
            # Add additional data if provided
            if additional_data:
                heartbeat_data.update(additional_data)
            
            # Save to Firebase
            if self.firebase and self.firebase.connected:
                success = self._save_heartbeat(heartbeat_data)
                
                if success:
                    self.last_heartbeat = datetime.now()
                    self.heartbeat_count += 1
                    logger.debug(f"💓 Heartbeat #{self.heartbeat_count} sent successfully")
                    return True
                else:
                    logger.warning("⚠️ Failed to send heartbeat")
                    return False
            else:
                logger.warning("⚠️ Firebase not connected, cannot send heartbeat")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending heartbeat: {e}")
            return False
    
    def _save_heartbeat(self, heartbeat_data: dict) -> bool:
        """Save heartbeat to Firebase with retry"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Use a fixed document ID so we always update the same document
                doc_ref = self.firebase.firestore_db.collection('system_status').document('heartbeat')
                doc_ref.set(heartbeat_data, merge=True)
                return True
                
            except Exception as e:
                logger.warning(f"⚠️ Heartbeat save attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        return False
    
    def send_status_change(self, status: str, message: Optional[str] = None):
        """
        Send status change notification
        
        Args:
            status: Status string (e.g., 'starting', 'running', 'stopping', 'error')
            message: Optional status message
        """
        try:
            status_data = {
                'status': status,
                'timestamp': datetime.now().isoformat(),
                'message': message or f"System {status}"
            }
            
            if self.firebase and self.firebase.connected:
                doc_ref = self.firebase.firestore_db.collection('system_status').document('heartbeat')
                doc_ref.set(status_data, merge=True)
                logger.info(f"📢 Status change: {status} - {message}")
                
        except Exception as e:
            logger.error(f"❌ Error sending status change: {e}")
    
    def send_shutdown_signal(self):
        """Send shutdown signal to Firebase"""
        try:
            shutdown_data = {
                'status': 'offline',
                'timestamp': datetime.now().isoformat(),
                'message': 'System shutting down',
                'uptime_hours': round((datetime.now() - self.start_time).total_seconds() / 3600, 2),
                'total_heartbeats': self.heartbeat_count
            }
            
            if self.firebase and self.firebase.connected:
                doc_ref = self.firebase.firestore_db.collection('system_status').document('heartbeat')
                doc_ref.set(shutdown_data, merge=True)
                logger.info("💤 Shutdown signal sent to Firebase")
                
        except Exception as e:
            logger.error(f"❌ Error sending shutdown signal: {e}")
    
    def get_status_summary(self) -> dict:
        """Get current status summary"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            
            return {
                'status': 'online',
                'uptime_hours': round(uptime_seconds / 3600, 2),
                'memory_mb': round(memory_info.rss / 1024 / 1024, 2),
                'heartbeat_count': self.heartbeat_count,
                'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting status summary: {e}")
            return {'status': 'error', 'error': str(e)}


def cleanup_old_heartbeats(firebase_manager):
    """
    Cleanup old heartbeat records (older than 7 days)
    
    Args:
        firebase_manager: FirebaseManager instance
    """
    try:
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=7)
        
        collection = firebase_manager.firestore_db.collection('system_status')
        old_docs = collection.where('timestamp', '<', cutoff_date.isoformat()).limit(100).stream()
        
        count = 0
        for doc in old_docs:
            if doc.id != 'heartbeat':  # Don't delete the main heartbeat document
                doc.reference.delete()
                count += 1
        
        if count > 0:
            logger.info(f"🗑️ Cleaned up {count} old heartbeat records")
            
    except Exception as e:
        logger.warning(f"⚠️ Error cleaning up old heartbeats: {e}")