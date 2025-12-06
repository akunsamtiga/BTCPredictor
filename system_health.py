"""
System Health Monitoring for VPS
Monitors CPU, memory, disk, and network connectivity
ALL TIMESTAMPS IN WIB (UTC+7)
"""

import psutil
import logging
import os
import gc
from datetime import datetime
from typing import Dict, Optional
from config import HEALTH_CONFIG, VPS_CONFIG
from timezone_utils import get_local_now, now_iso_wib

logger = logging.getLogger(__name__)


class SystemHealthMonitor:
    """Monitor system health and resources"""
    
    def __init__(self):
        self.last_check = None
        self.alerts = []
        self.restart_count = 0
    
    def check_memory(self) -> Dict:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            status = {
                'total_mb': round(memory.total / 1024 / 1024, 2),
                'available_mb': round(memory.available / 1024 / 1024, 2),
                'used_percent': memory.percent,
                'process_memory_mb': round(process_memory, 2),
                'swap_percent': swap.percent,
                'status': 'OK'
            }
            
            # Check for alerts
            if process_memory > HEALTH_CONFIG['max_memory_mb']:
                status['status'] = 'WARNING'
                status['alert'] = f"Process memory ({process_memory:.0f}MB) exceeds limit"
                logger.warning(f"⚠️ High memory usage: {process_memory:.0f}MB")
                
                # Trigger garbage collection
                if VPS_CONFIG['enable_memory_optimization']:
                    self._optimize_memory()
            
            if memory.percent > 90:
                status['status'] = 'CRITICAL'
                status['alert'] = f"System memory critical: {memory.percent:.1f}%"
                logger.error(f"❌ Critical memory usage: {memory.percent:.1f}%")
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error checking memory: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def check_cpu(self) -> Dict:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            process = psutil.Process(os.getpid())
            process_cpu = process.cpu_percent(interval=1)
            
            status = {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'process_cpu_percent': process_cpu,
                'status': 'OK'
            }
            
            if cpu_percent > HEALTH_CONFIG['max_cpu_percent']:
                status['status'] = 'WARNING'
                status['alert'] = f"High CPU usage: {cpu_percent:.1f}%"
                logger.warning(f"⚠️ High CPU usage: {cpu_percent:.1f}%")
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error checking CPU: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def check_disk(self) -> Dict:
        """Check disk space"""
        try:
            disk = psutil.disk_usage('/')
            
            free_gb = disk.free / 1024 / 1024 / 1024
            
            status = {
                'total_gb': round(disk.total / 1024 / 1024 / 1024, 2),
                'used_gb': round(disk.used / 1024 / 1024 / 1024, 2),
                'free_gb': round(free_gb, 2),
                'used_percent': disk.percent,
                'status': 'OK'
            }
            
            if free_gb < HEALTH_CONFIG['disk_space_min_gb']:
                status['status'] = 'WARNING'
                status['alert'] = f"Low disk space: {free_gb:.2f}GB"
                logger.warning(f"⚠️ Low disk space: {free_gb:.2f}GB")
            
            if disk.percent > 90:
                status['status'] = 'CRITICAL'
                status['alert'] = f"Critical disk usage: {disk.percent:.1f}%"
                logger.error(f"❌ Critical disk usage: {disk.percent:.1f}%")
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error checking disk: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def check_network(self) -> Dict:
        """Check network connectivity"""
        try:
            import socket
            
            # Test DNS resolution
            socket.gethostbyname('google.com')
            
            # Test HTTPS connection
            import requests
            response = requests.get('https://www.google.com', timeout=5)
            
            status = {
                'dns_ok': True,
                'https_ok': response.status_code == 200,
                'status': 'OK' if response.status_code == 200 else 'WARNING'
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Network check failed: {e}")
            return {
                'dns_ok': False,
                'https_ok': False,
                'status': 'ERROR',
                'error': str(e)
            }
    
    def check_processes(self) -> Dict:
        """Check running processes"""
        try:
            process = psutil.Process(os.getpid())
            
            status = {
                'pid': process.pid,
                'status': process.status(),
                'num_threads': process.num_threads(),
                'create_time': datetime.fromtimestamp(process.create_time()).isoformat(),
                'uptime_hours': round((datetime.now().timestamp() - process.create_time()) / 3600, 2),
                'status': 'OK'
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error checking processes: {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def get_full_health_report(self) -> Dict:
        """Get comprehensive health report"""
        report = {
            'timestamp': now_iso_wib(),  # WIB
            'memory': self.check_memory(),
            'cpu': self.check_cpu(),
            'disk': self.check_disk(),
            'network': self.check_network(),
            'process': self.check_processes(),
        }
        
        # Determine overall status
        statuses = [
            report['memory']['status'],
            report['cpu']['status'],
            report['disk']['status'],
            report['network']['status']
        ]
        
        if 'CRITICAL' in statuses or 'ERROR' in statuses:
            report['overall_status'] = 'CRITICAL'
        elif 'WARNING' in statuses:
            report['overall_status'] = 'WARNING'
        else:
            report['overall_status'] = 'HEALTHY'
        
        self.last_check = get_local_now()
        
        return report
    
    def _optimize_memory(self):
        """Optimize memory usage"""
        try:
            logger.info("🧹 Running memory optimization...")
            
            # Force garbage collection
            gc.collect()
            
            # Clear TensorFlow session if enabled
            if VPS_CONFIG['clear_tensorflow_session']:
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                    logger.info("✅ TensorFlow session cleared")
                except:
                    pass
            
            logger.info("✅ Memory optimization completed")
            
        except Exception as e:
            logger.error(f"❌ Error optimizing memory: {e}")
    
    def should_restart(self) -> bool:
        """Check if system should restart"""
        if not HEALTH_CONFIG['auto_restart_on_error']:
            return False
        
        report = self.get_full_health_report()
        
        if report['overall_status'] == 'CRITICAL':
            if self.restart_count < HEALTH_CONFIG['max_auto_restarts']:
                logger.warning(f"⚠️ System critical, restart {self.restart_count + 1}/{HEALTH_CONFIG['max_auto_restarts']}")
                self.restart_count += 1
                return True
        
        return False
    
    def log_health_summary(self):
        """Log health summary"""
        report = self.get_full_health_report()
        
        logger.info("=" * 80)
        logger.info("🏥 SYSTEM HEALTH CHECK")
        logger.info("=" * 80)
        logger.info(f"Overall Status: {report['overall_status']}")
        logger.info(f"Memory: {report['memory']['process_memory_mb']:.0f}MB / {report['memory']['total_mb']:.0f}MB ({report['memory']['used_percent']:.1f}%)")
        logger.info(f"CPU: {report['cpu']['cpu_percent']:.1f}% (Process: {report['cpu']['process_cpu_percent']:.1f}%)")
        logger.info(f"Disk: {report['disk']['free_gb']:.2f}GB free ({report['disk']['used_percent']:.1f}% used)")
        logger.info(f"Network: {'✅ OK' if report['network']['status'] == 'OK' else '❌ ERROR'}")
        logger.info(f"Uptime: {report['process']['uptime_hours']:.1f} hours")
        logger.info("=" * 80)


def monitor_health(firebase_manager=None):
    """Standalone health monitoring function"""
    monitor = SystemHealthMonitor()
    report = monitor.get_full_health_report()
    
    # Log to console
    monitor.log_health_summary()
    
    # Save to Firebase if available
    if firebase_manager:
        try:
            firebase_manager.save_system_health(report)
        except Exception as e:
            logger.warning(f"⚠️ Could not save health report to Firebase: {e}")
    
    return report