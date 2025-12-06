"""
Timezone Utilities for Bitcoin Predictor
Handles timezone conversions - ALL storage uses WIB (UTC+7)
"""

from datetime import datetime, timedelta
import pytz
import logging

logger = logging.getLogger(__name__)

# Local timezone: Asia/Jakarta (WIB - UTC+7)
LOCAL_TIMEZONE = pytz.timezone('Asia/Jakarta')
UTC_TIMEZONE = pytz.UTC


def get_local_now():
    """
    Get current datetime in WIB
    
    Returns:
        datetime: Current time in WIB
    """
    return datetime.now(LOCAL_TIMEZONE)


def get_utc_now():
    """
    Get current datetime in UTC
    
    Returns:
        datetime: Current time in UTC
    """
    return datetime.now(UTC_TIMEZONE)


def local_to_utc(local_dt):
    """
    Convert WIB datetime to UTC
    
    Args:
        local_dt: datetime object (naive or aware)
    
    Returns:
        datetime: UTC datetime
    """
    try:
        if local_dt.tzinfo is None:
            local_dt = LOCAL_TIMEZONE.localize(local_dt)
        
        utc_dt = local_dt.astimezone(UTC_TIMEZONE)
        return utc_dt
        
    except Exception as e:
        logger.error(f"Error converting local to UTC: {e}")
        return local_dt


def utc_to_local(utc_dt):
    """
    Convert UTC datetime to WIB
    
    Args:
        utc_dt: datetime object (naive or aware)
    
    Returns:
        datetime: WIB datetime
    """
    try:
        if utc_dt.tzinfo is None:
            utc_dt = UTC_TIMEZONE.localize(utc_dt)
        
        local_dt = utc_dt.astimezone(LOCAL_TIMEZONE)
        return local_dt
        
    except Exception as e:
        logger.error(f"Error converting UTC to local: {e}")
        return utc_dt


def format_local_datetime(dt, format_str='%Y-%m-%d %H:%M:%S'):
    """
    Format datetime as WIB time string
    
    Args:
        dt: datetime object
        format_str: Format string
    
    Returns:
        str: Formatted datetime string
    """
    try:
        if dt.tzinfo is None:
            dt = LOCAL_TIMEZONE.localize(dt)
        
        local_dt = dt.astimezone(LOCAL_TIMEZONE)
        return local_dt.strftime(format_str)
        
    except Exception as e:
        logger.error(f"Error formatting datetime: {e}")
        return str(dt)


def get_local_isoformat(dt=None):
    """
    Get ISO format string in WIB timezone
    IMPORTANT: This is what gets stored in Firebase
    
    Args:
        dt: datetime object (None = current time)
    
    Returns:
        str: ISO format datetime string with WIB timezone
    """
    try:
        if dt is None:
            dt = get_local_now()
        elif dt.tzinfo is None:
            dt = LOCAL_TIMEZONE.localize(dt)
        
        local_dt = dt.astimezone(LOCAL_TIMEZONE)
        return local_dt.isoformat()
        
    except Exception as e:
        logger.error(f"Error getting ISO format: {e}")
        return get_local_now().isoformat()


def parse_iso_to_local(iso_string):
    """
    Parse ISO format string to WIB datetime
    
    Args:
        iso_string: ISO format datetime string
    
    Returns:
        datetime: WIB datetime object
    """
    try:
        # Parse ISO string
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        
        # Convert to WIB
        if dt.tzinfo is None:
            dt = LOCAL_TIMEZONE.localize(dt)
        
        return dt.astimezone(LOCAL_TIMEZONE)
        
    except Exception as e:
        logger.error(f"Error parsing ISO string: {e}")
        return get_local_now()


def add_minutes_local(dt, minutes):
    """
    Add minutes to datetime and return in WIB
    
    Args:
        dt: datetime object
        minutes: Number of minutes to add
    
    Returns:
        datetime: New datetime in WIB
    """
    try:
        if dt.tzinfo is None:
            dt = LOCAL_TIMEZONE.localize(dt)
        
        new_dt = dt + timedelta(minutes=minutes)
        return new_dt.astimezone(LOCAL_TIMEZONE)
        
    except Exception as e:
        logger.error(f"Error adding minutes: {e}")
        return dt


def get_timezone_info():
    """
    Get current timezone information
    
    Returns:
        dict: Timezone information
    """
    now_local = get_local_now()
    now_utc = get_utc_now()
    
    return {
        'local_timezone': str(LOCAL_TIMEZONE),
        'local_time': now_local.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'utc_time': now_utc.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'offset_hours': now_local.utcoffset().total_seconds() / 3600,
        'is_dst': bool(now_local.dst())
    }


def display_timezone_info():
    """Display timezone information (for debugging)"""
    info = get_timezone_info()
    
    logger.info("=" * 80)
    logger.info("🌏 TIMEZONE INFORMATION")
    logger.info("=" * 80)
    logger.info(f"Local Timezone: {info['local_timezone']}")
    logger.info(f"UTC Offset:     +{info['offset_hours']:.0f} hours")
    logger.info(f"Local Time:     {info['local_time']}")
    logger.info(f"UTC Time:       {info['utc_time']}")
    logger.info("=" * 80)


def prepare_firebase_timestamp(dt=None):
    """
    Prepare datetime for Firebase storage
    CHANGED: Now stores in WIB instead of UTC
    
    Args:
        dt: datetime object (None = current time)
    
    Returns:
        str: ISO format string in WIB
    """
    try:
        if dt is None:
            dt = get_local_now()
        elif dt.tzinfo is None:
            dt = LOCAL_TIMEZONE.localize(dt)
        
        # Store as WIB
        wib_dt = dt.astimezone(LOCAL_TIMEZONE)
        return wib_dt.isoformat()
        
    except Exception as e:
        logger.error(f"Error preparing Firebase timestamp: {e}")
        return get_local_now().isoformat()


def format_firebase_timestamp(timestamp_str, format_str='%d/%m/%Y %H:%M:%S'):
    """
    Format Firebase timestamp string
    Since storage is already in WIB, just parse and format
    
    Args:
        timestamp_str: ISO format timestamp string from Firebase
        format_str: Desired format string
    
    Returns:
        str: Formatted datetime
    """
    try:
        dt = parse_iso_to_local(timestamp_str)
        return dt.strftime(format_str)
        
    except Exception as e:
        logger.error(f"Error formatting Firebase timestamp: {e}")
        return timestamp_str


def now_iso_wib():
    """
    Shorthand for current time in ISO format (WIB)
    
    Returns:
        str: Current time as ISO string in WIB
    """
    return get_local_isoformat()