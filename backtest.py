"""
Backtesting System for Bitcoin Predictor
Validates prediction strategy before live trading
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from config import BACKTEST_CONFIG, get_timeframe_label, get_timeframe_category
from timezone_utils import now_iso_wib

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Backtest result data class"""
    timeframe_minutes: int
    total_predictions: int
    wins: int
    losses: int
    win_rate: float
    avg_error: float
    avg_error_pct: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown: float
    passed: bool
    period_days: int


class BacktestEngine:
    """Backtesting engine for prediction validation"""
    
    def __init__(self, predictor, firebase_manager=None):
        self.predictor = predictor
        self.firebase = firebase_manager
        self.results = {}
        
        logger.info("📊 Backtest engine initialized")
    
    def run_backtest(self, df: pd.DataFrame, timeframe_minutes: int,
                     periods: List[int] = None) -> Dict[int, BacktestResult]:
        """
        Run backtest for specified periods
        
        Args:
            df: Historical data
            timeframe_minutes: Timeframe to test
            periods: List of periods in days to test (default from config)
        
        Returns:
            Dict of period -> BacktestResult
        """
        if periods is None:
            periods = BACKTEST_CONFIG['backtest_periods']
        
        label = get_timeframe_label(timeframe_minutes)
        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 BACKTESTING {label}")
        logger.info(f"{'='*80}")
        
        results = {}
        
        for period in periods:
            logger.info(f"\n📈 Testing {period}-day period...")
            result = self._backtest_period(df, timeframe_minutes, period)
            results[period] = result
            
            # Log result
            status = "✅ PASSED" if result.passed else "❌ FAILED"
            logger.info(f"{status} - Win Rate: {result.win_rate:.1f}% "
                       f"({result.wins}/{result.total_predictions})")
        
        # Store results
        self.results[timeframe_minutes] = results
        
        # Save to Firebase
        if self.firebase:
            self._save_backtest_results(timeframe_minutes, results)
        
        return results
    
    def _backtest_period(self, df: pd.DataFrame, timeframe_minutes: int,
                        period_days: int) -> BacktestResult:
        """Run backtest for a single period"""
        
        # Prepare data
        df = df.copy()
        df = df.sort_values('datetime', ascending=True).reset_index(drop=True)
        
        # Get data for period
        cutoff_date = df.iloc[-1]['datetime'] - timedelta(days=period_days)
        test_df = df[df['datetime'] >= cutoff_date].reset_index(drop=True)
        
        if len(test_df) < 50:
            logger.warning(f"⚠️ Insufficient test data: {len(test_df)} points")
            return self._empty_result(timeframe_minutes, period_days)
        
        # Run predictions on historical data
        predictions = []
        actuals = []
        errors = []
        
        # Calculate how many predictions we can make
        prediction_interval = max(1, timeframe_minutes // 60)  # Hours
        
        for i in range(0, len(test_df) - timeframe_minutes, prediction_interval):
            try:
                # Get data up to this point
                train_data = test_df.iloc[:i+1].copy()
                
                if len(train_data) < 100:  # Need minimum data
                    continue
                
                # Make prediction
                prediction = self.predictor.predict(train_data, timeframe_minutes)
                
                if not prediction:
                    continue
                
                # Get actual future price
                future_idx = min(i + timeframe_minutes, len(test_df) - 1)
                actual_price = test_df.iloc[future_idx]['price']
                
                predictions.append(prediction)
                actuals.append(actual_price)
                
                # Calculate error
                error = abs(actual_price - prediction['predicted_price'])
                errors.append(error)
                
            except Exception as e:
                logger.debug(f"Prediction error at index {i}: {e}")
                continue
        
        if len(predictions) < 10:
            logger.warning(f"⚠️ Too few predictions: {len(predictions)}")
            return self._empty_result(timeframe_minutes, period_days)
        
        # Calculate metrics
        wins = 0
        losses = 0
        returns = []
        
        for pred, actual in zip(predictions, actuals):
            predicted_price = pred['predicted_price']
            current_price = pred['current_price']
            
            # Determine if prediction was correct
            predicted_direction = 'up' if predicted_price > current_price else 'down'
            actual_direction = 'up' if actual > current_price else 'down'
            
            if predicted_direction == actual_direction:
                wins += 1
                # Simulate profit
                ret = abs(actual - current_price) / current_price
                returns.append(ret)
            else:
                losses += 1
                # Simulate loss
                ret = -abs(actual - current_price) / current_price
                returns.append(ret)
        
        total_predictions = wins + losses
        win_rate = (wins / total_predictions * 100) if total_predictions > 0 else 0
        
        avg_error = np.mean(errors) if errors else 0
        avg_error_pct = (avg_error / np.mean([p['current_price'] for p in predictions]) * 100) if predictions else 0
        
        # Calculate advanced metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        profit_factor = self._calculate_profit_factor(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        
        # Determine if passed
        min_winrate = BACKTEST_CONFIG.get('min_backtest_winrate', 52.0)
        min_samples = BACKTEST_CONFIG.get('backtest_sample_size', 100)
        
        passed = (win_rate >= min_winrate and 
                 total_predictions >= min_samples and
                 sharpe_ratio > 0.5)
        
        return BacktestResult(
            timeframe_minutes=timeframe_minutes,
            total_predictions=total_predictions,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            avg_error=avg_error,
            avg_error_pct=avg_error_pct,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            passed=passed,
            period_days=period_days
        )
    
    def _empty_result(self, timeframe_minutes: int, period_days: int) -> BacktestResult:
        """Return empty result for failed backtest"""
        return BacktestResult(
            timeframe_minutes=timeframe_minutes,
            total_predictions=0,
            wins=0,
            losses=0,
            win_rate=0.0,
            avg_error=0.0,
            avg_error_pct=0.0,
            sharpe_ratio=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            passed=False,
            period_days=period_days
        )
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        if returns_array.std() == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        sharpe = (returns_array.mean() / returns_array.std()) * np.sqrt(252)
        return float(sharpe)
    
    def _calculate_profit_factor(self, returns: List[float]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not returns:
            return 0.0
        
        profits = sum([r for r in returns if r > 0])
        losses = abs(sum([r for r in returns if r < 0]))
        
        if losses == 0:
            return float('inf') if profits > 0 else 0.0
        
        return profits / losses
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not returns:
            return 0.0
        
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max)
        
        return float(abs(drawdown.min())) if len(drawdown) > 0 else 0.0
    
    def _save_backtest_results(self, timeframe_minutes: int, 
                               results: Dict[int, BacktestResult]):
        """Save backtest results to Firebase"""
        try:
            from config import FIREBASE_COLLECTIONS
            
            collection = self.firebase.firestore_db.collection(
                FIREBASE_COLLECTIONS['backtest_results']
            )
            
            for period, result in results.items():
                doc_data = {
                    'timestamp': now_iso_wib(),
                    'timeframe_minutes': result.timeframe_minutes,
                    'timeframe_label': get_timeframe_label(result.timeframe_minutes),
                    'category': get_timeframe_category(result.timeframe_minutes),
                    'period_days': result.period_days,
                    'total_predictions': result.total_predictions,
                    'wins': result.wins,
                    'losses': result.losses,
                    'win_rate': result.win_rate,
                    'avg_error': result.avg_error,
                    'avg_error_pct': result.avg_error_pct,
                    'sharpe_ratio': result.sharpe_ratio,
                    'profit_factor': result.profit_factor,
                    'max_drawdown': result.max_drawdown,
                    'passed': result.passed
                }
                
                collection.add(doc_data)
            
            logger.info("✅ Backtest results saved to Firebase")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save backtest results: {e}")
    
    def print_backtest_summary(self, timeframe_minutes: int):
        """Print backtest summary"""
        if timeframe_minutes not in self.results:
            logger.warning(f"No backtest results for {get_timeframe_label(timeframe_minutes)}")
            return
        
        results = self.results[timeframe_minutes]
        label = get_timeframe_label(timeframe_minutes)
        
        print(f"\n{'='*80}")
        print(f"📊 BACKTEST SUMMARY - {label}")
        print(f"{'='*80}")
        
        for period, result in results.items():
            status = "✅ PASSED" if result.passed else "❌ FAILED"
            
            print(f"\n{period}-Day Period: {status}")
            print(f"  Total Predictions: {result.total_predictions}")
            print(f"  Win Rate:          {result.win_rate:.2f}% ({result.wins}W / {result.losses}L)")
            print(f"  Avg Error:         ${result.avg_error:.2f} ({result.avg_error_pct:.2f}%)")
            print(f"  Sharpe Ratio:      {result.sharpe_ratio:.3f}")
            print(f"  Profit Factor:     {result.profit_factor:.3f}")
            print(f"  Max Drawdown:      {result.max_drawdown:.2%}")
        
        print(f"{'='*80}\n")
    
    def should_enable_timeframe(self, timeframe_minutes: int) -> bool:
        """
        Check if timeframe should be enabled based on backtest
        
        Returns:
            True if passed backtest, False otherwise
        """
        if not BACKTEST_CONFIG.get('enable_backtesting'):
            return True  # Backtesting disabled, allow all
        
        if timeframe_minutes not in self.results:
            return False  # Not tested yet
        
        results = self.results[timeframe_minutes]
        
        # Check if majority of periods passed
        passed_count = sum(1 for r in results.values() if r.passed)
        total_periods = len(results)
        
        # Need at least 50% pass rate
        return passed_count / total_periods >= 0.5
    
    def get_best_timeframes(self, top_n: int = 3) -> List[Tuple[int, float]]:
        """
        Get best performing timeframes
        
        Returns:
            List of (timeframe_minutes, avg_winrate) tuples
        """
        performance = []
        
        for tf, results in self.results.items():
            if not results:
                continue
            
            # Calculate average win rate across periods
            avg_winrate = np.mean([r.win_rate for r in results.values()])
            performance.append((tf, avg_winrate))
        
        # Sort by win rate
        performance.sort(key=lambda x: x[1], reverse=True)
        
        return performance[:top_n]


def run_comprehensive_backtest(predictor, data_fetcher, 
                               timeframes: List[int],
                               firebase_manager=None) -> BacktestEngine:
    """
    Run comprehensive backtest for all timeframes
    
    Args:
        predictor: ML predictor instance
        data_fetcher: Function to fetch data
        timeframes: List of timeframes to test
        firebase_manager: Firebase manager instance
    
    Returns:
        BacktestEngine with results
    """
    logger.info(f"\n{'='*80}")
    logger.info("🧪 COMPREHENSIVE BACKTESTING")
    logger.info(f"{'='*80}")
    logger.info(f"Testing {len(timeframes)} timeframes...")
    
    engine = BacktestEngine(predictor, firebase_manager)
    
    for tf in timeframes:
        try:
            label = get_timeframe_label(tf)
            category = get_timeframe_category(tf)
            
            logger.info(f"\n📊 Fetching data for {label} ({category})...")
            
            # Fetch appropriate data
            from config import get_data_config_for_timeframe
            data_config = get_data_config_for_timeframe(tf)
            
            df = data_fetcher(
                days=data_config['days'] * 2,  # Extra data for testing
                interval=data_config['interval']
            )
            
            if df is None or len(df) < 200:
                logger.warning(f"⚠️ Insufficient data for {label}")
                continue
            
            # Run backtest
            results = engine.run_backtest(df, tf)
            
            # Print summary
            engine.print_backtest_summary(tf)
            
        except Exception as e:
            logger.error(f"❌ Backtest failed for {label}: {e}")
            continue
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("🏆 TOP PERFORMING TIMEFRAMES")
    print(f"{'='*80}")
    
    best = engine.get_best_timeframes(5)
    for i, (tf, winrate) in enumerate(best, 1):
        label = get_timeframe_label(tf)
        enabled = "✅" if engine.should_enable_timeframe(tf) else "❌"
        print(f"{i}. {enabled} {label:8} - {winrate:.2f}% win rate")
    
    print(f"{'='*80}\n")
    
    return engine