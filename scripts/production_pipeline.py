"""
Kronos Production Pipeline
Runs the full trading pipeline on a schedule
"""

import os
import sys
import json
import logging
import schedule
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.yahoo_finance import DataLoader
from signals.finbert_sentiment import NewsSentimentFetcher
from signals.macro_regime import MacroRegimeDetector
from strategy.kronos_alpha import KronosAlphaGenerator
from portfolio.optimizer import PositionOptimizer
from portfolio.risk_manager import RiskManager
from execution.paper_trading import PaperTradingBroker
from execution.alpaca_executor import AlpacaExecutor


class ProductionPipeline:
    """Production trading pipeline"""
    
    def __init__(self):
        self.mode = os.getenv('KRONOS_MODE', 'paper')
        self.data_source = os.getenv('DATA_SOURCE', 'yfinance')
        self.universe = os.getenv('UNIVERSE', 'sp500')
        self.model_size = os.getenv('MODEL_SIZE', 'small')
        self.rebalance_freq = os.getenv('REBALANCE_FREQUENCY', 'weekly')
        
        # Initialize components
        self.data_loader = DataLoader()
        self.sentiment_fetcher = NewsSentimentFetcher()
        self.regime_detector = MacroRegimeDetector()
        self.alpha_generator = KronosAlphaGenerator(model_size=self.model_size)
        self.optimizer = PositionOptimizer()
        self.risk_manager = RiskManager()
        
        # Initialize broker based on mode
        if self.mode == 'paper':
            self.broker = AlpacaExecutor(paper=True)
        elif self.mode == 'live':
            self.broker = AlpacaExecutor(paper=False)
        else:
            self.broker = PaperTradingBroker()
        
        logger.info(f"Production pipeline initialized: mode={self.mode}, universe={self.universe}")
    
    def fetch_data(self) -> pd.DataFrame:
        """Fetch latest market data"""
        logger.info("Fetching market data...")
        
        if self.universe == 'sp500':
            tickers = self._get_sp500_tickers()
        elif self.universe == 'sp100':
            tickers = self._get_sp100_tickers()
        else:
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        
        # Fetch recent data (last 30 days)
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=30)
        
        data = self.data_loader.fetch_data(
            tickers=tickers,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        logger.info(f"Fetched data for {len(tickers)} tickers")
        return data
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate ensemble signals"""
        logger.info("Generating signals...")
        
        # 1. Kronos model signals
        kronos_signals = self.alpha_generator.generate_signals(data)
        
        # 2. FinBERT sentiment signals
        tickers = kronos_signals.index.tolist()
        sentiment_data = self.sentiment_fetcher.get_universe_sentiment(tickers)
        
        # 3. Macro regime detection
        regime = self.regime_detector.detect_regime()
        
        # 4. Combine signals with adaptive weights
        ensemble_signals = self._combine_signals(
            kronos_signals,
            sentiment_data,
            regime
        )
        
        logger.info(f"Generated ensemble signals for {len(ensemble_signals)} tickers")
        logger.info(f"Current regime: {regime['regime']}")
        
        return ensemble_signals
    
    def optimize_portfolio(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize portfolio positions"""
        logger.info("Optimizing portfolio...")
        
        # Get current positions
        current_positions = self.broker.get_positions()
        
        # Risk constraints based on regime
        regime = self.regime_detector.detect_regime()
        max_gross_exposure = regime.get('max_gross_exposure', 1.0)
        max_turnover = regime.get('max_turnover', 0.3)
        
        # Optimize positions
        target_positions = self.optimizer.optimize(
            signals=signals,
            current_positions=current_positions,
            data=data,
            max_gross_exposure=max_gross_exposure,
            max_turnover=max_turnover
        )
        
        # Apply risk management
        final_positions = self.risk_manager.apply_risk_constraints(
            target_positions,
            data
        )
        
        logger.info(f"Optimized {len(final_positions)} positions")
        return final_positions
    
    def execute_trades(self, positions: pd.DataFrame):
        """Execute trades through broker"""
        logger.info("Executing trades...")
        
        # Calculate orders
        current_positions = self.broker.get_positions()
        orders = self._calculate_orders(current_positions, positions)
        
        if not orders:
            logger.info("No trades to execute")
            return
        
        # Submit orders
        for order in orders:
            try:
                self.broker.submit_order(order)
                logger.info(f"Executed: {order['ticker']} {order['side']} {order['qty']}")
            except Exception as e:
                logger.error(f"Failed to execute {order['ticker']}: {e}")
        
        logger.info(f"Executed {len(orders)} trades")
    
    def save_state(self, positions: pd.DataFrame, signals: pd.DataFrame):
        """Save pipeline state"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save positions
        positions.to_parquet(f'data/cache/positions_{timestamp}.parquet')
        
        # Save signals
        signals.to_parquet(f'data/cache/signals_{timestamp}.parquet')
        
        # Save portfolio snapshot
        portfolio_value = self.broker.get_portfolio_value()
        snapshot = {
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'num_positions': len(positions),
            'exposure': positions['weight'].abs().sum(),
            'regime': self.regime_detector.detect_regime()['regime']
        }
        
        with open(f'data/cache/snapshot_{timestamp}.json', 'w') as f:
            json.dump(snapshot, f, indent=2)
        
        logger.info(f"Saved state at {timestamp}")
    
    def run(self):
        """Run full pipeline"""
        logger.info("=" * 50)
        logger.info("Starting production pipeline run")
        logger.info("=" * 50)
        
        try:
            # 1. Fetch data
            data = self.fetch_data()
            
            # 2. Generate signals
            signals = self.generate_signals(data)
            
            # 3. Optimize portfolio
            positions = self.optimize_portfolio(signals, data)
            
            # 4. Execute trades
            self.execute_trades(positions)
            
            # 5. Save state
            self.save_state(positions, signals)
            
            logger.info("Production pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            self._send_alert(f"Pipeline failed: {e}")
    
    def _combine_signals(self, kronos: pd.DataFrame, sentiment: pd.DataFrame, 
                        regime: Dict) -> pd.DataFrame:
        """Combine signals with adaptive weights"""
        # Get adaptive weights based on regime
        weights = self._get_regime_weights(regime['regime'])
        
        # Merge signals
        combined = kronos.copy()
        combined['kronos_signal'] = combined['signal']
        
        # Add sentiment (fill missing with 0)
        if 'sentiment_score' in sentiment.columns:
            combined['sentiment_signal'] = sentiment['sentiment_score']
            combined['sentiment_signal'] = combined['sentiment_signal'].fillna(0)
        else:
            combined['sentiment_signal'] = 0
        
        # Add macro signal (all stocks get same macro bias)
        macro_signal = regime.get('equity_bias', 0)
        combined['macro_signal'] = macro_signal
        
        # Weighted ensemble
        combined['ensemble_signal'] = (
            weights['kronos'] * combined['kronos_signal'] +
            weights['sentiment'] * combined['sentiment_signal'] +
            weights['macro'] * combined['macro_signal']
        )
        
        return combined[['ensemble_signal', 'kronos_signal', 'sentiment_signal', 'macro_signal']]
    
    def _get_regime_weights(self, regime: str) -> Dict:
        """Get signal weights based on regime"""
        weights = {
            'expansion': {'kronos': 0.5, 'sentiment': 0.3, 'macro': 0.2},
            'recovery': {'kronos': 0.5, 'sentiment': 0.3, 'macro': 0.2},
            'contraction': {'kronos': 0.4, 'sentiment': 0.2, 'macro': 0.4},
            'downturn': {'kronos': 0.3, 'sentiment': 0.2, 'macro': 0.5},
            'neutral': {'kronos': 0.5, 'sentiment': 0.3, 'macro': 0.2}
        }
        return weights.get(regime, weights['neutral'])
    
    def _get_sp500_tickers(self) -> List[str]:
        """Get S&P 500 tickers"""
        try:
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            return table['Symbol'].tolist()
        except:
            # Fallback to common tickers
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 
                   'JPM', 'V', 'UNH', 'JNJ', 'XOM', 'WMT', 'PG', 'MA', 'HD', 'CVX']
    
    def _get_sp100_tickers(self) -> List[str]:
        """Get S&P 100 tickers"""
        try:
            table = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')[0]
            return table['Symbol'].tolist()
        except:
            return self._get_sp500_tickers()[:100]
    
    def _calculate_orders(self, current: pd.DataFrame, target: pd.DataFrame) -> List[Dict]:
        """Calculate orders needed to reach target positions"""
        orders = []
        
        # Get all tickers
        all_tickers = set(current.index.tolist() + target.index.tolist())
        
        for ticker in all_tickers:
            current_qty = current.get(ticker, {}).get('qty', 0)
            target_qty = target.get(ticker, {}).get('shares', 0) if ticker in target else 0
            
            delta = target_qty - current_qty
            
            if abs(delta) > 0:
                orders.append({
                    'ticker': ticker,
                    'side': 'buy' if delta > 0 else 'sell',
                    'qty': abs(int(delta)),
                    'type': 'market'
                })
        
        return orders
    
    def _send_alert(self, message: str):
        """Send alert notification"""
        webhook = os.getenv('SLACK_WEBHOOK_URL')
        if webhook:
            import requests
            requests.post(webhook, json={'text': f'🚨 Kronos Alert: {message}'})
        
        logger.warning(f"ALERT: {message}")


def scheduled_run():
    """Run pipeline on schedule"""
    pipeline = ProductionPipeline()
    pipeline.run()


def main():
    """Main entry point"""
    # Setup schedule
    rebalance_freq = os.getenv('REBALANCE_FREQUENCY', 'weekly')
    
    if rebalance_freq == 'daily':
        schedule.every().day.at("09:30").do(scheduled_run)
    elif rebalance_freq == 'weekly':
        schedule.every().monday.at("09:30").do(scheduled_run)
    elif rebalance_freq == 'monthly':
        schedule.every().month.at("09:30").do(scheduled_run)
    
    logger.info(f"Scheduler started: {rebalance_freq} rebalancing")
    
    # Run immediately on startup
    scheduled_run()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == '__main__':
    main()
