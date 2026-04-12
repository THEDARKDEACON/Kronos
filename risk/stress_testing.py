"""
Stress Testing Module
Implements historical scenario analysis and hypothetical shock testing.
Includes 2008 Financial Crisis, COVID-19 Pandemic, and custom scenarios.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import warnings


class HistoricalScenario(Enum):
    """Pre-defined historical stress scenarios."""
    FINANCIAL_CRISIS_2008 = "2008_financial_crisis"
    FLASH_CRASH_2010 = "2010_flash_crash"
    EURO_DEBT_CRISIS = "2011_euro_crisis"
    TAPER_TANTRUM = "2013_taper_tantrum"
    CHINA_DEvaluation = "2015_china_devaluation"
    COVID_PANDEMIC = "2020_covid_pandemic"
    RUSSIA_UKRAINE = "2022_russia_ukraine"
    RATE_SHOCK_2022 = "2022_rate_shock"


class HypotheticalShock(Enum):
    """Hypothetical shock types."""
    EQUITY_CRASH = "equity_crash_30pct"
    RATE_SPIKE = "rate_spike_300bps"
    CREDIT_CRISIS = "credit_spread_widening_500bps"
    LIQUIDITY_FREEZE = "liquidity_crisis"
    CURRENCY_CRISIS = "currency_devaluation_20pct"
    INFLATION_SHOCK = "inflation_surprise_500bps"
    GEOPOLITICAL = "geopolitical_crisis"


@dataclass
class ScenarioResult:
    """Results from a stress test scenario."""
    scenario_name: str
    scenario_type: str  # 'historical' or 'hypothetical'
    portfolio_value_change: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    factor_exposures: Dict[str, float]
    sector_pnl: Dict[str, float]
    position_pnl: Dict[str, float]
    days_to_recovery: Optional[int] = None
    recovery_probability: float = 0.0


@dataclass
class HistoricalPeriod:
    """Defines a historical stress period."""
    name: str
    start_date: datetime
    end_date: datetime
    peak_date: Optional[datetime] = None
    trough_date: Optional[datetime] = None
    recovery_date: Optional[datetime] = None
    description: str = ""
    

class HistoricalScenarios:
    """Database of historical market stress periods."""
    
    SCENARIOS = {
        HistoricalScenario.FINANCIAL_CRISIS_2008: HistoricalPeriod(
            name="2008 Financial Crisis",
            start_date=datetime(2007, 10, 1),
            end_date=datetime(2009, 3, 31),
            peak_date=datetime(2007, 10, 9),
            trough_date=datetime(2009, 3, 6),
            description="Global financial crisis triggered by subprime mortgage collapse"
        ),
        HistoricalScenario.FLASH_CRASH_2010: HistoricalPeriod(
            name="2010 Flash Crash",
            start_date=datetime(2010, 5, 1),
            end_date=datetime(2010, 5, 31),
            peak_date=datetime(2010, 5, 6),
            trough_date=datetime(2010, 5, 6),
            description="May 6, 2010 intraday market crash and recovery"
        ),
        HistoricalScenario.EURO_DEBT_CRISIS: HistoricalPeriod(
            name="2011 European Debt Crisis",
            start_date=datetime(2011, 7, 1),
            end_date=datetime(2011, 12, 31),
            description="Sovereign debt crisis in Greece, Italy, Spain"
        ),
        HistoricalScenario.TAPER_TANTRUM: HistoricalPeriod(
            name="2013 Taper Tantrum",
            start_date=datetime(2013, 5, 1),
            end_date=datetime(2013, 9, 30),
            description="Bond market volatility after Fed signaled QE reduction"
        ),
        HistoricalScenario.CHINA_DEvaluation: HistoricalPeriod(
            name="2015 China Devaluation",
            start_date=datetime(2015, 8, 1),
            end_date=datetime(2015, 9, 30),
            description="Chinese yuan devaluation and market crash"
        ),
        HistoricalScenario.COVID_PANDEMIC: HistoricalPeriod(
            name="COVID-19 Pandemic",
            start_date=datetime(2020, 2, 1),
            end_date=datetime(2020, 5, 31),
            peak_date=datetime(2020, 2, 19),
            trough_date=datetime(2020, 3, 23),
            recovery_date=datetime(2020, 8, 18),
            description="Global pandemic market crash and recovery"
        ),
        HistoricalScenario.RUSSIA_UKRAINE: HistoricalPeriod(
            name="Russia-Ukraine Conflict",
            start_date=datetime(2022, 2, 1),
            end_date=datetime(2022, 4, 30),
            description="Geopolitical crisis and commodity price shock"
        ),
        HistoricalScenario.RATE_SHOCK_2022: HistoricalPeriod(
            name="2022 Rate Shock",
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2022, 10, 31),
            description="Aggressive Fed rate hikes to combat inflation"
        )
    }
    
    @classmethod
    def get_scenario(cls, scenario: HistoricalScenario) -> HistoricalPeriod:
        return cls.SCENARIOS.get(scenario)
    
    @classmethod
    def get_all_scenarios(cls) -> Dict[HistoricalScenario, HistoricalPeriod]:
        return cls.SCENARIOS


class HypotheticalScenarios:
    """Hypothetical shock scenario definitions."""
    
    SHOCKS = {
        HypotheticalShock.EQUITY_CRASH: {
            'equity_return': -0.30,
            'volatility_spike': 2.5,
            'correlation_spike': 0.95,
            'description': '30% equity market decline with volatility explosion'
        },
        HypotheticalShock.RATE_SPIKE: {
            'rate_change': 0.03,  # 300bps
            'duration_impact': -0.05,  # -5% per year of duration
            'credit_spread_widening': 0.02,
            'description': 'Aggressive rate hiking cycle'
        },
        HypotheticalShock.CREDIT_CRISIS: {
            'credit_spread_widening': 0.05,  # 500bps
            'default_rate': 0.10,
            'liquidity_factor': 0.30,
            'description': 'High yield credit crisis'
        },
        HypotheticalShock.LIQUIDITY_FREEZE: {
            'bid_ask_spread_multiplier': 5.0,
            'market_depth_reduction': 0.80,
            'correlation_spike': 0.90,
            'description': 'Market liquidity crisis'
        },
        HypotheticalShock.CURRENCY_CRISIS: {
            'currency_devaluation': -0.20,
            'inflation_shock': 0.05,
            'rate_hike': 0.05,
            'description': 'Emerging market currency crisis'
        },
        HypotheticalShock.INFLATION_SHOCK: {
            'inflation_surprise': 0.05,
            'rate_hike': 0.04,
            'growth_impact': -0.02,
            'description': 'Unexpected inflation surge'
        },
        HypotheticalShock.GEOPOLITICAL: {
            'oil_shock': 0.50,
            'equity_impact': -0.15,
            'flight_to_quality': 0.20,
            'correlation_spike': 0.85,
            'description': 'Major geopolitical conflict'
        }
    }
    
    @classmethod
    def get_shock(cls, shock: HypotheticalShock) -> Dict:
        return cls.SHOCKS.get(shock, {})


class StressTestEngine:
    """
    Main stress testing engine for portfolio risk assessment.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence = confidence_level
        self.historical_db = HistoricalScenarios()
        self.hypothetical_db = HypotheticalScenarios()
        self.results: List[ScenarioResult] = []
        
    def run_historical_scenario(self,
                                scenario: HistoricalScenario,
                                portfolio_weights: Dict[str, float],
                                price_history_fn: Callable[[datetime, datetime], Dict[str, pd.DataFrame]],
                                factor_exposures: Optional[Dict[str, Dict[str, float]]] = None) -> ScenarioResult:
        """
        Run stress test using historical market data.
        
        Args:
            scenario: Historical scenario to simulate
            portfolio_weights: Current portfolio weights
            price_history_fn: Function that returns price data for date range
            factor_exposures: Optional factor exposures by symbol
        """
        period = self.historical_db.get_scenario(scenario)
        if not period:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        print(f"[Stress Test] Running {period.name}")
        print(f"[Stress Test] Period: {period.start_date.date()} to {period.end_date.date()}")
        
        # Get historical price data
        try:
            price_data = price_history_fn(period.start_date, period.end_date)
        except Exception as e:
            warnings.warn(f"Could not fetch price data: {e}")
            price_data = {}
        
        # Calculate P&L for each position
        position_pnl = {}
        sector_pnl = {}
        total_pnl = 0.0
        
        for symbol, weight in portfolio_weights.items():
            if symbol in price_data and len(price_data[symbol]) > 0:
                prices = price_data[symbol]['close']
                if len(prices) > 1:
                    period_return = prices.iloc[-1] / prices.iloc[0] - 1
                else:
                    period_return = -0.20  # Assume -20% if no data
            else:
                # Use scenario-typical return based on historical average
                period_return = self._estimate_typical_return(scenario, symbol)
            
            position_pnl[symbol] = weight * period_return
            total_pnl += position_pnl[symbol]
            
            # Aggregate by sector if available
            # (Would need sector mapping in production)
        
        # Calculate drawdown during period
        if price_data:
            max_dd = self._calculate_historical_drawdown(price_data, portfolio_weights)
        else:
            max_dd = -0.30  # Conservative estimate
        
        # Factor exposure analysis
        if factor_exposures:
            portfolio_factor_pnl = self._calculate_factor_pnl(factor_exposures, scenario)
        else:
            portfolio_factor_pnl = {}
        
        result = ScenarioResult(
            scenario_name=period.name,
            scenario_type='historical',
            portfolio_value_change=total_pnl,
            max_drawdown=max_dd,
            var_95=self._estimate_var(scenario, total_pnl),
            cvar_95=self._estimate_cvar(scenario, total_pnl),
            factor_exposures=portfolio_factor_pnl,
            sector_pnl=sector_pnl,
            position_pnl=position_pnl,
            days_to_recovery=self._estimate_recovery_days(scenario, total_pnl),
            recovery_probability=self._estimate_recovery_probability(scenario, total_pnl)
        )
        
        self.results.append(result)
        
        print(f"[Stress Test] Portfolio impact: {total_pnl:.2%}")
        print(f"[Stress Test] Max drawdown: {max_dd:.2%}")
        
        return result
    
    def run_hypothetical_shock(self,
                              shock: HypotheticalShock,
                              portfolio_weights: Dict[str, float],
                              portfolio_factors: Optional[Dict[str, float]] = None,
                              sector_mapping: Optional[Dict[str, str]] = None) -> ScenarioResult:
        """
        Run hypothetical shock scenario.
        
        Args:
            shock: Type of shock to simulate
            portfolio_weights: Current portfolio weights
            portfolio_factors: Optional portfolio factor exposures
            sector_mapping: Optional sector mapping for targeted shocks
        """
        shock_params = self.hypothetical_db.get_shock(shock)
        if not shock_params:
            raise ValueError(f"Unknown shock: {shock}")
        
        print(f"[Stress Test] Running hypothetical shock: {shock.value}")
        print(f"[Stress Test] {shock_params.get('description', '')}")
        
        # Calculate impact based on shock parameters
        total_impact = 0.0
        position_pnl = {}
        
        if shock == HypotheticalShock.EQUITY_CRASH:
            equity_return = shock_params['equity_return']
            vol_multiplier = shock_params['volatility_spike']
            
            # Long positions lose more in crash
            for symbol, weight in portfolio_weights.items():
                impact = weight * equity_return * (1 if weight > 0 else 0.5)
                # Volatility drag on short positions
                if weight < 0:
                    impact += abs(weight) * vol_multiplier * 0.02  # Vol drag
                position_pnl[symbol] = impact
                total_impact += impact
        
        elif shock == HypotheticalShock.RATE_SPIKE:
            rate_change = shock_params['rate_change']
            duration_impact = shock_params['duration_impact']
            
            # Assume growth stocks = long duration, value = short duration
            for symbol, weight in portfolio_weights.items():
                if weight > 0:
                    # Growth stocks hurt more
                    impact = weight * duration_impact * 5  # High duration
                else:
                    # Shorts benefit
                    impact = weight * abs(duration_impact) * 2
                position_pnl[symbol] = impact
                total_impact += impact
        
        elif shock == HypotheticalShock.CREDIT_CRISIS:
            spread_widening = shock_params['credit_spread_widening']
            default_rate = shock_params['default_rate']
            
            # High yield sectors suffer more
            for symbol, weight in portfolio_weights.items():
                credit_exposure = 0.5  # Assume 50% credit sensitivity
                impact = -weight * (spread_widening * credit_exposure + default_rate * 0.2)
                position_pnl[symbol] = impact
                total_impact += impact
        
        else:
            # Generic shock: -15% for longs, +10% for shorts
            for symbol, weight in portfolio_weights.items():
                if weight > 0:
                    impact = -weight * 0.15
                else:
                    impact = abs(weight) * 0.10
                position_pnl[symbol] = impact
                total_impact += impact
        
        result = ScenarioResult(
            scenario_name=shock.value,
            scenario_type='hypothetical',
            portfolio_value_change=total_impact,
            max_drawdown=total_impact * 1.2,  # Assume some overshoot
            var_95=total_impact * 1.5,
            cvar_95=total_impact * 1.8,
            factor_exposures=portfolio_factors or {},
            sector_pnl={},
            position_pnl=position_pnl,
            days_to_recovery=None,
            recovery_probability=0.5 if abs(total_impact) < 0.20 else 0.3
        )
        
        self.results.append(result)
        
        print(f"[Stress Test] Estimated portfolio impact: {total_impact:.2%}")
        
        return result
    
    def run_monte_carlo_stress(self,
                              portfolio_weights: Dict[str, float],
                              n_simulations: int = 10000,
                              correlation_regime: str = 'stressed') -> pd.DataFrame:
        """
        Run Monte Carlo stress simulation.
        
        Args:
            portfolio_weights: Portfolio weights
            n_simulations: Number of Monte Carlo paths
            correlation_regime: 'normal' or 'stressed' (higher correlations)
        """
        symbols = list(portfolio_weights.keys())
        weights = np.array([portfolio_weights[s] for s in symbols])
        
        # Stressed market parameters
        if correlation_regime == 'stressed':
            mean_returns = np.full(len(symbols), -0.002)  # -0.2% daily
            volatilities = np.full(len(symbols), 0.03)   # 3% daily vol (48% annual)
            correlations = np.full((len(symbols), len(symbols)), 0.85)
            np.fill_diagonal(correlations, 1.0)
        else:
            mean_returns = np.full(len(symbols), 0.0003)
            volatilities = np.full(len(symbols), 0.015)
            correlations = np.eye(len(symbols)) * 0.3 + 0.7
        
        # Build covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlations
        
        # Monte Carlo simulation
        simulated_returns = np.random.multivariate_normal(
            mean_returns, cov_matrix, n_simulations
        )
        
        portfolio_returns = simulated_returns @ weights
        
        results_df = pd.DataFrame({
            'portfolio_return': portfolio_returns,
            'scenario': range(n_simulations)
        })
        
        return results_df
    
    def generate_stress_report(self) -> pd.DataFrame:
        """Generate comprehensive stress test report."""
        if not self.results:
            return pd.DataFrame()
        
        report_data = []
        for result in self.results:
            report_data.append({
                'Scenario': result.scenario_name,
                'Type': result.scenario_type,
                'Portfolio_Impact': f"{result.portfolio_value_change:.2%}",
                'Max_Drawdown': f"{result.max_drawdown:.2%}",
                'VaR_95': f"{result.var_95:.2%}",
                'CVaR_95': f"{result.cvar_95:.2%}",
                'Days_to_Recovery': result.days_to_recovery if result.days_to_recovery else 'N/A',
                'Recovery_Prob': f"{result.recovery_probability:.1%}"
            })
        
        return pd.DataFrame(report_data)
    
    def _estimate_typical_return(self, scenario: HistoricalScenario, symbol: str) -> float:
        """Estimate typical return for scenario based on historical data."""
        typical_returns = {
            HistoricalScenario.FINANCIAL_CRISIS_2008: -0.37,
            HistoricalScenario.COVID_PANDEMIC: -0.34,
            HistoricalScenario.FLASH_CRASH_2010: -0.10,
            HistoricalScenario.EURO_DEBT_CRISIS: -0.19,
            HistoricalScenario.RATE_SHOCK_2022: -0.25
        }
        return typical_returns.get(scenario, -0.20)
    
    def _calculate_historical_drawdown(self, price_data: Dict[str, pd.DataFrame], 
                                     weights: Dict[str, float]) -> float:
        """Calculate portfolio drawdown from historical price data."""
        # Build portfolio value series
        portfolio_values = []
        
        # Get common dates
        all_dates = set()
        for df in price_data.values():
            if 'timestamps' in df.columns:
                all_dates.update(df['timestamps'])
        
        sorted_dates = sorted(all_dates)
        
        for date in sorted_dates:
            value = 0.0
            for symbol, weight in weights.items():
                if symbol in price_data:
                    df = price_data[symbol]
                    price_row = df[df['timestamps'] == date]
                    if len(price_row) > 0:
                        # Normalize to start at 1.0
                        start_price = df['close'].iloc[0]
                        current_price = price_row['close'].iloc[0]
                        normalized_return = current_price / start_price - 1
                        value += weight * (1 + normalized_return)
            portfolio_values.append(value if value != 0 else 1.0)
        
        if not portfolio_values:
            return -0.30
        
        # Calculate max drawdown
        peak = 1.0
        max_dd = 0.0
        for val in portfolio_values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak
            max_dd = max(max_dd, dd)
        
        return -max_dd
    
    def _calculate_factor_pnl(self, factor_exposures: Dict[str, Dict[str, float]], 
                             scenario: HistoricalScenario) -> Dict[str, float]:
        """Calculate factor-based P&L attribution."""
        # Factor returns during each scenario (typical)
        factor_returns = {
            HistoricalScenario.FINANCIAL_CRISIS_2008: {
                'Market': -0.37, 'Size': -0.05, 'Value': 0.15, 
                'Momentum': -0.25, 'Quality': 0.10
            },
            HistoricalScenario.COVID_PANDEMIC: {
                'Market': -0.34, 'Size': -0.30, 'Value': -0.20,
                'Momentum': -0.35, 'Quality': -0.05
            }
        }
        
        scenario_factors = factor_returns.get(scenario, {})
        portfolio_factor_pnl = {}
        
        for symbol, exposures in factor_exposures.items():
            factor_pnl = 0.0
            for factor, exposure in exposures.items():
                if factor in scenario_factors:
                    factor_pnl += exposure * scenario_factors[factor]
            portfolio_factor_pnl[symbol] = factor_pnl
        
        return portfolio_factor_pnl
    
    def _estimate_var(self, scenario: HistoricalScenario, base_impact: float) -> float:
        """Estimate 95% VaR for scenario."""
        # VaR is typically 1.5x the mean impact in stress
        return base_impact * 1.5
    
    def _estimate_cvar(self, scenario: HistoricalScenario, base_impact: float) -> float:
        """Estimate 95% CVaR (Expected Shortfall) for scenario."""
        # CVaR is typically 1.8x the mean impact in stress
        return base_impact * 1.8
    
    def _estimate_recovery_days(self, scenario: HistoricalScenario, impact: float) -> Optional[int]:
        """Estimate days to recover to pre-crisis level."""
        recovery_times = {
            HistoricalScenario.FINANCIAL_CRISIS_2008: 1260,  # ~5 years
            HistoricalScenario.COVID_PANDEMIC: 126,  # ~6 months
            HistoricalScenario.FLASH_CRASH_2010: 1,  # Same day
            HistoricalScenario.TAPER_TANTRUM: 90,
            HistoricalScenario.CHINA_DEvaluation: 60
        }
        return recovery_times.get(scenario)
    
    def _estimate_recovery_probability(self, scenario: HistoricalScenario, impact: float) -> float:
        """Estimate probability of full recovery."""
        if impact > -0.30:
            return 0.95
        elif impact > -0.50:
            return 0.80
        else:
            return 0.50


class ReverseStressTest:
    """
    Reverse stress testing: Find scenarios that would cause insolvency.
    """
    
    def __init__(self, capital: float = 1.0, recovery_requirement: float = 0.50):
        self.capital = capital
        self.recovery_req = recovery_requirement
        
    def find_insolvency_scenarios(self,
                                 portfolio_weights: Dict[str, float],
                                 possible_shocks: List[HypotheticalShock],
                                 threshold: float = -0.50) -> List[Dict]:
        """
        Find combinations of shocks that would cause insolvency.
        
        Returns list of scenarios that would cause >50% loss.
        """
        insolvency_scenarios = []
        
        # Test individual shocks
        engine = StressTestEngine()
        for shock in possible_shocks:
            result = engine.run_hypothetical_shock(shock, portfolio_weights)
            if result.portfolio_value_change < threshold:
                insolvency_scenarios.append({
                    'shocks': [shock.value],
                    'impact': result.portfolio_value_change,
                    'description': f"Single shock: {shock.value}"
                })
        
        # Test combinations (simplified - just additive)
        for i, shock1 in enumerate(possible_shocks):
            for shock2 in possible_shocks[i+1:]:
                # Rough estimate of combined effect
                result1 = engine.run_hypothetical_shock(shock1, portfolio_weights)
                result2 = engine.run_hypothetical_shock(shock2, portfolio_weights)
                combined = result1.portfolio_value_change + result2.portfolio_value_change * 0.7  # Correlation factor
                
                if combined < threshold:
                    insolvency_scenarios.append({
                        'shocks': [shock1.value, shock2.value],
                        'impact': combined,
                        'description': f"Combined: {shock1.value} + {shock2.value}"
                    })
        
        return insolvency_scenarios
    
    def calculate_capital_buffer(self,
                                portfolio_weights: Dict[str, float],
                                target_confidence: float = 0.99) -> float:
        """
        Calculate required capital buffer to survive 99% of scenarios.
        """
        # Run Monte Carlo to find 99th percentile loss
        engine = StressTestEngine()
        mc_results = engine.run_monte_carlo_stress(portfolio_weights, n_simulations=10000)
        
        var_99 = mc_results['portfolio_return'].quantile(1 - target_confidence)
        
        # Capital buffer = 99% VaR
        buffer = abs(var_99)
        
        return buffer
