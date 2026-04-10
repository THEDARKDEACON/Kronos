import pandas as pd
import numpy as np
import cvxpy as cp
from typing import Dict

def construct_portfolio(
    signals_df: pd.DataFrame, 
    fundamentals_df: pd.DataFrame, 
    ohlcv_dict: Dict[str, pd.DataFrame],
    risk_aversion: float = 1.0,
    l2_penalty: float = 0.5
) -> pd.DataFrame:
    """
    Constructs a SOTA Market-Neutral Market Portfolio using Long/Short Optimization.
    Incorporates Dynamic Volatility Boundary Scaling and Macro VIX Tail Hedging.
    """
    if signals_df.empty or not ohlcv_dict:
        return pd.DataFrame()
        
    target_tickers = signals_df.sort_values(by='Predicted_Return', ascending=False).head(150).index.tolist()
    
    price_series = {}
    for t in target_tickers:
        if t in ohlcv_dict:
            df = ohlcv_dict[t].set_index('timestamps')
            price_series[t] = df['close']
            
    price_matrix = pd.DataFrame(price_series).fillna(method='ffill').dropna()
    
    if price_matrix.empty:
        return pd.DataFrame()

    daily_returns = price_matrix.pct_change().dropna()
    cov_matrix = daily_returns.cov() * 252 
    valid_tickers = cov_matrix.columns.tolist()
    
    # === MACRO TAIL RISK HEDGING (CIRCUIT BREAKER) ===
    # Calculate synthetic index volatility (how wild the entire market is swinging)
    synthetic_index = daily_returns.mean(axis=1) # Proxy for S&P 500
    market_volatility = synthetic_index.std() * np.sqrt(252)
    
    max_gross_exposure = 1.0
    vix_hedge_allocation = 0.0
    if market_volatility > 0.25:
        print(f"CRITICAL: Macro Systemic Volatility detected at {market_volatility:.2%}. Engaging Circuit Breaker.")
        max_gross_exposure = 0.97
        vix_hedge_allocation = 0.03
        print("Withheld 3% of capital for pure Volatility (VIX) Hedging.")
        
    # === DYNAMIC BOUNDARY SCALING (LONG / SHORT) ===
    recent_returns = daily_returns.tail(30)
    volatility = recent_returns.std() * np.sqrt(252)
    
    bounds = []
    for t in valid_tickers:
        vol = volatility.get(t, 0.2) 
        dynamic_cap = min(0.05, 0.01 / max(vol, 0.01))
        # Now allows SHORT selling up to the exact same dynamic limit
        bounds.append((-dynamic_cap, dynamic_cap)) 
        
    mu = np.array([signals_df.loc[t, 'Predicted_Return'] for t in valid_tickers])
    
    # === CVXPY CONVEX OPTIMIZATION ===
    # Stanford OSQP/ECOS solvers handle absolute values and 300+ variables natively
    n = len(valid_tickers)
    w = cp.Variable(n)
    
    # Objective: Maximize return - risk_aversion * variance - l2_penalty * ||w||^2
    port_return = mu @ w
    port_variance = cp.quad_form(w, cov_matrix.values)
    l2_reg = l2_penalty * cp.sum_squares(w)
    objective = cp.Maximize(port_return - risk_aversion * port_variance - l2_reg)
    
    # Constraints:
    # 1. Market neutral: sum(weights) = 0
    # 2. Long/Short bounds per asset
    # 3. Gross exposure limit: sum(|weights|) <= max_gross_exposure
    constraints = [
        cp.sum(w) == 0,  # Market neutral
        cp.sum(cp.abs(w)) <= max_gross_exposure,  # Gross exposure limit (CVXPY handles abs natively)
    ]
    for i, (low, high) in enumerate(bounds):
        constraints.append(w[i] >= low)
        constraints.append(w[i] <= high)
    
    prob = cp.Problem(objective, constraints)
    
    print(f"Running CVXPY Long/Short MVO solver for {n} assets...")
    # OSQP is fast for quadratic programs; ECOS is robust fallback
    result = prob.solve(solver=cp.OSQP, verbose=False)
    
    if result is None or w.value is None:
        print("Warning: CVXPY optimization failed, falling back to equal weights")
        raw_weights = np.zeros(n)
    else:
        raw_weights = w.value
    current_gross = np.sum(np.abs(raw_weights))
    if current_gross > 0.0001:
        # Scale the weights proportionately to perfectly match our Gross Target
        normalization_factor = max_gross_exposure / current_gross
        # Only scale down if it overshot, or if you want strict 100% force it
        # A true neutral fund forces exactly 100% gross
        final_weights_array = raw_weights * normalization_factor
    else:
        final_weights_array = raw_weights
        
    final_weights = {}
    for i, t in enumerate(valid_tickers):
        w = final_weights_array[i]
        if abs(w) > 0.001: 
            final_weights[t] = w
            
    port_df = pd.DataFrame([
        {
            'Ticker': t, 
            'Sector': fundamentals_df.loc[t, 'Sector'] if t in fundamentals_df.index else 'Unknown',
            'Weight': w, 
            'Position': 'LONG' if w > 0 else 'SHORT',
            'Predicted_Return': signals_df.loc[t, 'Predicted_Return']
        }
        for t, w in final_weights.items()
    ]).set_index('Ticker').sort_values(by='Weight', ascending=False)
    
    if vix_hedge_allocation > 0:
        hedge_df = pd.DataFrame([{
            'Ticker': 'VIXY', 'Sector': 'MACRO_HEDGE', 'Weight': vix_hedge_allocation, 
            'Position': 'LONG', 'Predicted_Return': 0.0
        }]).set_index('Ticker')
        port_df = pd.concat([hedge_df, port_df])
        
    return port_df

if __name__ == "__main__":
    pass
