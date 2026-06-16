import pandas as pd
import numpy as np
import cvxpy as cp
from typing import Dict

def construct_portfolio(
    signals_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    ohlcv_dict: Dict[str, pd.DataFrame],
    risk_aversion: float = 1.0,
    l2_penalty: float = 0.5,
    prev_weights: Dict[str, float] = None,
    transaction_cost: float = 0.0005,  # 5 bps per side
    max_sector_exposure: float = 0.20,  # Max 20% gross exposure per sector
    liquidity_scaling: bool = True,     # Enable liquidity-based position sizing
    min_avg_volume: float = 1_000_000,  # Minimum 1M shares avg daily volume
    max_position_by_volume: float = 0.02,  # Max 2% of average daily volume
    use_kelly_criterion: bool = False,  # Use Kelly criterion instead of mean-variance
    kelly_fraction: float = 0.25,       # Fractional Kelly (quarter-Kelly recommended)
    portfolio_aum: float = 1_000_000,   # H-2: portfolio AUM in dollars for liquidity cap
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
            df = ohlcv_dict[t]
            # L-4: guard against missing timestamps column to avoid KeyError
            if 'timestamps' not in df.columns:
                continue
            df = df.set_index('timestamps')
            price_series[t] = df['close']
            
    price_matrix = pd.DataFrame(price_series).ffill().dropna()
    
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
    liquidity_caps = {}
    for t in valid_tickers:
        # Base volatility cap
        vol = volatility.get(t, 0.2) 
        dynamic_cap = min(0.05, 0.01 / max(vol, 0.01))
        
        # Liquidity-based position sizing
        if liquidity_scaling and t in ohlcv_dict:
            df = ohlcv_dict[t]
            if 'volume' in df.columns and len(df) >= 20:
                avg_volume = df['volume'].tail(20).mean()
                avg_price = df['close'].tail(20).mean()

                # Skip illiquid stocks entirely
                if avg_volume < min_avg_volume:
                    print(f"   [Liquidity] Excluding {t}: avg volume {avg_volume:,.0f} < {min_avg_volume:,.0f}")
                    bounds.append((0, 0))  # Zero position
                    continue

                # H-2 FIX: cap = (max_pct_adv * ADV * price) / portfolio_AUM
                # This correctly normalises the dollar volume cap to a portfolio weight.
                # Old code divided by $1 (unit portfolio), making the cap non-binding.
                liquidity_cap = (max_position_by_volume * avg_volume * avg_price) / portfolio_aum
                liquidity_caps[t] = min(dynamic_cap, liquidity_cap)
                bounds.append((-liquidity_cap, liquidity_cap))
                print(f"   [Liquidity] {t}: cap {liquidity_cap:.4f} (adv: {avg_volume:,.0f}, price: {avg_price:.2f})")
            else:
                bounds.append((-dynamic_cap, dynamic_cap))
        else:
            bounds.append((-dynamic_cap, dynamic_cap))
        
    mu = np.array([signals_df.loc[t, 'Predicted_Return'] for t in valid_tickers])
    
    # Build sector mapping for constraints
    sector_map = {}
    for i, t in enumerate(valid_tickers):
        sector = fundamentals_df.loc[t, 'Sector'] if t in fundamentals_df.index else 'Unknown'
        if sector not in sector_map:
            sector_map[sector] = []
        sector_map[sector].append(i)
    
    # === CVXPY CONVEX OPTIMIZATION ===
    # Stanford OSQP/ECOS solvers handle absolute values and 300+ variables natively
    n = len(valid_tickers)
    w = cp.Variable(n)
    
    if use_kelly_criterion:
        # === KELLY CRITERION OBJECTIVE ===
        # Kelly maximizes: w'μ - 0.5 * w'Σw (log-utility approximation)
        # This is equivalent to mean-variance with risk_aversion = 1.0
        print(f"   [Optimizer] Using Kelly Criterion (fractional: {kelly_fraction}x)")
        
        port_return = mu @ w
        port_variance = cp.quad_form(w, cov_matrix.values)
        
        # Kelly objective: maximize expected log wealth
        # Approximation: μ'w - 0.5 * w'Σw
        kelly_objective = port_return - 0.5 * port_variance
        
        # Apply fractional Kelly for safety
        if kelly_fraction != 1.0:
            # Scale down the Kelly-optimal weights via regularization
            kelly_reg = (1.0 - kelly_fraction) * cp.sum_squares(w)
        else:
            kelly_reg = 0
        
        # Turnover penalty
        if prev_weights is not None:
            prev_w_array = np.array([prev_weights.get(t, 0.0) for t in valid_tickers])
            turnover = cp.sum(cp.abs(w - prev_w_array))
            turnover_penalty = transaction_cost * turnover
        else:
            turnover_penalty = 0
        
        objective = cp.Maximize(kelly_objective - kelly_reg - turnover_penalty)
        
    else:
        # === MEAN-VARIANCE OBJECTIVE (Default) ===
        # Objective: Maximize return - risk_aversion * variance - l2_penalty * ||w||^2 - transaction_costs
        port_return = mu @ w
        port_variance = cp.quad_form(w, cov_matrix.values)
        l2_reg = l2_penalty * cp.sum_squares(w)
        
        # Turnover penalty: penalize deviation from previous weights (5 bps per side)
        if prev_weights is not None:
            prev_w_array = np.array([prev_weights.get(t, 0.0) for t in valid_tickers])
            turnover = cp.sum(cp.abs(w - prev_w_array))
            turnover_penalty = transaction_cost * turnover
            print(f"   [Optimizer] Including turnover penalty: {transaction_cost:.4f} per side")
        else:
            turnover_penalty = 0
            print(f"   [Optimizer] No previous weights, skipping turnover penalty")
        
        objective = cp.Maximize(port_return - risk_aversion * port_variance - l2_reg - turnover_penalty)
    
    # Constraints:
    # 1. Market neutral: sum(weights) = 0
    # 2. Long/Short bounds per asset
    # 3. Gross exposure limit: sum(|weights|) <= max_gross_exposure
    # 4. Sector exposure limits (max 20% gross per sector)
    constraints = [
        cp.sum(w) == 0,  # Market neutral
        cp.sum(cp.abs(w)) <= max_gross_exposure,  # Gross exposure limit
    ]
    
    # Per-asset bounds
    for i, (low, high) in enumerate(bounds):
        constraints.append(w[i] >= low)
        constraints.append(w[i] <= high)
    
    # Sector exposure constraints (max 20% gross per sector)
    for sector, indices in sector_map.items():
        sector_weights = cp.sum([cp.abs(w[i]) for i in indices])
        constraints.append(sector_weights <= max_sector_exposure)
        print(f"   [Optimizer] Sector {sector}: {len(indices)} assets, max exposure {max_sector_exposure:.1%}")
    
    prob = cp.Problem(objective, constraints)
    
    print(f"Running CVXPY Long/Short MVO solver for {n} assets...")
    # OSQP is fast for quadratic programs; ECOS is robust fallback
    result = prob.solve(solver=cp.OSQP, verbose=False)
    
    if result is None or w.value is None:
        # M-4 FIX: message said "equal weights" but code produces zeros → no positions.
        print("Warning: CVXPY optimization failed. Returning zero weights (no trade this step).")
        raw_weights = np.zeros(n)
    else:
        raw_weights = w.value
    
    # Debug: Check constraint satisfaction
    current_gross = np.sum(np.abs(raw_weights))
    print(f"   [DEBUG] CVXPY gross exposure: {current_gross:.4f} (target: {max_gross_exposure:.4f})")
    if current_gross > 0.0001:
        # Scale the weights proportionately to perfectly match our Gross Target
        normalization_factor = max_gross_exposure / current_gross
        # Only scale down if it overshot, or if you want strict 100% force it
        # A true neutral fund forces exactly 100% gross
        final_weights_array = raw_weights * normalization_factor
    else:
        final_weights_array = raw_weights
        
    final_weights = {}
    filtered_count = 0
    for i, t in enumerate(valid_tickers):
        weight_val = final_weights_array[i]
        if abs(weight_val) > 0.001: 
            final_weights[t] = weight_val
            filtered_count += 1
    
    # Debug: Check what happened to the weights
    final_gross = sum(abs(w) for w in final_weights.values())
    print(f"   [DEBUG] Positions after 0.001 filter: {filtered_count}/{n}, gross exposure: {final_gross:.4f}")
            
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
