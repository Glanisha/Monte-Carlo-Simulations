import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date, timedelta
from typing import List

# --- Configuration Constants ---
TRADING_DAYS_PER_YEAR = 252
DEFAULT_SIMULATIONS = 1000
DEFAULT_TIME_HORIZON_DAYS = 252  # 1 year
# Risk levels for Value at Risk (VaR) and Conditional VaR (CVaR/Expected Shortfall)
CONFIDENCE_LEVEL = 0.95

# --- Pydantic Data Models for API Request Body ---

class Asset(BaseModel):
    """Defines a single asset (stock) in the portfolio."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL').")
    weight: float = Field(..., gt=0, description="Allocation weight (must sum to 1.0 across all assets).")

class PortfolioInput(BaseModel):
    """Defines the input parameters for the Monte Carlo simulation."""
    initial_investment: float = Field(..., gt=0, description="Starting cash value of the portfolio.")
    assets: List[Asset] = Field(..., min_items=1, description="List of assets with their respective weights.")
    num_simulations: int = Field(DEFAULT_SIMULATIONS, gt=0, le=10000, description="Number of Monte Carlo paths to simulate.")
    time_horizon_days: int = Field(DEFAULT_TIME_HORIZON_DAYS, gt=0, le=5 * TRADING_DAYS_PER_YEAR, description="Time horizon in trading days (e.g., 252 for 1 year).")
    start_date: date = Field(date.today() - timedelta(days=365*3), description="Start date for historical data fetching (e.g., 3 years ago).")
    end_date: date = Field(date.today(), description="End date for historical data fetching (Today).")

# --- FastAPI Application Setup ---

app = FastAPI(
    title="Monte Carlo Portfolio Simulation API",
    description="A FastAPI endpoint to calculate future portfolio value and risk metrics using Monte Carlo methods (Geometric Brownian Motion with correlated returns).",
    version="1.0.0"
)

# --- Core Simulation Functions ---

def calculate_portfolio_stats(assets: List[Asset], start_date: date, end_date: date):
    """
    1. Fetches historical data.
    2. Calculates daily logarithmic returns.
    3. Calculates annualized mean returns (mu) and the daily covariance matrix (Sigma).
    """
    tickers = [asset.ticker for asset in assets]
    weights = np.array([asset.weight for asset in assets])

    # 1. Fetch data
    try:
        # Fetch data with auto_adjust=True to use the 'Close' column as the Adjusted Close price.
        raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
        
        if raw_data.empty:
            raise ValueError(f"No data returned for tickers: {tickers}. Check dates or symbols.")

        # Extract only the adjusted 'Close' prices
        if isinstance(raw_data.columns, pd.MultiIndex):
            # Case 1: Multiple tickers were downloaded (MultiIndex)
            data = raw_data['Close']
        elif 'Close' in raw_data.columns:
            # Case 2: Single ticker was downloaded (Single Index)
            data = raw_data['Close']
            if isinstance(data, pd.Series):
                 # Convert Series back to a DataFrame for consistency if only one ticker
                 # This handles the case where yfinance returns a Series for a single ticker
                 data = data.to_frame(name=tickers[0])
        else:
            # Should not happen with auto_adjust=True, but as a fallback
            raise ValueError(f"Could not find 'Close' column (Adjusted Price) in downloaded data for tickers: {tickers}.")

    except Exception as e:
        # Re-raise error if it's the expected HTTPException, otherwise wrap it.
        if isinstance(e, HTTPException):
             raise e
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {e}")

    # 2. Calculate logarithmic returns
    # Log returns: R_t = ln(P_t / P_{t-1})
    log_returns = np.log(data / data.shift(1)).dropna()

    if log_returns.empty:
        raise ValueError("Insufficient data to calculate returns. Need more than one trading day.")
    
    # Check if all tickers are present in the returns (some might fail to download)
    present_tickers = log_returns.columns.tolist()
    if len(present_tickers) != len(tickers):
        # Filter weights and tickers to match available data
        present_weights = []
        for asset in assets:
            if asset.ticker in present_tickers:
                present_weights.append(asset.weight)
        
        if not present_weights:
            raise ValueError("No valid ticker data was available for the specified dates/symbols.")
            
        weights = np.array(present_weights)
        weights /= np.sum(weights)  # Re-normalize weights
        tickers = present_tickers


    # 3. Calculate annualized mean returns (mu) and daily covariance matrix (Sigma)
    
    # Annualized mean return vector (mu)
    # Annualized mean = Daily mean * TRADING_DAYS_PER_YEAR
    mu = log_returns.mean() * TRADING_DAYS_PER_YEAR
    
    # Daily covariance matrix (Sigma)
    # We use daily covariance for the simulation step
    sigma_daily = log_returns.cov()

    return mu, sigma_daily, weights, tickers, data.iloc[-1].values


def run_monte_carlo(
    initial_investment: float,
    mu: pd.Series,
    sigma_daily: pd.DataFrame,
    weights: np.ndarray,
    tickers: List[str],
    current_prices: np.ndarray,
    num_simulations: int,
    time_horizon_days: int
):
    """
    Runs the Monte Carlo simulation using correlated Geometric Brownian Motion.
    """
    
    # 1. Prepare Daily Parameters
    # Convert annualized mu to daily drift (mu_daily) for the simulation
    # The drift term in the GBM formula: (mu - 0.5 * sigma^2) * dt
    dt = 1 / TRADING_DAYS_PER_YEAR
    mu_daily_vector = mu / TRADING_DAYS_PER_YEAR
    
    # The volatility (sigma) is the standard deviation used for the random component.
    # We will use the Cholesky decomposition of the covariance matrix for correlated random variables.
    
    # 2. Cholesky Decomposition
    try:
        # L matrix such that L * L.T = Sigma_daily
        L = np.linalg.cholesky(sigma_daily.values * dt)
    except np.linalg.LinAlgError:
        raise HTTPException(
            status_code=500,
            detail="Covariance matrix is not positive semi-definite. Data may be insufficient or co-linear."
        )

    num_assets = len(tickers)
    
    # Array to store the final portfolio values for all simulations
    final_portfolio_values = np.zeros(num_simulations)

    # 3. Simulation Loop
    for m in range(num_simulations):
        # Start a new path (simulation)
        
        # Array to store the price path for the current simulation
        price_path = np.zeros((time_horizon_days + 1, num_assets))
        price_path[0] = current_prices
        
        # Current portfolio value at the start (V_0)
        current_portfolio_value = initial_investment 
        
        # Calculate initial shares based on initial investment and weights
        initial_dollar_allocation = initial_investment * weights
        initial_shares = initial_dollar_allocation / current_prices
        
        
        for t in range(1, time_horizon_days + 1):
            
            # Generate independent standard normal random variables (Z_t)
            Z = np.random.standard_normal(num_assets)
            
            # Generate Correlated Price Changes (dW_t) using Cholesky factor L
            # dW = L * Z
            correlated_shocks = np.dot(L, Z)
            
            # Daily Returns (R_t) for correlated GBM
            # R_t = (mu_daily - 0.5 * sigma_daily^2) * dt + sigma_daily * dW_t
            # Since we are using the full covariance matrix approach, we use the following log-return approximation:
            
            # Log Return approximation for correlated assets:
            log_returns_daily = mu_daily_vector.values * dt + correlated_shocks
            
            # Price update: S_{t} = S_{t-1} * exp(R_t)
            price_path[t] = price_path[t-1] * np.exp(log_returns_daily)
            
            # Recalculate portfolio value based on the latest prices and initial shares
            current_portfolio_value = np.sum(initial_shares * price_path[t])
            
        final_portfolio_values[m] = current_portfolio_value

    return final_portfolio_values


# --- API Endpoint Definition ---

@app.post("/simulate_portfolio")
def simulate_portfolio(params: PortfolioInput):
    """
    Runs a Monte Carlo simulation for a given stock portfolio and returns key risk and return metrics.
    
    The simulation models future asset prices using Geometric Brownian Motion (GBM) 
    and incorporates asset correlation via the Cholesky Decomposition of the historical 
    covariance matrix.
    """
    
    # 1. Data Retrieval and Statistical Calculation
    try:
        mu, sigma_daily, weights, tickers, current_prices = calculate_portfolio_stats(
            params.assets, params.start_date, params.end_date
        )
    except (ValueError, HTTPException) as e:
        # Re-raise the HTTPException or return an error for ValueError
        if isinstance(e, HTTPException):
            return e
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Run Monte Carlo Simulation
    final_values = run_monte_carlo(
        initial_investment=params.initial_investment,
        mu=mu,
        sigma_daily=sigma_daily,
        weights=weights,
        tickers=tickers,
        current_prices=current_prices,
        num_simulations=params.num_simulations,
        time_horizon_days=params.time_horizon_days
    )

    # 3. Analyze Results and Calculate Risk Metrics
    
    # Sort the final values for VaR/CVaR calculation
    final_values_sorted = np.sort(final_values)
    
    # Expected Value (Mean)
    expected_value = np.mean(final_values)
    
    # Standard Deviation (Volatility of final outcomes)
    std_dev_value = np.std(final_values)
    
    # Value at Risk (VaR)
    # VaR at (1 - CONFIDENCE_LEVEL) percentile (e.g., 5th percentile for 95% confidence)
    # VaR gives the worst expected loss at a given confidence level.
    var_percentile = 1 - CONFIDENCE_LEVEL 
    var_value = np.percentile(final_values_sorted, var_percentile * 100)
    
    # Conditional Value at Risk (CVaR) or Expected Shortfall (ES)
    # CVaR is the expected loss *if* the VaR threshold is breached.
    cvar_breach_values = final_values_sorted[final_values_sorted <= var_value]
    cvar_value = np.mean(cvar_breach_values)
    
    # Probability of loss (Percentage of paths where final value < initial investment)
    prob_of_loss = np.sum(final_values < params.initial_investment) / params.num_simulations
    
    # 4. Format Output
    simulation_metrics = {
        "initial_investment": params.initial_investment,
        "time_horizon_days": params.time_horizon_days,
        "num_simulations": params.num_simulations,
        "expected_final_value": round(expected_value, 2),
        "median_final_value": round(np.median(final_values), 2),
        "standard_deviation": round(std_dev_value, 2),
        "value_at_risk_95_percentile": round(var_value, 2),
        "value_at_risk_95_loss": round(params.initial_investment - var_value, 2), # VaR as an absolute loss amount
        "conditional_VaR_expected_shortfall": round(cvar_value, 2),
        "probability_of_loss": round(prob_of_loss * 100, 2)
    }
    
    # To demonstrate the distribution, we return percentiles
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    percentile_results = {f"{p}th_percentile": round(np.percentile(final_values_sorted, p), 2) for p in percentiles}

    # Final Summary for the API response
    results = {
        "message": f"Successfully ran {params.num_simulations} Monte Carlo simulations over {params.time_horizon_days} trading days.",
        "input_summary": {
            "tickers": tickers,
            "weights": weights.tolist(),
            "current_prices": current_prices.tolist(),
            "annual_drift_mu": mu.apply(lambda x: round(x, 4)).to_dict(),
        },
        "simulation_metrics": simulation_metrics,
        "percentile_outcomes": percentile_results
    }
    
    # Note: For efficiency, we avoid returning the raw list of final_values, 
    # but you could uncomment the line below for full data visualization in a separate tool.
    # results["raw_final_values"] = final_values.tolist()

    return results

if __name__ == "__main__":
    import uvicorn
    # This block is for local testing via console.
    print("Starting Monte Carlo API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
