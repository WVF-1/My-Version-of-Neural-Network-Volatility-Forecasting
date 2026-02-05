"""
Utility metrics and helper functions.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def annualize_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualize returns.
    
    Parameters
    ----------
    returns : pd.Series
        Period returns
    periods_per_year : int
        Number of periods per year (252 for daily, 12 for monthly)
        
    Returns
    -------
    float
        Annualized return
    """
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    n_years = n_periods / periods_per_year
    
    if n_years <= 0:
        return 0.0
    
    annualized = (1 + total_return) ** (1 / n_years) - 1
    return annualized


def annualize_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualize volatility.
    
    Parameters
    ----------
    returns : pd.Series
        Period returns
    periods_per_year : int
        Number of periods per year
        
    Returns
    -------
    float
        Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(returns: pd.Series,
                          risk_free_rate: float = 0.0,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Period returns
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Number of periods per year
        
    Returns
    -------
    float
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if len(excess_returns) == 0:
        return 0.0
    
    annualized_return = annualize_return(excess_returns, periods_per_year)
    annualized_vol = annualize_volatility(returns, periods_per_year)
    
    if annualized_vol == 0:
        return 0.0
    
    return annualized_return / annualized_vol


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Parameters
    ----------
    returns : pd.Series
        Period returns
        
    Returns
    -------
    float
        Maximum drawdown (negative value)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    return drawdown.min()


def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (return / max drawdown).
    
    Parameters
    ----------
    returns : pd.Series
        Period returns
    periods_per_year : int
        Number of periods per year
        
    Returns
    -------
    float
        Calmar ratio
    """
    annualized_return = annualize_return(returns, periods_per_year)
    max_dd = calculate_max_drawdown(returns)
    
    if max_dd >= 0:
        return 0.0
    
    return annualized_return / abs(max_dd)


def calculate_information_ratio(returns: pd.Series,
                                benchmark_returns: pd.Series,
                                periods_per_year: int = 252) -> float:
    """
    Calculate information ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Portfolio returns
    benchmark_returns : pd.Series
        Benchmark returns
    periods_per_year : int
        Number of periods per year
        
    Returns
    -------
    float
        Information ratio
    """
    # Align series
    common_idx = returns.index.intersection(benchmark_returns.index)
    returns = returns.loc[common_idx]
    benchmark_returns = benchmark_returns.loc[common_idx]
    
    # Active returns
    active_returns = returns - benchmark_returns
    
    # Tracking error
    tracking_error = active_returns.std() * np.sqrt(periods_per_year)
    
    if tracking_error == 0:
        return 0.0
    
    # Active return
    active_return = annualize_return(active_returns, periods_per_year)
    
    return active_return / tracking_error


def rolling_window_stats(series: pd.Series, 
                         window: int) -> pd.DataFrame:
    """
    Calculate rolling window statistics.
    
    Parameters
    ----------
    series : pd.Series
        Input series
    window : int
        Window size
        
    Returns
    -------
    pd.DataFrame
        Rolling statistics
    """
    stats = pd.DataFrame(index=series.index)
    
    stats['mean'] = series.rolling(window=window).mean()
    stats['std'] = series.rolling(window=window).std()
    stats['min'] = series.rolling(window=window).min()
    stats['max'] = series.rolling(window=window).max()
    stats['median'] = series.rolling(window=window).median()
    
    return stats


def ensure_positive_definite(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Ensure matrix is positive definite.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix
    epsilon : float
        Minimum eigenvalue
        
    Returns
    -------
    np.ndarray
        Positive definite matrix
    """
    # Symmetrize
    matrix = (matrix + matrix.T) / 2
    
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(matrix)
    
    # Clip eigenvalues
    eigvals = np.maximum(eigvals, epsilon)
    
    # Reconstruct
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """
    Normalize weights to sum to 1.
    
    Parameters
    ----------
    weights : np.ndarray
        Input weights
        
    Returns
    -------
    np.ndarray
        Normalized weights
    """
    total = np.sum(weights)
    
    if total == 0:
        return np.ones_like(weights) / len(weights)
    
    return weights / total


def calculate_portfolio_statistics(weights: np.ndarray,
                                   returns: pd.DataFrame) -> dict:
    """
    Calculate portfolio statistics given weights and returns.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    returns : pd.DataFrame
        Asset returns
        
    Returns
    -------
    dict
        Portfolio statistics
    """
    # Portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Statistics
    stats = {
        'total_return': (1 + portfolio_returns).prod() - 1,
        'annualized_return': annualize_return(portfolio_returns),
        'annualized_volatility': annualize_volatility(portfolio_returns),
        'sharpe_ratio': calculate_sharpe_ratio(portfolio_returns),
        'max_drawdown': calculate_max_drawdown(portfolio_returns),
        'calmar_ratio': calculate_calmar_ratio(portfolio_returns),
    }
    
    return stats


def format_number(value: float, decimals: int = 2, percentage: bool = False) -> str:
    """
    Format number for display.
    
    Parameters
    ----------
    value : float
        Number to format
    decimals : int
        Number of decimal places
    percentage : bool
        Whether to format as percentage
        
    Returns
    -------
    str
        Formatted string
    """
    if percentage:
        return f"{value * 100:.{decimals}f}%"
    else:
        return f"{value:.{decimals}f}"


def generate_date_range(start_date: str,
                       end_date: str,
                       freq: str = 'D') -> pd.DatetimeIndex:
    """
    Generate date range with business days.
    
    Parameters
    ----------
    start_date : str
        Start date
    end_date : str
        End date
    freq : str
        Frequency
        
    Returns
    -------
    pd.DatetimeIndex
        Date range
    """
    if freq == 'B':
        return pd.bdate_range(start=start_date, end=end_date)
    else:
        return pd.date_range(start=start_date, end=end_date, freq=freq)


if __name__ == '__main__':
    # Test utilities
    print("Testing utility functions...")
    
    # Create dummy returns
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    returns = pd.Series(np.random.randn(252) * 0.01, index=dates)
    
    # Test metrics
    print(f"\nAnnualized Return: {annualize_return(returns):.2%}")
    print(f"Annualized Volatility: {annualize_volatility(returns):.2%}")
    print(f"Sharpe Ratio: {calculate_sharpe_ratio(returns):.2f}")
    print(f"Max Drawdown: {calculate_max_drawdown(returns):.2%}")
    print(f"Calmar Ratio: {calculate_calmar_ratio(returns):.2f}")
    
    # Test matrix utilities
    matrix = np.random.randn(5, 5)
    matrix = matrix @ matrix.T  # Make symmetric
    
    psd_matrix = ensure_positive_definite(matrix)
    print(f"\nOriginal matrix min eigenvalue: {np.linalg.eigvals(matrix).min():.6f}")
    print(f"PSD matrix min eigenvalue: {np.linalg.eigvals(psd_matrix).min():.6f}")
    
    # Test weight normalization
    weights = np.array([0.3, 0.5, 0.2, 0.1])
    normalized = normalize_weights(weights)
    print(f"\nOriginal weights sum: {weights.sum():.4f}")
    print(f"Normalized weights sum: {normalized.sum():.4f}")
    
    print("\n✓ Utility functions test successful!")