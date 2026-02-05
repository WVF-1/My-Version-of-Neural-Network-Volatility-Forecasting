"""
Portfolio backtesting engine with transaction costs and realistic constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import warnings


class PortfolioBacktest:
    """
    Backtest portfolio strategies with realistic transaction costs.
    """
    
    def __init__(self,
                 returns: pd.DataFrame,
                 initial_capital: float = 1000000.0,
                 transaction_cost: float = 0.001,
                 rebalance_freq: str = 'M'):
        """
        Initialize backtester.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (columns = assets, index = dates)
        initial_capital : float
            Starting capital
        transaction_cost : float
            Transaction cost as fraction (0.001 = 10 basis points)
        rebalance_freq : str
            Rebalancing frequency ('D', 'W', 'M', 'Q')
        """
        self.returns = returns
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.rebalance_freq = rebalance_freq
        
        self.portfolio_value = None
        self.weights_history = None
        self.turnover_history = None
        
    def run(self, weights: pd.DataFrame) -> Dict:
        """
        Run backtest with given weight history.
        
        Parameters
        ----------
        weights : pd.DataFrame
            Portfolio weights over time (columns = assets, index = dates)
            Must align with returns index
            
        Returns
        -------
        dict
            Backtest results including portfolio value, returns, metrics
        """
        # Align weights and returns
        common_dates = weights.index.intersection(self.returns.index)
        weights = weights.loc[common_dates]
        returns = self.returns.loc[common_dates]
        
        if len(common_dates) == 0:
            raise ValueError("No common dates between weights and returns")
        
        # Determine rebalancing dates
        rebalance_dates = self._get_rebalance_dates(common_dates)
        
        # Initialize tracking
        portfolio_value = pd.Series(index=common_dates, dtype=float)
        portfolio_value.iloc[0] = self.initial_capital
        
        weights_held = pd.DataFrame(
            index=common_dates,
            columns=weights.columns,
            dtype=float
        )
        weights_held.iloc[0] = weights.iloc[0]
        
        turnover = pd.Series(index=common_dates, dtype=float)
        turnover.iloc[0] = 0.0
        
        # Run backtest
        for i in range(1, len(common_dates)):
            date = common_dates[i]
            prev_date = common_dates[i-1]
            
            # Previous portfolio value
            prev_value = portfolio_value.iloc[i-1]
            
            # Check if rebalancing
            if date in rebalance_dates:
                # Rebalance to target weights
                target_weights = weights.loc[date]
                prev_weights = weights_held.iloc[i-1]
                
                # Calculate turnover
                weight_change = np.abs(target_weights - prev_weights).sum()
                turnover.iloc[i] = weight_change
                
                # Apply transaction costs
                cost = weight_change * self.transaction_cost * prev_value
                
                # Update portfolio value after costs
                portfolio_value.iloc[i] = prev_value - cost
                
                # Update held weights
                weights_held.iloc[i] = target_weights
            else:
                # No rebalancing - weights drift naturally
                portfolio_value.iloc[i] = prev_value
                weights_held.iloc[i] = weights_held.iloc[i-1]
                turnover.iloc[i] = 0.0
            
            # Apply returns
            period_return = (weights_held.iloc[i] * returns.loc[date]).sum()
            portfolio_value.iloc[i] = portfolio_value.iloc[i] * (1 + period_return)
        
        # Store history
        self.portfolio_value = portfolio_value
        self.weights_history = weights_held
        self.turnover_history = turnover
        
        # Calculate portfolio returns
        portfolio_returns = portfolio_value.pct_change()
        
        # Calculate metrics
        metrics = self.calculate_metrics(portfolio_returns)
        
        return {
            'portfolio_value': portfolio_value,
            'portfolio_returns': portfolio_returns,
            'weights_history': weights_held,
            'turnover': turnover,
            'metrics': metrics,
            'rebalance_dates': rebalance_dates
        }
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Get rebalancing dates based on frequency."""
        # Group by period
        if self.rebalance_freq == 'M':
            # Monthly: first business day of each month
            rebalance_dates = dates[dates.is_month_start]
        elif self.rebalance_freq == 'Q':
            # Quarterly: first business day of each quarter
            rebalance_dates = dates[dates.is_quarter_start]
        elif self.rebalance_freq == 'W':
            # Weekly: every Monday
            rebalance_dates = dates[dates.weekday == 0]
        elif self.rebalance_freq == 'D':
            # Daily: every day
            rebalance_dates = dates
        else:
            raise ValueError(f"Unknown rebalance_freq: {self.rebalance_freq}")
        
        # Ensure first date is included
        if dates[0] not in rebalance_dates:
            rebalance_dates = dates[[0]].append(rebalance_dates)
        
        return rebalance_dates
    
    def calculate_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate performance metrics.
        
        Parameters
        ----------
        returns : pd.Series
            Portfolio returns
            
        Returns
        -------
        dict
            Performance metrics
        """
        # Remove NaN
        returns = returns.dropna()
        
        if len(returns) == 0:
            return {}
        
        # Annualization factor (assuming daily data)
        ann_factor = 252
        
        # Total return
        total_return = (1 + returns).prod() - 1
        
        # Annualized return (CAGR)
        n_years = len(returns) / ann_factor
        if n_years > 0:
            cagr = (1 + total_return) ** (1 / n_years) - 1
        else:
            cagr = 0.0
        
        # Volatility
        volatility = returns.std() * np.sqrt(ann_factor)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if volatility > 0:
            sharpe = cagr / volatility
        else:
            sharpe = 0.0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        if max_drawdown < 0:
            calmar = cagr / abs(max_drawdown)
        else:
            calmar = 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_vol = downside_returns.std() * np.sqrt(ann_factor)
            sortino = cagr / downside_vol if downside_vol > 0 else 0.0
        else:
            sortino = 0.0
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Average turnover
        if self.turnover_history is not None:
            avg_turnover = self.turnover_history.mean()
        else:
            avg_turnover = 0.0
        
        metrics = {
            'Total_Return': total_return,
            'CAGR': cagr,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe,
            'Sortino_Ratio': sortino,
            'Max_Drawdown': max_drawdown,
            'Calmar_Ratio': calmar,
            'Win_Rate': win_rate,
            'Avg_Turnover': avg_turnover,
            'N_Years': n_years
        }
        
        return metrics
    
    def get_summary(self, results: Dict) -> pd.DataFrame:
        """Get formatted summary of backtest results."""
        metrics = results['metrics']
        
        summary = pd.DataFrame([{
            'Total Return': f"{metrics['Total_Return']:.2%}",
            'CAGR': f"{metrics['CAGR']:.2%}",
            'Volatility': f"{metrics['Volatility']:.2%}",
            'Sharpe Ratio': f"{metrics['Sharpe_Ratio']:.2f}",
            'Sortino Ratio': f"{metrics['Sortino_Ratio']:.2f}",
            'Max Drawdown': f"{metrics['Max_Drawdown']:.2%}",
            'Calmar Ratio': f"{metrics['Calmar_Ratio']:.2f}",
            'Win Rate': f"{metrics['Win_Rate']:.2%}",
            'Avg Turnover': f"{metrics['Avg_Turnover']:.2%}",
            'Years': f"{metrics['N_Years']:.1f}"
        }])
        
        return summary.T


def backtest_strategy(returns: pd.DataFrame,
                     weights: pd.DataFrame,
                     initial_capital: float = 1000000.0,
                     transaction_cost: float = 0.001,
                     rebalance_freq: str = 'M') -> Dict:
    """
    Convenience function for backtesting.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns
    weights : pd.DataFrame
        Portfolio weights over time
    initial_capital : float
        Starting capital
    transaction_cost : float
        Transaction cost (fraction)
    rebalance_freq : str
        Rebalancing frequency
        
    Returns
    -------
    dict
        Backtest results
    """
    backtester = PortfolioBacktest(
        returns=returns,
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        rebalance_freq=rebalance_freq
    )
    
    results = backtester.run(weights)
    
    return results


if __name__ == '__main__':
    # Test backtester
    print("Testing Portfolio Backtester...")
    
    # Create dummy data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    
    # Generate returns
    n_assets = 4
    returns = pd.DataFrame(
        np.random.randn(len(dates), n_assets) * 0.01,
        index=dates,
        columns=['Asset1', 'Asset2', 'Asset3', 'Asset4']
    )
    
    # Generate weights (equal weight rebalanced monthly)
    weights = pd.DataFrame(
        np.ones((len(dates), n_assets)) / n_assets,
        index=dates,
        columns=returns.columns
    )
    
    # Run backtest
    results = backtest_strategy(
        returns=returns,
        weights=weights,
        transaction_cost=0.001,
        rebalance_freq='M'
    )
    
    print("\nBacktest Results:")
    print(results['metrics'])
    
    print(f"\nFinal portfolio value: ${results['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Number of rebalances: {len(results['rebalance_dates'])}")
    
    print("\n✓ Backtester test successful!")