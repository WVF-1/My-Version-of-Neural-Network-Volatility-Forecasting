"""
Plotting utilities for volatility forecasting and portfolio analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_time_series(data: pd.DataFrame,
                     title: str = 'Time Series',
                     ylabel: str = 'Value',
                     figsize: Tuple[int, int] = (14, 6),
                     save_path: Optional[str] = None):
    """Plot multiple time series."""
    fig, ax = plt.subplots(figsize=figsize)
    
    for col in data.columns:
        ax.plot(data.index, data[col], label=col, linewidth=1.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_predictions_vs_actuals(y_true: np.ndarray,
                                y_pred: np.ndarray,
                                dates: Optional[pd.DatetimeIndex] = None,
                                title: str = 'Predictions vs Actuals',
                                save_path: Optional[str] = None):
    """Plot predictions against actual values."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    if dates is not None:
        x = dates
    else:
        x = np.arange(len(y_true))
    
    # Time series plot
    ax1.plot(x, y_true, label='Actual', linewidth=1.5, alpha=0.8)
    ax1.plot(x, y_pred, label='Predicted', linewidth=1.5, alpha=0.8)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Volatility')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Volatility')
    ax2.set_ylabel('Predicted Volatility')
    ax2.set_title('Prediction Scatter Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_training_history(history: dict,
                          title: str = 'Training History',
                          save_path: Optional[str] = None):
    """Plot training and validation loss."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax.plot(epochs, history['train_loss'], label='Training Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_portfolio_performance(portfolio_value: pd.Series,
                               title: str = 'Portfolio Performance',
                               save_path: Optional[str] = None):
    """Plot portfolio value over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Portfolio value
    ax1.plot(portfolio_value.index, portfolio_value.values, linewidth=2, color='navy')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    
    # Drawdown
    running_max = portfolio_value.cummax()
    drawdown = (portfolio_value - running_max) / running_max
    
    ax2.fill_between(drawdown.index, 0, drawdown.values, 
                     color='red', alpha=0.3, label='Drawdown')
    ax2.set_ylabel('Drawdown')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_weights_over_time(weights: pd.DataFrame,
                           title: str = 'Portfolio Weights Over Time',
                           save_path: Optional[str] = None):
    """Plot portfolio weight allocation over time."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.stackplot(weights.index, *[weights[col] for col in weights.columns],
                labels=weights.columns, alpha=0.8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Weight')
    ax.set_xlabel('Date')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_correlation_matrix(data: pd.DataFrame,
                            title: str = 'Correlation Matrix',
                            save_path: Optional[str] = None):
    """Plot correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr = data.corr()
    
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_volatility_forecast(dates: pd.DatetimeIndex,
                             actual_vol: np.ndarray,
                             predicted_vol: np.ndarray,
                             asset_name: str = 'Asset',
                             save_path: Optional[str] = None):
    """Plot volatility forecast comparison."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(dates, actual_vol, label='Realized Volatility', 
            linewidth=2, alpha=0.8, color='navy')
    ax.plot(dates, predicted_vol, label='Predicted Volatility', 
            linewidth=2, alpha=0.8, color='orange')
    
    ax.fill_between(dates, actual_vol, predicted_vol, 
                    alpha=0.2, color='gray')
    
    ax.set_title(f'Volatility Forecast: {asset_name}', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Annualized Volatility')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_risk_contributions(risk_contrib: np.ndarray,
                            asset_names: List[str],
                            title: str = 'Risk Contributions',
                            save_path: Optional[str] = None):
    """Plot risk contribution pie chart."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(asset_names)))
    
    wedges, texts, autotexts = ax.pie(
        risk_contrib,
        labels=asset_names,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    
    # Improve text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_returns_distribution(returns: pd.Series,
                              title: str = 'Returns Distribution',
                              save_path: Optional[str] = None):
    """Plot returns distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(returns, bins=50, alpha=0.7, color='navy', edgecolor='black')
    ax1.axvline(returns.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {returns.mean():.4f}')
    ax1.axvline(returns.median(), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {returns.median():.4f}')
    ax1.set_title('Returns Histogram', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Returns')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    # Test plotting functions
    print("Testing plotting functions...")
    
    # Generate dummy data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    
    # Test time series plot
    data = pd.DataFrame({
        'Series1': np.cumsum(np.random.randn(252) * 0.01),
        'Series2': np.cumsum(np.random.randn(252) * 0.01)
    }, index=dates)
    
    plot_time_series(data, title='Test Time Series')
    
    print("\n✓ Plotting test successful!")