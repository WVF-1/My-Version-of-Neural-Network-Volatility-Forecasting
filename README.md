# My-Version-of-Neural-Network-Volatility-Forecasting
Utilizing PyTorch and LSTM modeling to forecast the future market volatility of different investment allocations.

# Neural Network Volatility Forecasting

A sophisticated machine learning system for forecasting market volatility using LSTM neural networks and constructing optimal risk-parity portfolios. This project implements a complete pipeline from data acquisition to portfolio backtesting, leveraging deep learning for volatility prediction and modern portfolio theory for asset allocation.

## Overview

This project demonstrates the application of Long Short-Term Memory (LSTM) neural networks to predict future volatility of financial assets and uses these forecasts to construct risk-parity portfolios. The system integrates:

- **Advanced Time Series Modeling**: Multi-layer LSTM architecture with dropout regularization
- **Feature Engineering**: Technical indicators, rolling volatilities, and exponentially weighted moving averages
- **Walk-Forward Validation**: Robust out-of-sample testing methodology
- **Risk Parity Optimization**: Portfolio construction targeting equal risk contribution across assets
- **Comprehensive Backtesting**: Transaction costs, rebalancing, and performance metrics

## Key Features

- **End-to-End Pipeline**: Automated workflow from data fetching to performance analysis
- **FRED API Integration**: Automatic retrieval of economic and financial data
- **PyTorch Implementation**: Efficient GPU-accelerated neural network training
- **Multiple Validation Schemes**: Walk-forward, K-fold, and rolling LOOCV
- **Portfolio Optimization**: Risk parity with configurable volatility targeting
- **Extensive Visualization**: Performance charts, weight evolution, and prediction analysis
- **Jupyter Notebooks**: Detailed exploratory analysis and model development notebooks

## Project Structure

```
.
├── data/
│   ├── raw/                      # Raw data from FRED API
│   └── processed/                # Preprocessed data and features
├── notebooks/
│   ├── Data Exploration and Regime Analysis.ipynb
│   ├── Sequence Builder.ipynb
│   ├── LSTM Model Training.ipynb
│   ├── Baseline Models.ipynb
│   ├── Walk Forward Validation.ipynb
│   ├── Portfolio Construction.ipynb
│   └── Results Analysis.ipynb
├── src/
│   ├── data/
│   │   ├── fetch_data.py        # FRED API data retrieval
│   │   ├── preprocessing.py     # Data cleaning and preprocessing
│   │   └── feature_engineering.py # Feature creation
│   ├── models/
│   │   ├── lstm.py              # LSTM model architecture
│   │   ├── train.py             # Training loops and optimization
│   │   └── evaluate.py          # Model evaluation metrics
│   ├── validation/
│   │   ├── walk_forward.py      # Walk-forward validation
│   │   ├── kfold.py             # K-fold cross-validation
│   │   └── rolling_loocv.py     # Rolling leave-one-out CV
│   ├── portfolio/
│   │   ├── risk_parity.py       # Risk parity optimization
│   │   └── backtest.py          # Portfolio backtesting engine
│   └── utils/
│       ├── helpers.py           # Utility functions
│       ├── metrics.py           # Performance metrics
│       └── plotting.py          # Visualization functions
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- (Optional) CUDA-capable GPU for faster training

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Neural-Network-Volatility-Forecasting.git
   cd Neural-Network-Volatility-Forecasting
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up FRED API key** (for data fetching)
   - Get a free API key from [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)
   - Set as environment variable:
     ```bash
     export FRED_API_KEY='your_api_key_here'
     ```

## Usage

### Quick Start

Run the complete pipeline with default settings:

```bash
python main.py
```

This will execute:
1. Data fetching from FRED API
2. Data preprocessing and feature engineering
3. LSTM model training for each asset
4. Portfolio construction using risk parity
5. Backtesting and performance analysis
6. Visualization generation

### Jupyter Notebooks

For detailed exploration and analysis:

```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and run notebooks in order:
1. **Data Exploration and Regime Analysis**: Understand the data and market regimes
2. **Sequence Builder**: Create sequences for LSTM training
3. **LSTM Model Training**: Train and tune the neural network
4. **Baseline Models**: Compare against traditional methods
5. **Walk Forward Validation**: Validate model performance
6. **Portfolio Construction**: Build risk parity portfolios
7. **Results Analysis**: Comprehensive performance evaluation

### Custom Configuration

Modify the configuration dictionary in `main.py`:

```python
config = {
    'data': {
        'start_date': '2010-01-01',
        # ... other data settings
    },
    'model': {
        'sequence_length': 60,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'epochs': 100,
        # ... other model settings
    },
    'portfolio': {
        'vol_target': 0.10,
        'rebalance_freq': 'M',
        'transaction_cost': 0.001,
        # ... other portfolio settings
    }
}
```

## Methodology

### 1. Data Pipeline

- **Sources**: FRED API for economic indicators, market indices, commodities
- **Preprocessing**: Missing value handling, outlier detection, normalization
- **Feature Engineering**: 
  - Rolling volatilities (5, 21, 63-day windows)
  - Exponentially weighted moving averages
  - Technical indicators
  - Lagged returns and volatilities

### 2. LSTM Model

The model architecture consists of:
- Input layer (multi-dimensional features)
- Stacked LSTM layers with dropout
- Fully connected output layer
- MSE loss function with Adam optimizer
- Early stopping with patience

### 3. Validation Strategy

**Walk-Forward Validation**:
- Initial training window: 1000 observations
- Step size: 21 days (approximately 1 month)
- Window type: Expanding (growing training set)
- Out-of-sample testing on each step

### 4. Portfolio Construction

**Risk Parity Approach**:
- Equal risk contribution from each asset
- Volatility forecasts from LSTM predictions
- Optional correlation matrix for covariance estimation
- Target portfolio volatility (default: 10% annualized)

### 5. Backtesting

- Transaction costs: 0.1% per trade (configurable)
- Rebalancing frequency: Monthly (configurable)
- Performance metrics: Sharpe ratio, max drawdown, Calmar ratio, etc.

## Key Dependencies

- **PyTorch**: Deep learning framework for LSTM implementation
- **NumPy/Pandas**: Data manipulation and analysis
- **FRED API**: Economic data retrieval
- **yfinance**: Market data backup source
- **cvxpy**: Convex optimization for portfolio construction
- **riskfolio-lib**: Portfolio optimization utilities
- **matplotlib/seaborn/plotly**: Visualization

## Configuration Options

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sequence_length` | 60 | Number of historical observations for LSTM input |
| `hidden_size` | 64 | Number of hidden units in LSTM layers |
| `num_layers` | 2 | Number of stacked LSTM layers |
| `dropout` | 0.2 | Dropout rate for regularization |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `batch_size` | 32 | Training batch size |
| `epochs` | 100 | Maximum training epochs |
| `early_stopping_patience` | 15 | Epochs without improvement before stopping |

### Portfolio Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vol_target` | 0.10 | Target portfolio volatility (10% annualized) |
| `rebalance_freq` | 'M' | Rebalancing frequency ('D'=daily, 'M'=monthly) |
| `transaction_cost` | 0.001 | Transaction cost (0.1% per trade) |
| `initial_capital` | 1,000,000 | Starting portfolio value |

## Results Interpretation

The system outputs:

1. **Model Performance Metrics**:
   - Training and validation loss curves
   - Out-of-sample prediction accuracy (RMSE, MAE, R²)
   - Directional accuracy of volatility forecasts

2. **Portfolio Performance**:
   - Total return and annualized return
   - Sharpe ratio and Sortino ratio
   - Maximum drawdown and Calmar ratio
   - Win rate and profit factor
   - Portfolio value over time

3. **Visualizations**:
   - Predicted vs. actual volatility
   - Portfolio weights evolution
   - Cumulative returns
   - Drawdown periods

---

**Disclaimer**: This project is for educational and research purposes only. The code and models provided should not be used for actual trading or investment decisions without proper due diligence and risk assessment. Past performance does not guarantee future results.
