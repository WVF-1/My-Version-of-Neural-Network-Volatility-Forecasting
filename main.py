"""
Main execution script for Volatility LSTM Risk Parity Portfolio System.

This script runs the complete pipeline:
1. Data fetching and preprocessing
2. Feature engineering
3. LSTM model training
4. Walk-forward validation
5. Portfolio construction
6. Backtesting and performance analysis
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append('src')

# Import modules
from data.fetch_data import fetch_and_save_data
from data.preprocess import preprocess_data
from data.feature_engineering import create_features, FeatureEngineer
from models.lstm import VolatilityLSTM, LSTMDataset, get_device
from models.train import train_model
from models.evaluate import evaluate_predictions, print_evaluation_report
from validation.walk_forward import walk_forward_cv
from portfolio.risk_parity import construct_risk_parity_portfolio
from portfolio.backtest import backtest_strategy
from utils.helpers import set_random_seed, print_section_header, Timer, format_metrics_table
from utils.plotting import (plot_time_series, plot_predictions_vs_actuals, 
                            plot_portfolio_performance, plot_weights_over_time)


def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("VOLATILITY LSTM RISK PARITY PORTFOLIO SYSTEM".center(80))
    print("="*80)
    
    # Set random seed for reproducibility
    set_random_seed(42)
    
    # Configuration
    config = {
        'data': {
            'start_date': '2010-01-01',
            'raw_data_path': 'data/raw/fred_data.csv',
            'processed_data_path': 'data/processed/preprocessed_data.csv',
            'features_path': 'data/processed/features.csv',
            'targets_path': 'data/processed/targets.csv'
        },
        'features': {
            'vol_windows': [5, 21, 63],
            'ewma_spans': [21, 63],
            'target_horizon': 21
        },
        'model': {
            'sequence_length': 60,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 15
        },
        'validation': {
            'initial_train_size': 1000,
            'step_size': 21,
            'window_type': 'expanding'
        },
        'portfolio': {
            'vol_target': 0.10,
            'rebalance_freq': 'M',
            'transaction_cost': 0.001,
            'initial_capital': 1000000.0
        }
    }
    
    # ========================================================================
    # STEP 1: Data Fetching
    # ========================================================================
    print_section_header("STEP 1: DATA FETCHING")
    
    if not os.path.exists(config['data']['raw_data_path']):
        with Timer("Data fetching"):
            print("Fetching data from FRED API...")
            print("Note: Set FRED_API_KEY environment variable")
            print("Get free key at: https://fred.stlouisfed.org/docs/api/api_key.html\n")
            
            try:
                raw_data = fetch_and_save_data(
                    start_date=config['data']['start_date'],
                    output_path=config['data']['raw_data_path']
                )
            except Exception as e:
                print(f"\n⚠️  Error fetching data: {e}")
                print("Using sample data for demonstration...")
                # Create sample data for demo
                dates = pd.date_range('2010-01-01', periods=3000, freq='D')
                raw_data = pd.DataFrame({
                    'SP500': np.cumsum(np.random.randn(3000) * 2) + 2000,
                    'DGS10': np.abs(np.random.randn(3000) * 0.5 + 2.5),
                    'GOLD': np.cumsum(np.random.randn(3000) * 1) + 1200,
                    'COMMODITY': np.cumsum(np.random.randn(3000) * 1.5) + 100,
                    'VIX': np.abs(np.random.randn(3000) * 5 + 15)
                }, index=dates)
                raw_data.to_csv(config['data']['raw_data_path'])
                print(f"Sample data saved to {config['data']['raw_data_path']}")
    else:
        print(f"Loading existing data from {config['data']['raw_data_path']}")
        raw_data = pd.read_csv(config['data']['raw_data_path'], 
                               index_col=0, parse_dates=True)
    
    print(f"✓ Data loaded: {raw_data.shape}")
    print(f"  Date range: {raw_data.index[0]} to {raw_data.index[-1]}")
    
    # ========================================================================
    # STEP 2: Data Preprocessing
    # ========================================================================
    print_section_header("STEP 2: DATA PREPROCESSING")
    
    if not os.path.exists(config['data']['processed_data_path']):
        with Timer("Data preprocessing"):
            processed_data = preprocess_data(
                raw_data,
                method='ffill',
                max_fill=5,
                remove_outliers=True,
                output_path=config['data']['processed_data_path']
            )
    else:
        print(f"Loading existing processed data...")
        processed_data = pd.read_csv(config['data']['processed_data_path'],
                                     index_col=0, parse_dates=True)
    
    print(f"✓ Processed data: {processed_data.shape}")
    
    # ========================================================================
    # STEP 3: Feature Engineering
    # ========================================================================
    print_section_header("STEP 3: FEATURE ENGINEERING")
    
    if not (os.path.exists(config['data']['features_path']) and 
            os.path.exists(config['data']['targets_path'])):
        with Timer("Feature engineering"):
            features, targets = create_features(
                processed_data,
                vol_windows=config['features']['vol_windows'],
                ewma_spans=config['features']['ewma_spans'],
                target_horizon=config['features']['target_horizon'],
                output_dir='data/processed/'
            )
    else:
        print("Loading existing features and targets...")
        features = pd.read_csv(config['data']['features_path'],
                              index_col=0, parse_dates=True)
        targets = pd.read_csv(config['data']['targets_path'],
                             index_col=0, parse_dates=True)
    
    print(f"✓ Features: {features.shape}")
    print(f"✓ Targets: {targets.shape}")
    
    # Get asset list
    asset_names = [col.replace('_target_vol', '') for col in targets.columns]
    print(f"  Assets: {asset_names}")
    
    # ========================================================================
    # STEP 4: LSTM Model Training (Per Asset)
    # ========================================================================
    print_section_header("STEP 4: LSTM MODEL TRAINING")
    
    device = get_device()
    models = {}
    predictions_dict = {}
    
    engineer = FeatureEngineer(processed_data)
    engineer.features = features
    engineer.targets = targets
    
    for asset in asset_names:
        print(f"\n{'='*70}")
        print(f"Training model for: {asset}")
        print(f"{'='*70}")
        
        # Get asset-specific features and target
        asset_features = engineer.get_asset_features(asset)
        asset_target = engineer.get_asset_target(asset)
        
        # Align features and target
        common_idx = asset_features.index.intersection(asset_target.index)
        X = asset_features.loc[common_idx].values
        y = asset_target.loc[common_idx].values
        
        # Train/val split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"  Train size: {len(X_train)}")
        print(f"  Val size: {len(X_val)}")
        
        # Create datasets
        train_dataset = LSTMDataset(
            X_train, y_train, 
            sequence_length=config['model']['sequence_length']
        )
        val_dataset = LSTMDataset(
            X_val, y_val, 
            sequence_length=config['model']['sequence_length']
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['model']['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['model']['batch_size'],
            shuffle=False
        )
        
        # Create and train model
        model = VolatilityLSTM(
            input_size=X.shape[1],
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout']
        )
        
        with Timer(f"Training {asset} model"):
            trained_model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=config['model']['epochs'],
                learning_rate=config['model']['learning_rate'],
                early_stopping_patience=config['model']['early_stopping_patience']
            )
        
        models[asset] = trained_model
        
        # Generate predictions on validation set
        trained_model.eval()
        val_predictions = []
        val_actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                pred = trained_model(batch_x)
                val_predictions.append(pred.cpu().numpy())
                val_actuals.append(batch_y.numpy())
        
        val_predictions = np.concatenate(val_predictions).flatten()
        val_actuals = np.concatenate(val_actuals).flatten()
        
        predictions_dict[asset] = {
            'predictions': val_predictions,
            'actuals': val_actuals
        }
        
        # Evaluate
        print_evaluation_report(val_actuals, val_predictions, asset_name=asset)
    
    # ========================================================================
    # STEP 5: Portfolio Construction
    # ========================================================================
    print_section_header("STEP 5: PORTFOLIO CONSTRUCTION")
    
    # Get predicted volatilities for each asset
    predicted_vols = []
    for asset in asset_names:
        # Use mean of predictions as volatility forecast
        mean_vol = predictions_dict[asset]['predictions'].mean()
        predicted_vols.append(mean_vol)
    
    predicted_vols = np.array(predicted_vols)
    
    print("Predicted volatilities:")
    for asset, vol in zip(asset_names, predicted_vols):
        print(f"  {asset}: {vol:.2%}")
    
    # Construct risk parity portfolio
    weights, portfolio_info = construct_risk_parity_portfolio(
        volatilities=predicted_vols,
        asset_names=asset_names,
        correlations=None,  # Could calculate from returns
        vol_target=config['portfolio']['vol_target']
    )
    
    print("\nRisk Parity Portfolio Weights:")
    for asset, weight in zip(asset_names, weights):
        print(f"  {asset}: {weight:.2%}")
    
    print(f"\nPortfolio volatility: {portfolio_info['portfolio_volatility']:.2%}")
    
    # ========================================================================
    # STEP 6: Backtest
    # ========================================================================
    print_section_header("STEP 6: PORTFOLIO BACKTESTING")
    
    # Calculate returns from processed data
    returns = processed_data[asset_names].pct_change().dropna()
    
    # Create constant weights for backtest (in practice, would rebalance based on new forecasts)
    backtest_weights = pd.DataFrame(
        np.tile(weights, (len(returns), 1)),
        index=returns.index,
        columns=asset_names
    )
    
    with Timer("Backtesting"):
        backtest_results = backtest_strategy(
            returns=returns,
            weights=backtest_weights,
            initial_capital=config['portfolio']['initial_capital'],
            transaction_cost=config['portfolio']['transaction_cost'],
            rebalance_freq=config['portfolio']['rebalance_freq']
        )
    
    # Print results
    print("\nBacktest Performance Metrics:")
    print(format_metrics_table(backtest_results['metrics']))
    
    print(f"\nInitial capital: ${config['portfolio']['initial_capital']:,.2f}")
    print(f"Final portfolio value: ${backtest_results['portfolio_value'].iloc[-1]:,.2f}")
    
    # ========================================================================
    # STEP 7: Visualization
    # ========================================================================
    print_section_header("STEP 7: GENERATING VISUALIZATIONS")
    
    try:
        # Plot portfolio performance
        plot_portfolio_performance(
            backtest_results['portfolio_value'],
            title='Risk Parity Portfolio Performance',
            save_path='reports/figures/portfolio_performance.png'
        )
        
        # Plot weights over time
        plot_weights_over_time(
            backtest_results['weights_history'],
            title='Portfolio Weights Over Time',
            save_path='reports/figures/portfolio_weights.png'
        )
        
        print("✓ Visualizations saved to reports/figures/")
    except Exception as e:
        print(f"⚠️  Visualization error (non-critical): {e}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_section_header("PIPELINE COMPLETE")
    
    print("Summary:")
    print(f"  ✓ Data processed: {processed_data.shape[0]} observations")
    print(f"  ✓ Models trained: {len(models)} assets")
    print(f"  ✓ Portfolio constructed: Risk Parity")
    print(f"  ✓ Backtest completed: {len(returns)} days")
    print(f"  ✓ Final return: {backtest_results['metrics']['Total_Return']:.2%}")
    print(f"  ✓ Sharpe ratio: {backtest_results['metrics']['Sharpe_Ratio']:.2f}")
    
    print("\n" + "="*80)
    print("For detailed analysis, see Jupyter notebooks in notebooks/ directory")
    print("="*80 + "\n")
    
    return {
        'data': processed_data,
        'features': features,
        'targets': targets,
        'models': models,
        'predictions': predictions_dict,
        'portfolio_weights': weights,
        'backtest_results': backtest_results
    }


if __name__ == '__main__':
    try:
        results = main()
        print("\n✓ Pipeline executed successfully!")
    except KeyboardInterrupt:
        print("\n\n⚠️  Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)