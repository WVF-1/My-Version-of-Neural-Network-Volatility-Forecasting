"""
Evaluation module for volatility forecasting models.
Implements RMSE, MAE, and QLIKE metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        RMSE
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        MAE
    """
    return mean_absolute_error(y_true, y_pred)


def qlike(y_true: np.ndarray, y_pred: np.ndarray, epsilon=1e-8) -> float:
    """
    Calculate QLIKE (Quasi-Likelihood) loss for volatility forecasting.
    
    QLIKE is specifically designed for volatility forecasting and is
    robust to the heteroskedastic nature of volatility.
    
    QLIKE = (y_true / y_pred) - log(y_true / y_pred) - 1
    
    Parameters
    ----------
    y_true : np.ndarray
        True realized volatility
    y_pred : np.ndarray
        Predicted volatility
    epsilon : float
        Small constant to avoid division by zero
        
    Returns
    -------
    float
        Average QLIKE loss
    """
    # Ensure positive values
    y_true = np.maximum(y_true, epsilon)
    y_pred = np.maximum(y_pred, epsilon)
    
    # Calculate QLIKE
    ratio = y_true / y_pred
    qlike_loss = ratio - np.log(ratio) - 1
    
    return np.mean(qlike_loss)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon=1e-8) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    epsilon : float
        Small constant to avoid division by zero
        
    Returns
    -------
    float
        MAPE (in percentage)
    """
    y_true = np.maximum(np.abs(y_true), epsilon)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R-squared score.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        R-squared score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def evaluate_predictions(y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        prefix: str = '') -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    prefix : str
        Prefix for metric names
        
    Returns
    -------
    dict
        Dictionary of metrics
    """
    metrics = {
        f'{prefix}RMSE': rmse(y_true, y_pred),
        f'{prefix}MAE': mae(y_true, y_pred),
        f'{prefix}QLIKE': qlike(y_true, y_pred),
        f'{prefix}MAPE': mape(y_true, y_pred),
        f'{prefix}R2': r2_score(y_true, y_pred)
    }
    
    return metrics


def directional_accuracy(y_true: np.ndarray, 
                        y_pred: np.ndarray,
                        threshold: float = 0.0) -> float:
    """
    Calculate directional accuracy (for changes in volatility).
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    threshold : float
        Threshold for considering a change
        
    Returns
    -------
    float
        Directional accuracy (0 to 1)
    """
    if len(y_true) < 2:
        return np.nan
    
    # Calculate changes
    true_change = np.diff(y_true) > threshold
    pred_change = np.diff(y_pred) > threshold
    
    # Calculate accuracy
    accuracy = np.mean(true_change == pred_change)
    
    return accuracy


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
    
    def evaluate(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 dates: Optional[pd.DatetimeIndex] = None,
                 asset_name: str = '') -> pd.DataFrame:
        """
        Comprehensive evaluation of predictions.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        dates : pd.DatetimeIndex, optional
            Date index for predictions
        asset_name : str
            Name of asset being evaluated
            
        Returns
        -------
        pd.DataFrame
            Evaluation results
        """
        # Overall metrics
        metrics = evaluate_predictions(y_true, y_pred)
        
        # Directional accuracy
        metrics['Directional_Accuracy'] = directional_accuracy(y_true, y_pred)
        
        # Prediction bias
        metrics['Mean_Bias'] = np.mean(y_pred - y_true)
        metrics['Median_Bias'] = np.median(y_pred - y_true)
        
        # Store results
        self.results[asset_name] = metrics
        
        # Create results DataFrame
        results_df = pd.DataFrame([metrics])
        results_df.insert(0, 'Asset', asset_name)
        
        return results_df
    
    def evaluate_by_period(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          dates: pd.DatetimeIndex,
                          period: str = 'Y') -> pd.DataFrame:
        """
        Evaluate predictions by time period.
        
        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        dates : pd.DatetimeIndex
            Date index
        period : str
            Pandas period code ('Y', 'Q', 'M')
            
        Returns
        -------
        pd.DataFrame
            Metrics by period
        """
        df = pd.DataFrame({
            'true': y_true,
            'pred': y_pred
        }, index=dates)
        
        # Group by period
        grouped = df.groupby(pd.Grouper(freq=period))
        
        results = []
        for period_name, group in grouped:
            if len(group) < 2:
                continue
            
            metrics = evaluate_predictions(
                group['true'].values,
                group['pred'].values
            )
            metrics['Period'] = period_name
            metrics['N_Samples'] = len(group)
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def compare_models(self, 
                      models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Parameters
        ----------
        models_dict : dict
            Dictionary mapping model names to (y_true, y_pred) tuples
            
        Returns
        -------
        pd.DataFrame
            Comparison results
        """
        results = []
        
        for model_name, (y_true, y_pred) in models_dict.items():
            metrics = evaluate_predictions(y_true, y_pred)
            metrics['Model'] = model_name
            results.append(metrics)
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df[['Model'] + [c for c in comparison_df.columns if c != 'Model']]
        
        return comparison_df
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all evaluations."""
        if not self.results:
            return pd.DataFrame()
        
        summary = pd.DataFrame(self.results).T
        summary.index.name = 'Asset'
        
        return summary.reset_index()


def print_evaluation_report(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           asset_name: str = 'Asset'):
    """
    Print formatted evaluation report.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    asset_name : str
        Asset name
    """
    metrics = evaluate_predictions(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Report: {asset_name}")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Value':>15}")
    print(f"{'-'*60}")
    
    for metric, value in metrics.items():
        print(f"{metric:<25} {value:>15.6f}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Test evaluation metrics
    print("Testing evaluation metrics...")
    
    # Create dummy data
    np.random.seed(42)
    n = 1000
    y_true = np.abs(np.random.randn(n) * 0.2 + 0.15)
    y_pred = y_true + np.random.randn(n) * 0.05
    
    # Test metrics
    print(f"\nRMSE: {rmse(y_true, y_pred):.6f}")
    print(f"MAE: {mae(y_true, y_pred):.6f}")
    print(f"QLIKE: {qlike(y_true, y_pred):.6f}")
    print(f"MAPE: {mape(y_true, y_pred):.2f}%")
    print(f"R²: {r2_score(y_true, y_pred):.6f}")
    
    # Test evaluator
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(y_true, y_pred, asset_name='TEST')
    print("\nEvaluation results:")
    print(results)
    
    # Test comparison
    models = {
        'Model_A': (y_true, y_pred),
        'Model_B': (y_true, y_pred + np.random.randn(n) * 0.03)
    }
    comparison = evaluator.compare_models(models)
    print("\nModel comparison:")
    print(comparison)
    
    print("\n✓ Evaluation test successful!")