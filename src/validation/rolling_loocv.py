"""
Rolling-origin Leave-One-Out Cross-Validation (LOOCV).
Tests model on each time point using only past data.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable, Optional
from tqdm import tqdm
import warnings


class RollingOriginCV:
    """
    Rolling-origin cross-validation (similar to LOOCV for time series).
    
    For each test point, trains on a window of past data and predicts
    just that single point. Very thorough but computationally expensive.
    """
    
    def __init__(self,
                 min_train_size: int = 500,
                 window_size: Optional[int] = None,
                 step: int = 1):
        """
        Initialize rolling-origin validator.
        
        Parameters
        ----------
        min_train_size : int
            Minimum size of training window
        window_size : int, optional
            Fixed window size. If None, uses expanding window
        step : int
            Step size between predictions (1 = predict every point)
        """
        self.min_train_size = min_train_size
        self.window_size = window_size
        self.step = step
    
    def split(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test split indices.
        
        Parameters
        ----------
        n_samples : int
            Total number of samples
            
        Returns
        -------
        list
            List of (train_idx, test_idx) tuples
        """
        if self.min_train_size >= n_samples:
            raise ValueError("min_train_size must be less than n_samples")
        
        splits = []
        
        # Start from min_train_size and predict each subsequent point
        for test_pos in range(self.min_train_size, n_samples, self.step):
            # Test index (single point)
            test_idx = np.array([test_pos])
            
            # Train indices
            if self.window_size is not None:
                # Fixed rolling window
                train_start = max(0, test_pos - self.window_size)
                train_idx = np.arange(train_start, test_pos)
            else:
                # Expanding window
                train_idx = np.arange(0, test_pos)
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def validate(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 model_fn: Callable,
                 dates: Optional[pd.DatetimeIndex] = None,
                 max_folds: Optional[int] = None,
                 verbose: bool = True) -> Dict:
        """
        Perform rolling-origin validation.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target array
        model_fn : callable
            Model function: model_fn(X_train, y_train, X_test) -> y_pred
        dates : pd.DatetimeIndex, optional
            Date index
        max_folds : int, optional
            Maximum number of folds to run (for faster testing)
        verbose : bool
            Whether to show progress
            
        Returns
        -------
        dict
            Validation results
        """
        splits = self.split(len(X))
        
        # Limit number of folds if specified
        if max_folds is not None and len(splits) > max_folds:
            # Sample evenly across time
            indices = np.linspace(0, len(splits)-1, max_folds, dtype=int)
            splits = [splits[i] for i in indices]
            if verbose:
                print(f"  Limited to {max_folds} folds (sampled from {len(self.split(len(X)))})")
        
        if verbose:
            print(f"\nRolling-Origin Cross-Validation:")
            print(f"  Min train size: {self.min_train_size}")
            print(f"  Window type: {'fixed' if self.window_size else 'expanding'}")
            if self.window_size:
                print(f"  Window size: {self.window_size}")
            print(f"  Step: {self.step}")
            print(f"  Number of folds: {len(splits)}")
        
        all_predictions = []
        all_actuals = []
        all_dates = []
        fold_indices = []
        
        iterator = tqdm(splits, desc="Rolling-origin") if verbose else splits
        
        for fold_num, (train_idx, test_idx) in enumerate(iterator):
            # Get data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            try:
                # Train and predict
                y_pred = model_fn(X_train, y_train, X_test)
                
                # Store results
                all_predictions.append(y_pred)
                all_actuals.append(y_test)
                fold_indices.append(test_idx)
                
                if dates is not None:
                    all_dates.append(dates[test_idx])
                    
            except Exception as e:
                warnings.warn(f"Error in fold {fold_num}: {e}")
                continue
        
        # Concatenate results
        predictions = np.concatenate(all_predictions)
        actuals = np.concatenate(all_actuals)
        
        results = {
            'predictions': predictions,
            'actuals': actuals,
            'fold_indices': fold_indices,
            'n_folds': len(splits)
        }
        
        if dates is not None:
            results['dates'] = pd.DatetimeIndex(np.concatenate(all_dates))
        
        return results
    
    def get_fold_info(self, n_samples: int, max_display: int = 20) -> pd.DataFrame:
        """Get information about folds."""
        splits = self.split(n_samples)
        
        # Limit display
        display_splits = splits[:max_display] if len(splits) > max_display else splits
        
        fold_info = []
        for i, (train_idx, test_idx) in enumerate(display_splits):
            fold_info.append({
                'Fold': i + 1,
                'Train_Size': len(train_idx),
                'Test_Point': test_idx[0],
                'Train_Start': train_idx[0],
                'Train_End': train_idx[-1]
            })
        
        df = pd.DataFrame(fold_info)
        
        if len(splits) > max_display:
            df.loc[len(df)] = ['...'] * len(df.columns)
        
        return df


def rolling_origin_cv(X: np.ndarray,
                     y: np.ndarray,
                     model_fn: Callable,
                     min_train_size: int = 500,
                     window_size: Optional[int] = None,
                     step: int = 21,
                     max_folds: Optional[int] = 100,
                     dates: Optional[pd.DatetimeIndex] = None) -> Dict:
    """
    Convenience function for rolling-origin CV.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target array
    model_fn : callable
        Model function
    min_train_size : int
        Minimum training window size
    window_size : int, optional
        Fixed window size (None = expanding)
    step : int
        Step size between predictions
    max_folds : int, optional
        Maximum number of folds
    dates : pd.DatetimeIndex, optional
        Date index
        
    Returns
    -------
    dict
        Validation results
    """
    validator = RollingOriginCV(
        min_train_size=min_train_size,
        window_size=window_size,
        step=step
    )
    
    results = validator.validate(
        X=X,
        y=y,
        model_fn=model_fn,
        dates=dates,
        max_folds=max_folds,
        verbose=True
    )
    
    return results


if __name__ == '__main__':
    # Test rolling-origin CV
    print("Testing Rolling-Origin Cross-Validation...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Simple model function
    def simple_model(X_train, y_train, X_test):
        return np.full(len(X_test), y_train.mean())
    
    # Test with expanding window
    print("\n1. Expanding Window:")
    validator_expanding = RollingOriginCV(
        min_train_size=500,
        window_size=None,
        step=10
    )
    
    fold_info = validator_expanding.get_fold_info(n_samples, max_display=10)
    print("\nFirst 10 folds:")
    print(fold_info)
    
    results = validator_expanding.validate(
        X=X,
        y=y,
        model_fn=simple_model,
        dates=dates,
        max_folds=20,
        verbose=True
    )
    
    print(f"\nResults:")
    print(f"  Total predictions: {len(results['predictions'])}")
    
    # Test with rolling window
    print("\n2. Rolling Window:")
    validator_rolling = RollingOriginCV(
        min_train_size=250,
        window_size=500,
        step=10
    )
    
    fold_info = validator_rolling.get_fold_info(n_samples, max_display=10)
    print("\nFirst 10 folds:")
    print(fold_info)
    
    results = validator_rolling.validate(
        X=X,
        y=y,
        model_fn=simple_model,
        dates=dates,
        max_folds=20,
        verbose=True
    )
    
    print(f"\nResults:")
    print(f"  Total predictions: {len(results['predictions'])}")
    
    print("\n✓ Rolling-origin CV test successful!")