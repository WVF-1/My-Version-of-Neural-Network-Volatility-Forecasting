"""
Time-series K-Fold cross-validation.
Uses expanding window approach to maintain temporal ordering.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable, Optional
from tqdm import tqdm
import warnings


class TimeSeriesKFold:
    """
    Time series K-Fold cross-validation.
    
    Unlike standard k-fold, this respects temporal ordering by:
    1. Using only past data for training
    2. Using expanding windows (not random splits)
    3. Never using future data to predict past
    """
    
    def __init__(self, n_splits: int = 5, gap: int = 0):
        """
        Initialize time series k-fold validator.
        
        Parameters
        ----------
        n_splits : int
            Number of splits
        gap : int
            Number of samples to skip between train and test
            (useful to avoid leakage in overlapping features)
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        
        self.n_splits = n_splits
        self.gap = gap
    
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
        # Calculate test size for each fold
        test_size = n_samples // (self.n_splits + 1)
        
        if test_size < 1:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_splits} splits"
            )
        
        splits = []
        
        for i in range(self.n_splits):
            # Calculate split points
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                test_end = n_samples
            
            # Train on all data before test (with gap)
            train_end = test_start - self.gap
            
            if train_end < test_size:
                # Skip if not enough training data
                continue
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def validate(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 model_fn: Callable,
                 dates: Optional[pd.DatetimeIndex] = None,
                 verbose: bool = True) -> Dict:
        """
        Perform time-series k-fold validation.
        
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
        verbose : bool
            Whether to show progress
            
        Returns
        -------
        dict
            Validation results
        """
        splits = self.split(len(X))
        
        if verbose:
            print(f"\nTime-Series K-Fold Validation:")
            print(f"  Number of splits: {len(splits)}")
            print(f"  Gap: {self.gap}")
        
        all_predictions = []
        all_actuals = []
        all_dates = []
        fold_indices = []
        
        iterator = tqdm(splits, desc="K-Fold") if verbose else splits
        
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
    
    def get_fold_info(self, n_samples: int) -> pd.DataFrame:
        """Get information about each fold."""
        splits = self.split(n_samples)
        
        fold_info = []
        for i, (train_idx, test_idx) in enumerate(splits):
            fold_info.append({
                'Fold': i + 1,
                'Train_Size': len(train_idx),
                'Test_Size': len(test_idx),
                'Train_Start': train_idx[0],
                'Train_End': train_idx[-1],
                'Test_Start': test_idx[0],
                'Test_End': test_idx[-1]
            })
        
        return pd.DataFrame(fold_info)


def timeseries_kfold_cv(X: np.ndarray,
                       y: np.ndarray,
                       model_fn: Callable,
                       n_splits: int = 5,
                       gap: int = 0,
                       dates: Optional[pd.DatetimeIndex] = None) -> Dict:
    """
    Convenience function for time-series k-fold CV.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target array
    model_fn : callable
        Model function
    n_splits : int
        Number of splits
    gap : int
        Gap between train and test
    dates : pd.DatetimeIndex, optional
        Date index
        
    Returns
    -------
    dict
        Validation results
    """
    kfold = TimeSeriesKFold(n_splits=n_splits, gap=gap)
    
    results = kfold.validate(
        X=X,
        y=y,
        model_fn=model_fn,
        dates=dates,
        verbose=True
    )
    
    return results


if __name__ == '__main__':
    # Test time-series k-fold
    print("Testing Time-Series K-Fold validation...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 2000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Simple model function
    def simple_model(X_train, y_train, X_test):
        return np.full(len(X_test), y_train.mean())
    
    # Test k-fold
    kfold = TimeSeriesKFold(n_splits=5, gap=21)
    
    # Get fold info
    fold_info = kfold.get_fold_info(n_samples)
    print("\nFold information:")
    print(fold_info)
    
    # Run validation
    results = kfold.validate(
        X=X,
        y=y,
        model_fn=simple_model,
        dates=dates,
        verbose=True
    )
    
    print(f"\nValidation results:")
    print(f"  Total predictions: {len(results['predictions'])}")
    print(f"  Number of folds: {results['n_folds']}")
    
    print("\n✓ Time-series k-fold test successful!")