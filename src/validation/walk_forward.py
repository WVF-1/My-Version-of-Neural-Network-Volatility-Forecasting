"""
Walk-forward validation for time series models.
Implements expanding and rolling window strategies.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Callable, Optional
from tqdm import tqdm
import warnings


class WalkForwardValidator:
    """
    Walk-forward validation for time series.
    
    This is the gold standard for time series model validation,
    as it respects temporal ordering and avoids look-ahead bias.
    """
    
    def __init__(self,
                 initial_train_size: int,
                 step_size: int = 21,
                 window_type: str = 'expanding'):
        """
        Initialize walk-forward validator.
        
        Parameters
        ----------
        initial_train_size : int
            Size of initial training set
        step_size : int
            Number of periods to step forward in each iteration
        window_type : str
            'expanding' (growing window) or 'rolling' (fixed window)
        """
        self.initial_train_size = initial_train_size
        self.step_size = step_size
        self.window_type = window_type
        
        if window_type not in ['expanding', 'rolling']:
            raise ValueError("window_type must be 'expanding' or 'rolling'")
    
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
        if self.initial_train_size >= n_samples:
            raise ValueError("initial_train_size must be less than n_samples")
        
        splits = []
        
        # Start with initial training size
        train_end = self.initial_train_size
        
        while train_end + self.step_size <= n_samples:
            # Test indices
            test_start = train_end
            test_end = min(train_end + self.step_size, n_samples)
            test_idx = np.arange(test_start, test_end)
            
            # Train indices
            if self.window_type == 'expanding':
                # Expanding window: use all data from start
                train_idx = np.arange(0, train_end)
            else:
                # Rolling window: keep fixed window size
                train_start = max(0, train_end - self.initial_train_size)
                train_idx = np.arange(train_start, train_end)
            
            splits.append((train_idx, test_idx))
            
            # Move forward
            train_end = test_end
        
        return splits
    
    def validate(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 model_fn: Callable,
                 dates: Optional[pd.DatetimeIndex] = None,
                 verbose: bool = True) -> Dict:
        """
        Perform walk-forward validation.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target array
        model_fn : callable
            Function that takes (X_train, y_train, X_test) and returns predictions
            Signature: model_fn(X_train, y_train, X_test) -> y_pred
        dates : pd.DatetimeIndex, optional
            Date index for results
        verbose : bool
            Whether to show progress bar
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'predictions': array of all predictions
            - 'actuals': array of all actual values
            - 'dates': dates corresponding to predictions
            - 'fold_indices': list of test indices for each fold
        """
        # Generate splits
        splits = self.split(len(X))
        
        if verbose:
            print(f"\nWalk-forward validation:")
            print(f"  Window type: {self.window_type}")
            print(f"  Initial train size: {self.initial_train_size}")
            print(f"  Step size: {self.step_size}")
            print(f"  Number of folds: {len(splits)}")
        
        # Collect predictions
        all_predictions = []
        all_actuals = []
        all_dates = []
        fold_indices = []
        
        iterator = tqdm(splits, desc="Walk-forward folds") if verbose else splits
        
        for fold_num, (train_idx, test_idx) in enumerate(iterator):
            # Get train/test data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train and predict
            try:
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
        """
        Get information about folds.
        
        Parameters
        ----------
        n_samples : int
            Total number of samples
            
        Returns
        -------
        pd.DataFrame
            DataFrame with fold information
        """
        splits = self.split(n_samples)
        
        fold_info = []
        for i, (train_idx, test_idx) in enumerate(splits):
            fold_info.append({
                'Fold': i + 1,
                'Train_Start': train_idx[0],
                'Train_End': train_idx[-1],
                'Train_Size': len(train_idx),
                'Test_Start': test_idx[0],
                'Test_End': test_idx[-1],
                'Test_Size': len(test_idx)
            })
        
        return pd.DataFrame(fold_info)


def walk_forward_cv(X: np.ndarray,
                   y: np.ndarray,
                   model_fn: Callable,
                   initial_train_size: int = 1000,
                   step_size: int = 21,
                   window_type: str = 'expanding',
                   dates: Optional[pd.DatetimeIndex] = None) -> Dict:
    """
    Convenience function for walk-forward cross-validation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target array
    model_fn : callable
        Model function
    initial_train_size : int
        Initial training set size
    step_size : int
        Step size for moving forward
    window_type : str
        'expanding' or 'rolling'
    dates : pd.DatetimeIndex, optional
        Date index
        
    Returns
    -------
    dict
        Validation results
    """
    validator = WalkForwardValidator(
        initial_train_size=initial_train_size,
        step_size=step_size,
        window_type=window_type
    )
    
    results = validator.validate(
        X=X,
        y=y,
        model_fn=model_fn,
        dates=dates,
        verbose=True
    )
    
    return results


if __name__ == '__main__':
    # Test walk-forward validation
    print("Testing walk-forward validation...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 2000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Simple model function for testing
    def simple_model(X_train, y_train, X_test):
        # Simple mean prediction
        return np.full(len(X_test), y_train.mean())
    
    # Test validator
    validator = WalkForwardValidator(
        initial_train_size=1000,
        step_size=50,
        window_type='expanding'
    )
    
    # Get fold info
    fold_info = validator.get_fold_info(n_samples)
    print("\nFold information:")
    print(fold_info.head(10))
    
    # Run validation
    results = validator.validate(
        X=X,
        y=y,
        model_fn=simple_model,
        dates=dates,
        verbose=True
    )
    
    print(f"\nValidation results:")
    print(f"  Total predictions: {len(results['predictions'])}")
    print(f"  Number of folds: {results['n_folds']}")
    print(f"  Date range: {results['dates'][0]} to {results['dates'][-1]}")
    
    print("\n✓ Walk-forward validation test successful!")