"""
Data preprocessing module.
Handles missing values, outliers, and data alignment for time series.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings


class DataPreprocessor:
    """Preprocess raw FRED data for modeling."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        data : pd.DataFrame
            Raw data with DatetimeIndex
        """
        self.raw_data = data.copy()
        self.processed_data = None
        self.preprocessing_log = []
        
    def preprocess(self, 
                   method='ffill',
                   max_fill=5,
                   remove_outliers=True,
                   outlier_std=5.0) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Parameters
        ----------
        method : str
            Method for filling missing values: 'ffill', 'interpolate', or 'drop'
        max_fill : int
            Maximum number of consecutive NaNs to fill
        remove_outliers : bool
            Whether to remove outliers
        outlier_std : float
            Number of standard deviations for outlier detection
            
        Returns
        -------
        pd.DataFrame
            Preprocessed data
        """
        df = self.raw_data.copy()
        
        print("Starting data preprocessing...")
        print(f"  Initial shape: {df.shape}")
        
        # 1. Sort by date
        df = df.sort_index()
        self.preprocessing_log.append("Sorted by date")
        
        # 2. Remove duplicates
        df = self._remove_duplicates(df)
        
        # 3. Handle missing values
        df = self._handle_missing_values(df, method=method, max_fill=max_fill)
        
        # 4. Remove outliers
        if remove_outliers:
            df = self._remove_outliers(df, n_std=outlier_std)
        
        # 5. Ensure business day frequency
        df = self._align_to_business_days(df)
        
        # 6. Final validation
        self._validate_processed_data(df)
        
        self.processed_data = df
        
        print(f"  Final shape: {df.shape}")
        print("Preprocessing complete!\n")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate dates."""
        n_duplicates = df.index.duplicated().sum()
        if n_duplicates > 0:
            df = df[~df.index.duplicated(keep='first')]
            self.preprocessing_log.append(f"Removed {n_duplicates} duplicate dates")
            print(f"  Removed {n_duplicates} duplicate dates")
        return df
    
    def _handle_missing_values(self, 
                               df: pd.DataFrame, 
                               method='ffill',
                               max_fill=5) -> pd.DataFrame:
        """
        Handle missing values.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data
        method : str
            Filling method
        max_fill : int
            Maximum consecutive fills
            
        Returns
        -------
        pd.DataFrame
            Data with handled missing values
        """
        print(f"  Handling missing values (method={method})...")
        
        for col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing == 0:
                continue
                
            if method == 'ffill':
                # Forward fill with limit
                df[col] = df[col].ffill(limit=max_fill)
                
            elif method == 'interpolate':
                # Linear interpolation
                df[col] = df[col].interpolate(method='linear', limit=max_fill)
                
            elif method == 'drop':
                # Will drop rows with any NaN later
                pass
            
            remaining = df[col].isna().sum()
            filled = n_missing - remaining
            self.preprocessing_log.append(
                f"{col}: filled {filled} missing values, {remaining} remain"
            )
        
        # Drop rows with any remaining NaN
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        
        if dropped > 0:
            print(f"    Dropped {dropped} rows with remaining NaN values")
            self.preprocessing_log.append(f"Dropped {dropped} rows with NaN")
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, n_std=5.0) -> pd.DataFrame:
        """
        Remove extreme outliers based on rolling z-scores.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data
        n_std : float
            Number of standard deviations for outlier threshold
            
        Returns
        -------
        pd.DataFrame
            Data with outliers removed
        """
        print(f"  Removing outliers (threshold={n_std} std)...")
        
        df_clean = df.copy()
        
        for col in df.columns:
            # Calculate returns for outlier detection
            returns = df[col].pct_change()
            
            # Rolling z-score (30-day window)
            rolling_mean = returns.rolling(window=30, min_periods=10).mean()
            rolling_std = returns.rolling(window=30, min_periods=10).std()
            z_scores = (returns - rolling_mean) / rolling_std
            
            # Identify outliers
            outliers = np.abs(z_scores) > n_std
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                # Replace outliers with interpolated values
                df_clean.loc[outliers, col] = np.nan
                df_clean[col] = df_clean[col].interpolate(method='linear')
                
                print(f"    {col}: removed {n_outliers} outliers")
                self.preprocessing_log.append(
                    f"{col}: removed {n_outliers} outliers"
                )
        
        return df_clean
    
    def _align_to_business_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align data to business day calendar.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input data
            
        Returns
        -------
        pd.DataFrame
            Data aligned to business days
        """
        # Create business day range
        bday_range = pd.bdate_range(start=df.index[0], end=df.index[-1])
        
        # Reindex to business days, forward fill
        df_aligned = df.reindex(bday_range, method='ffill')
        
        added = len(df_aligned) - len(df)
        if added > 0:
            print(f"  Aligned to business days: added {added} dates")
            self.preprocessing_log.append(f"Aligned to {len(df_aligned)} business days")
        
        return df_aligned
    
    def _validate_processed_data(self, df: pd.DataFrame):
        """Validate processed data quality."""
        # Check for NaN
        if df.isna().any().any():
            warnings.warn("Processed data still contains NaN values!")
        
        # Check for infinite values
        if np.isinf(df.select_dtypes(include=[np.number])).any().any():
            warnings.warn("Processed data contains infinite values!")
        
        # Check for sufficient data
        if len(df) < 252:  # Less than 1 year of daily data
            warnings.warn(
                f"Only {len(df)} observations after preprocessing. "
                "Consider using longer date range."
            )
    
    def get_preprocessing_summary(self) -> str:
        """Get summary of preprocessing steps."""
        summary = "\n".join([
            "Preprocessing Summary:",
            "=" * 50
        ] + self.preprocessing_log)
        return summary
    
    def save_processed_data(self, filepath='data/processed/preprocessed_data.csv'):
        """Save preprocessed data."""
        if self.processed_data is None:
            raise ValueError("No processed data. Run preprocess() first.")
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        self.processed_data.to_csv(filepath)
        print(f"\nProcessed data saved to: {filepath}")


def preprocess_data(raw_data: pd.DataFrame,
                    method='ffill',
                    max_fill=5,
                    remove_outliers=True,
                    output_path='data/processed/preprocessed_data.csv') -> pd.DataFrame:
    """
    Convenience function for data preprocessing.
    
    Parameters
    ----------
    raw_data : pd.DataFrame
        Raw input data
    method : str
        Missing value handling method
    max_fill : int
        Maximum consecutive fills
    remove_outliers : bool
        Whether to remove outliers
    output_path : str
        Path to save processed data
        
    Returns
    -------
    pd.DataFrame
        Preprocessed data
    """
    preprocessor = DataPreprocessor(raw_data)
    processed = preprocessor.preprocess(
        method=method,
        max_fill=max_fill,
        remove_outliers=remove_outliers
    )
    
    print(preprocessor.get_preprocessing_summary())
    
    preprocessor.save_processed_data(filepath=output_path)
    
    return processed


if __name__ == '__main__':
    # Example usage
    import sys
    sys.path.append('.')
    from src.data.fetch_data import FREDDataFetcher
    
    # Load raw data
    fetcher = FREDDataFetcher()
    raw_data = fetcher.load_raw_data('data/raw/fred_data.csv')
    
    # Preprocess
    processed = preprocess_data(raw_data)
    
    print("\nProcessed data info:")
    print(processed.info())