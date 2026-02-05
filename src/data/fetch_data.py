"""
Data fetching module for FRED macro-financial data.
Handles API authentication, data retrieval, and initial validation.
"""

import os
import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FREDDataFetcher:
    """Fetch and validate FRED economic data."""
    
    # FRED series mapping
    SERIES_MAP = {
        'SP500': 'SP500',
        'DGS10': 'DGS10',
        'GOLD': 'GOLDAMGBD228NLBM',
        'COMMODITY': 'PALLFNFINDEXQ',
        'VIX': 'VIXCLS'
    }
    
    def __init__(self, api_key=None):
        """
        Initialize FRED API connection.
        
        Parameters
        ----------
        api_key : str, optional
            FRED API key. If None, reads from FRED_API_KEY environment variable.
        """
        if api_key is None:
            api_key = os.getenv('FRED_API_KEY')
            
        if api_key is None:
            raise ValueError(
                "FRED API key required. Set FRED_API_KEY environment variable "
                "or pass api_key parameter. Get key at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        
        self.fred = Fred(api_key=api_key)
        self.data = None
        
    def fetch_data(self, start_date='2010-01-01', end_date=None):
        """
        Fetch all required FRED series.
        
        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format. If None, uses today.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with all series aligned by date
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Fetching FRED data from {start_date} to {end_date}...")
        
        data_dict = {}
        
        for name, series_id in self.SERIES_MAP.items():
            try:
                print(f"  Fetching {name} ({series_id})...")
                series = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date
                )
                data_dict[name] = series
                print(f"    ✓ Retrieved {len(series)} observations")
                
            except Exception as e:
                print(f"    ✗ Error fetching {name}: {e}")
                data_dict[name] = pd.Series(dtype=float)
        
        # Combine into single DataFrame
        self.data = pd.DataFrame(data_dict)
        self.data.index.name = 'date'
        
        # Basic validation
        self._validate_data()
        
        return self.data
    
    def _validate_data(self):
        """Validate fetched data quality."""
        if self.data is None or self.data.empty:
            raise ValueError("No data available. Run fetch_data() first.")
        
        print("\nData Validation Summary:")
        print(f"  Date range: {self.data.index[0]} to {self.data.index[-1]}")
        print(f"  Total observations: {len(self.data)}")
        print(f"\nMissing values by series:")
        
        for col in self.data.columns:
            missing = self.data[col].isna().sum()
            missing_pct = 100 * missing / len(self.data)
            print(f"    {col}: {missing} ({missing_pct:.2f}%)")
        
        # Check for sufficient data
        min_obs = 500
        for col in self.data.columns:
            valid_obs = self.data[col].notna().sum()
            if valid_obs < min_obs:
                warnings.warn(
                    f"{col} has only {valid_obs} valid observations "
                    f"(minimum recommended: {min_obs})"
                )
    
    def save_raw_data(self, filepath='data/raw/fred_data.csv'):
        """
        Save raw data to CSV.
        
        Parameters
        ----------
        filepath : str
            Path to save CSV file
        """
        if self.data is None:
            raise ValueError("No data to save. Run fetch_data() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.data.to_csv(filepath)
        print(f"\nRaw data saved to: {filepath}")
        
    def load_raw_data(self, filepath='data/raw/fred_data.csv'):
        """
        Load previously saved raw data.
        
        Parameters
        ----------
        filepath : str
            Path to CSV file
            
        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Loaded data from: {filepath}")
        self._validate_data()
        return self.data
    
    def get_series_info(self, series_name):
        """
        Get metadata for a specific series.
        
        Parameters
        ----------
        series_name : str
            Name of series (e.g., 'SP500', 'DGS10')
            
        Returns
        -------
        dict
            Series metadata
        """
        if series_name not in self.SERIES_MAP:
            raise ValueError(f"Unknown series: {series_name}")
        
        series_id = self.SERIES_MAP[series_name]
        info = self.fred.get_series_info(series_id)
        
        return {
            'id': info['id'],
            'title': info['title'],
            'units': info['units'],
            'frequency': info['frequency'],
            'seasonal_adjustment': info['seasonal_adjustment'],
            'last_updated': info['last_updated']
        }


def fetch_and_save_data(api_key=None, start_date='2010-01-01', 
                        output_path='data/raw/fred_data.csv'):
    """
    Convenience function to fetch and save FRED data.
    
    Parameters
    ----------
    api_key : str, optional
        FRED API key
    start_date : str
        Start date for data
    output_path : str
        Path to save output CSV
        
    Returns
    -------
    pd.DataFrame
        Fetched data
    """
    fetcher = FREDDataFetcher(api_key=api_key)
    data = fetcher.fetch_data(start_date=start_date)
    fetcher.save_raw_data(filepath=output_path)
    return data


if __name__ == '__main__':
    # Example usage
    data = fetch_and_save_data()
    print("\nData shape:", data.shape)
    print("\nFirst few rows:")
    print(data.head())