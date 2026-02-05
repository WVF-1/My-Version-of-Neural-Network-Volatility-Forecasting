"""
Feature engineering module for volatility forecasting.
Creates returns, rolling volatility, EWMA, and target variables.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import warnings


class FeatureEngineer:
    """Engineer features for volatility forecasting."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize feature engineer.
        
        Parameters
        ----------
        data : pd.DataFrame
            Preprocessed price data
        """
        self.data = data.copy()
        self.features = None
        self.targets = None
        
    def create_all_features(self,
                           vol_windows=[5, 21, 63],
                           ewma_spans=[21, 63],
                           target_horizon=21,
                           include_vix=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create complete feature set.
        
        Parameters
        ----------
        vol_windows : list
            Windows for rolling volatility (in days)
        ewma_spans : list
            Spans for EWMA volatility
        target_horizon : int
            Forward horizon for target volatility (in days)
        include_vix : bool
            Whether to include VIX as feature
            
        Returns
        -------
        features : pd.DataFrame
            Feature matrix
        targets : pd.DataFrame
            Target variables (forward realized volatility)
        """
        print("Creating features...")
        
        # Calculate returns
        returns = self._calculate_returns()
        
        # Create features for each asset
        feature_list = []
        target_list = []
        
        # Get asset columns (exclude VIX from assets)
        asset_cols = [col for col in self.data.columns if col != 'VIX']
        
        for asset in asset_cols:
            print(f"  Processing {asset}...")
            
            asset_features = pd.DataFrame(index=self.data.index)
            
            # 1. Returns
            asset_features[f'{asset}_return'] = returns[asset]
            
            # 2. Rolling volatilities
            for window in vol_windows:
                vol = self._calculate_rolling_volatility(returns[asset], window)
                asset_features[f'{asset}_vol_{window}d'] = vol
            
            # 3. EWMA volatilities
            for span in ewma_spans:
                ewma_vol = self._calculate_ewma_volatility(returns[asset], span)
                asset_features[f'{asset}_ewma_{span}d'] = ewma_vol
            
            # 4. Lagged returns
            for lag in [1, 5, 21]:
                asset_features[f'{asset}_return_lag{lag}'] = returns[asset].shift(lag)
            
            # 5. Return autocorrelation features
            asset_features[f'{asset}_return_abs'] = np.abs(returns[asset])
            asset_features[f'{asset}_return_sq'] = returns[asset] ** 2
            
            # 6. VIX features (if available and requested)
            if include_vix and 'VIX' in self.data.columns:
                asset_features[f'{asset}_vix'] = self.data['VIX']
                asset_features[f'{asset}_vix_change'] = self.data['VIX'].pct_change()
            
            # Create target: forward realized volatility
            target = self._calculate_forward_volatility(
                returns[asset], 
                horizon=target_horizon
            )
            
            feature_list.append(asset_features)
            target_list.append(target.rename(f'{asset}_target_vol'))
        
        # Combine all features and targets
        self.features = pd.concat(feature_list, axis=1)
        self.targets = pd.concat(target_list, axis=1)
        
        # Remove rows with NaN (due to lags and forward-looking targets)
        valid_idx = self.features.notna().all(axis=1) & self.targets.notna().all(axis=1)
        self.features = self.features[valid_idx]
        self.targets = self.targets[valid_idx]
        
        print(f"\nFeature engineering complete!")
        print(f"  Feature shape: {self.features.shape}")
        print(f"  Target shape: {self.targets.shape}")
        print(f"  Date range: {self.features.index[0]} to {self.features.index[-1]}")
        
        return self.features, self.targets
    
    def _calculate_returns(self) -> pd.DataFrame:
        """Calculate log returns."""
        returns = np.log(self.data / self.data.shift(1))
        return returns
    
    def _calculate_rolling_volatility(self, 
                                     returns: pd.Series, 
                                     window: int) -> pd.Series:
        """
        Calculate rolling realized volatility.
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        window : int
            Rolling window size
            
        Returns
        -------
        pd.Series
            Rolling volatility (annualized)
        """
        # Annualization factor for daily data
        annualization_factor = np.sqrt(252)
        
        vol = returns.rolling(window=window, min_periods=window).std() * annualization_factor
        return vol
    
    def _calculate_ewma_volatility(self,
                                   returns: pd.Series,
                                   span: int) -> pd.Series:
        """
        Calculate EWMA volatility.
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        span : int
            EWMA span
            
        Returns
        -------
        pd.Series
            EWMA volatility (annualized)
        """
        annualization_factor = np.sqrt(252)
        
        ewma_var = returns.ewm(span=span, min_periods=span).var()
        ewma_vol = np.sqrt(ewma_var) * annualization_factor
        
        return ewma_vol
    
    def _calculate_forward_volatility(self,
                                      returns: pd.Series,
                                      horizon: int) -> pd.Series:
        """
        Calculate forward realized volatility (target variable).
        
        Parameters
        ----------
        returns : pd.Series
            Return series
        horizon : int
            Forward horizon in days
            
        Returns
        -------
        pd.Series
            Forward realized volatility (annualized)
        """
        annualization_factor = np.sqrt(252)
        
        # Use expanding window for forward calculation
        forward_vol = returns.rolling(window=horizon, min_periods=horizon).std().shift(-horizon)
        forward_vol = forward_vol * annualization_factor
        
        return forward_vol
    
    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary statistics of features."""
        if self.features is None:
            raise ValueError("Features not created yet. Run create_all_features().")
        
        summary = self.features.describe().T
        summary['missing'] = self.features.isna().sum()
        
        return summary
    
    def get_asset_features(self, asset: str) -> pd.DataFrame:
        """
        Extract features for a specific asset.
        
        Parameters
        ----------
        asset : str
            Asset name (e.g., 'SP500')
            
        Returns
        -------
        pd.DataFrame
            Features for the specified asset
        """
        if self.features is None:
            raise ValueError("Features not created yet.")
        
        # Get columns for this asset
        asset_cols = [col for col in self.features.columns if col.startswith(asset)]
        
        return self.features[asset_cols]
    
    def get_asset_target(self, asset: str) -> pd.Series:
        """
        Extract target for a specific asset.
        
        Parameters
        ----------
        asset : str
            Asset name
            
        Returns
        -------
        pd.Series
            Target variable
        """
        if self.targets is None:
            raise ValueError("Targets not created yet.")
        
        target_col = f'{asset}_target_vol'
        return self.targets[target_col]
    
    def save_features(self, 
                     features_path='data/processed/features.csv',
                     targets_path='data/processed/targets.csv'):
        """Save features and targets to CSV."""
        if self.features is None or self.targets is None:
            raise ValueError("Features not created yet.")
        
        import os
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        
        self.features.to_csv(features_path)
        self.targets.to_csv(targets_path)
        
        print(f"\nFeatures saved to: {features_path}")
        print(f"Targets saved to: {targets_path}")
    
    def load_features(self,
                     features_path='data/processed/features.csv',
                     targets_path='data/processed/targets.csv'):
        """Load previously saved features and targets."""
        self.features = pd.read_csv(features_path, index_col=0, parse_dates=True)
        self.targets = pd.read_csv(targets_path, index_col=0, parse_dates=True)
        
        print(f"Features loaded: {self.features.shape}")
        print(f"Targets loaded: {self.targets.shape}")
        
        return self.features, self.targets


def create_features(data: pd.DataFrame,
                   vol_windows=[5, 21, 63],
                   ewma_spans=[21, 63],
                   target_horizon=21,
                   output_dir='data/processed/') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function for feature engineering.
    
    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed price data
    vol_windows : list
        Rolling volatility windows
    ewma_spans : list
        EWMA spans
    target_horizon : int
        Target horizon for forward volatility
    output_dir : str
        Directory to save outputs
        
    Returns
    -------
    features : pd.DataFrame
        Feature matrix
    targets : pd.DataFrame
        Target variables
    """
    engineer = FeatureEngineer(data)
    features, targets = engineer.create_all_features(
        vol_windows=vol_windows,
        ewma_spans=ewma_spans,
        target_horizon=target_horizon
    )
    
    engineer.save_features(
        features_path=f'{output_dir}/features.csv',
        targets_path=f'{output_dir}/targets.csv'
    )
    
    print("\nFeature summary:")
    print(engineer.get_feature_summary().head(10))
    
    return features, targets


if __name__ == '__main__':
    # Example usage
    import sys
    sys.path.append('.')
    
    # Load preprocessed data
    data = pd.read_csv('data/processed/preprocessed_data.csv', 
                       index_col=0, parse_dates=True)
    
    # Create features
    features, targets = create_features(data)
    
    print("\nSample features:")
    print(features.head())
    print("\nSample targets:")
    print(targets.head())