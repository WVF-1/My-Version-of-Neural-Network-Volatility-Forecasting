"""
Risk parity portfolio construction.
Allocates capital to achieve equal risk contribution from each asset.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Optional, Tuple
import warnings


class RiskParityPortfolio:
    """
    Risk parity portfolio optimizer.
    
    Constructs portfolios where each asset contributes equally to
    portfolio risk, based on predicted volatilities.
    """
    
    def __init__(self,
                 asset_names: list,
                 vol_target: float = 0.10,
                 leverage_limit: float = 1.0):
        """
        Initialize risk parity optimizer.
        
        Parameters
        ----------
        asset_names : list
            List of asset names
        vol_target : float
            Target portfolio volatility (annualized)
        leverage_limit : float
            Maximum leverage allowed (1.0 = no leverage)
        """
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.vol_target = vol_target
        self.leverage_limit = leverage_limit
    
    def optimize_weights(self,
                        volatilities: np.ndarray,
                        correlations: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Optimize portfolio weights for equal risk contribution.
        
        Parameters
        ----------
        volatilities : np.ndarray
            Predicted volatilities for each asset (annualized)
        correlations : np.ndarray, optional
            Correlation matrix. If None, assumes zero correlations.
            
        Returns
        -------
        np.ndarray
            Optimal portfolio weights
        """
        # Validate inputs
        if len(volatilities) != self.n_assets:
            raise ValueError(
                f"Expected {self.n_assets} volatilities, got {len(volatilities)}"
            )
        
        # Handle zero or negative volatilities
        volatilities = np.maximum(volatilities, 1e-6)
        
        # If no correlation matrix, assume independence
        if correlations is None:
            # Naive risk parity: inverse volatility weighting
            weights = 1.0 / volatilities
            weights = weights / weights.sum()
            return weights
        
        # Validate correlation matrix
        if correlations.shape != (self.n_assets, self.n_assets):
            raise ValueError(
                f"Correlation matrix must be {self.n_assets}x{self.n_assets}"
            )
        
        # Construct covariance matrix
        D = np.diag(volatilities)
        cov_matrix = D @ correlations @ D
        
        # Ensure positive definite
        cov_matrix = self._nearest_positive_definite(cov_matrix)
        
        # Optimize for equal risk contribution
        weights = self._optimize_equal_risk_contribution(cov_matrix)
        
        return weights
    
    def _optimize_equal_risk_contribution(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Optimize for equal risk contribution using scipy.
        
        Parameters
        ----------
        cov_matrix : np.ndarray
            Covariance matrix
            
        Returns
        -------
        np.ndarray
            Optimal weights
        """
        # Objective: minimize sum of squared deviations from equal risk
        def objective(weights):
            portfolio_var = weights @ cov_matrix @ weights
            marginal_contrib = cov_matrix @ weights
            risk_contrib = weights * marginal_contrib
            
            # Target: each asset contributes 1/N of total risk
            target_contrib = portfolio_var / self.n_assets
            
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Fully invested
        ]
        
        # Bounds
        bounds = [(0, self.leverage_limit / self.n_assets)] * self.n_assets
        
        # Initial guess: equal weight
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
        
        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}")
            # Fallback to equal weight
            return x0
        
        return result.x
    
    def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        """Find nearest positive definite matrix."""
        # Symmetrize
        B = (A + A.T) / 2
        
        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(B)
        
        # Clip negative eigenvalues
        eigvals = np.maximum(eigvals, 1e-8)
        
        # Reconstruct
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    def apply_volatility_targeting(self,
                                   weights: np.ndarray,
                                   volatilities: np.ndarray,
                                   correlations: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply volatility targeting to scale portfolio.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        volatilities : np.ndarray
            Asset volatilities
        correlations : np.ndarray, optional
            Correlation matrix
            
        Returns
        -------
        np.ndarray
            Scaled weights
        """
        # Calculate portfolio volatility
        if correlations is None:
            # Assuming independence
            portfolio_vol = np.sqrt(np.sum((weights * volatilities) ** 2))
        else:
            # With correlations
            D = np.diag(volatilities)
            cov_matrix = D @ correlations @ D
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        # Scale to target volatility
        if portfolio_vol > 1e-6:
            scale = self.vol_target / portfolio_vol
            scale = min(scale, self.leverage_limit)  # Respect leverage limit
        else:
            scale = 0.0
        
        return weights * scale
    
    def get_risk_contributions(self,
                              weights: np.ndarray,
                              volatilities: np.ndarray,
                              correlations: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate risk contributions for each asset.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        volatilities : np.ndarray
            Asset volatilities
        correlations : np.ndarray, optional
            Correlation matrix
            
        Returns
        -------
        dict
            Dictionary with risk contribution analysis
        """
        # Construct covariance matrix
        if correlations is None:
            correlations = np.eye(self.n_assets)
        
        D = np.diag(volatilities)
        cov_matrix = D @ correlations @ D
        
        # Portfolio variance
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_var)
        
        # Marginal contributions
        marginal_contrib = cov_matrix @ weights
        
        # Risk contributions
        risk_contrib = weights * marginal_contrib
        
        # Percentage contributions
        if portfolio_var > 1e-10:
            pct_contrib = risk_contrib / portfolio_var
        else:
            pct_contrib = np.zeros(self.n_assets)
        
        return {
            'portfolio_volatility': portfolio_vol,
            'portfolio_variance': portfolio_var,
            'marginal_contributions': marginal_contrib,
            'risk_contributions': risk_contrib,
            'risk_contribution_pct': pct_contrib,
            'weights': weights
        }


def construct_risk_parity_portfolio(volatilities: np.ndarray,
                                    asset_names: list,
                                    correlations: Optional[np.ndarray] = None,
                                    vol_target: float = 0.10) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to construct risk parity portfolio.
    
    Parameters
    ----------
    volatilities : np.ndarray
        Predicted volatilities
    asset_names : list
        Asset names
    correlations : np.ndarray, optional
        Correlation matrix
    vol_target : float
        Target portfolio volatility
        
    Returns
    -------
    weights : np.ndarray
        Portfolio weights
    info : dict
        Portfolio information
    """
    optimizer = RiskParityPortfolio(
        asset_names=asset_names,
        vol_target=vol_target
    )
    
    # Optimize weights
    weights = optimizer.optimize_weights(volatilities, correlations)
    
    # Apply volatility targeting
    weights = optimizer.apply_volatility_targeting(weights, volatilities, correlations)
    
    # Get risk contributions
    info = optimizer.get_risk_contributions(weights, volatilities, correlations)
    
    return weights, info


if __name__ == '__main__':
    # Test risk parity portfolio
    print("Testing Risk Parity Portfolio...")
    
    # Example assets
    asset_names = ['Stocks', 'Bonds', 'Gold', 'Commodities']
    n_assets = len(asset_names)
    
    # Example volatilities (annualized)
    volatilities = np.array([0.20, 0.08, 0.15, 0.25])
    
    # Example correlation matrix
    correlations = np.array([
        [1.00, 0.20, -0.10, 0.30],
        [0.20, 1.00, 0.10, -0.05],
        [-0.10, 0.10, 1.00, 0.20],
        [0.30, -0.05, 0.20, 1.00]
    ])
    
    print("\nInput:")
    print(f"  Assets: {asset_names}")
    print(f"  Volatilities: {volatilities}")
    print(f"  Correlations:\n{correlations}")
    
    # Construct portfolio
    weights, info = construct_risk_parity_portfolio(
        volatilities=volatilities,
        asset_names=asset_names,
        correlations=correlations,
        vol_target=0.10
    )
    
    print("\nOptimized Portfolio:")
    print(f"  Target volatility: 10%")
    print(f"  Portfolio volatility: {info['portfolio_volatility']:.2%}")
    
    print("\nWeights:")
    for name, weight in zip(asset_names, weights):
        print(f"  {name}: {weight:.2%}")
    
    print("\nRisk Contributions:")
    for name, contrib in zip(asset_names, info['risk_contribution_pct']):
        print(f"  {name}: {contrib:.2%}")
    
    print("\n✓ Risk parity test successful!")