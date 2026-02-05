"""
LSTM neural network for volatility forecasting.
PyTorch implementation with proper regularization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class VolatilityLSTM(nn.Module):
    """
    LSTM neural network for volatility forecasting.
    
    Architecture:
    - Input layer
    - LSTM layers with dropout
    - Fully connected output layer
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 1):
        """
        Initialize LSTM model.
        
        Parameters
        ----------
        input_size : int
            Number of input features
        hidden_size : int
            Number of hidden units in LSTM
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout rate (applied between LSTM layers)
        output_size : int
            Number of outputs (typically 1 for volatility)
        """
        super(VolatilityLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout after LSTM
        self.dropout_layer = nn.Dropout(dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                elif 'fc' in name:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns
        -------
        torch.Tensor
            Output predictions of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Dropout
        lstm_out = self.dropout_layer(lstm_out)
        
        # Fully connected layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        # Ensure non-negative output (volatility must be positive)
        out = torch.abs(out)
        
        return out
    
    def get_model_size(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for time series data."""
    
    def __init__(self, 
                 features: np.ndarray, 
                 targets: np.ndarray,
                 sequence_length: int = 60):
        """
        Initialize dataset.
        
        Parameters
        ----------
        features : np.ndarray
            Feature array of shape (n_samples, n_features)
        targets : np.ndarray
            Target array of shape (n_samples,) or (n_samples, 1)
        sequence_length : int
            Length of input sequences
        """
        self.features = features
        self.targets = targets.reshape(-1, 1)  # Ensure 2D
        self.sequence_length = sequence_length
        
        # Validate inputs
        if len(features) != len(targets):
            raise ValueError("Features and targets must have same length")
        
        if len(features) < sequence_length:
            raise ValueError(
                f"Not enough data. Need at least {sequence_length} samples, "
                f"got {len(features)}"
            )
    
    def __len__(self):
        """Return number of sequences."""
        return len(self.features) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        """
        Get a single sequence and target.
        
        Parameters
        ----------
        idx : int
            Index of sequence
            
        Returns
        -------
        tuple
            (sequence, target) where:
            - sequence: (sequence_length, n_features)
            - target: (1,)
        """
        # Get sequence
        seq_start = idx
        seq_end = idx + self.sequence_length
        
        sequence = self.features[seq_start:seq_end]
        target = self.targets[seq_end - 1]  # Target at end of sequence
        
        return (
            torch.FloatTensor(sequence),
            torch.FloatTensor(target)
        )


def create_sequences(features: np.ndarray,
                    targets: np.ndarray,
                    sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training (alternative to Dataset).
    
    Parameters
    ----------
    features : np.ndarray
        Feature array (n_samples, n_features)
    targets : np.ndarray
        Target array (n_samples,)
    sequence_length : int
        Length of sequences
        
    Returns
    -------
    X : np.ndarray
        Sequences of shape (n_sequences, sequence_length, n_features)
    y : np.ndarray
        Targets of shape (n_sequences,)
    """
    X, y = [], []
    
    for i in range(len(features) - sequence_length + 1):
        X.append(features[i:i + sequence_length])
        y.append(targets[i + sequence_length - 1])
    
    return np.array(X), np.array(y)


def get_device():
    """Get available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


if __name__ == '__main__':
    # Test model
    print("Testing VolatilityLSTM model...")
    
    # Model parameters
    input_size = 10
    hidden_size = 64
    num_layers = 2
    batch_size = 32
    seq_length = 60
    
    # Create model
    model = VolatilityLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.2
    )
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nNumber of parameters: {model.get_model_size():,}")
    
    # Test forward pass
    device = get_device()
    model = model.to(device)
    
    # Create dummy data
    x = torch.randn(batch_size, seq_length, input_size).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions: {output[:5].squeeze().cpu().numpy()}")
    
    print("\n✓ Model test successful!")