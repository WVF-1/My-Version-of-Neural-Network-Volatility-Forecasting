"""
Training module for LSTM volatility forecasting models.
Includes early stopping, learning rate scheduling, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import time
from pathlib import Path


class HuberLoss(nn.Module):
    """Huber loss for robust volatility prediction."""
    
    def __init__(self, delta=1.0):
        """
        Initialize Huber loss.
        
        Parameters
        ----------
        delta : float
            Threshold parameter
        """
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, pred, target):
        """Compute Huber loss."""
        error = pred - target
        abs_error = torch.abs(error)
        
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return loss.mean()


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Initialize early stopping.
        
        Parameters
        ----------
        patience : int
            Number of epochs to wait before stopping
        min_delta : float
            Minimum change to qualify as improvement
        mode : str
            'min' or 'max' depending on metric
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        self.multiplier = -1 if mode == 'min' else 1
    
    def __call__(self, score):
        """
        Check if training should stop.
        
        Parameters
        ----------
        score : float
            Current validation score
            
        Returns
        -------
        bool
            True if should stop, False otherwise
        """
        score = self.multiplier * score
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


class LSTMTrainer:
    """Trainer for LSTM volatility forecasting models."""
    
    def __init__(self,
                 model,
                 device='cpu',
                 learning_rate=0.001,
                 loss_fn='huber',
                 weight_decay=1e-5):
        """
        Initialize trainer.
        
        Parameters
        ----------
        model : nn.Module
            LSTM model to train
        device : str or torch.device
            Device to train on
        learning_rate : float
            Initial learning rate
        loss_fn : str
            Loss function: 'mse' or 'huber'
        weight_decay : float
            L2 regularization parameter
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_fn == 'huber':
            self.criterion = HuberLoss(delta=1.0)
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
            
        Returns
        -------
        float
            Average training loss
        """
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self, val_loader):
        """
        Validate model.
        
        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader
            
        Returns
        -------
        float
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def fit(self,
            train_loader,
            val_loader,
            epochs=100,
            early_stopping_patience=15,
            verbose=True):
        """
        Train model with early stopping.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        epochs : int
            Maximum number of epochs
        early_stopping_patience : int
            Patience for early stopping
        verbose : bool
            Whether to print progress
            
        Returns
        -------
        dict
            Training history
        """
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=1e-6,
            mode='min'
        )
        
        best_val_loss = float('inf')
        best_model_state = None
        
        if verbose:
            print(f"\nTraining for up to {epochs} epochs...")
            print(f"Device: {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.6f}")
            
            # Early stopping
            if early_stopping(val_loss):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\nTraining complete!")
            print(f"Time: {elapsed_time:.2f}s")
            print(f"Best validation loss: {best_val_loss:.6f}")
        
        return self.history
    
    def predict(self, data_loader):
        """
        Make predictions.
        
        Parameters
        ----------
        data_loader : DataLoader
            Data loader for predictions
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                pred = self.model(batch_x)
                predictions.append(pred.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def save_model(self, filepath):
        """Save model checkpoint."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Model loaded from: {filepath}")


def train_model(model,
                train_loader,
                val_loader,
                device='cpu',
                epochs=100,
                learning_rate=0.001,
                early_stopping_patience=15,
                loss_fn='huber') -> Tuple[nn.Module, Dict]:
    """
    Convenience function to train model.
    
    Parameters
    ----------
    model : nn.Module
        LSTM model
    train_loader : DataLoader
        Training data
    val_loader : DataLoader
        Validation data
    device : str
        Device to train on
    epochs : int
        Maximum epochs
    learning_rate : float
        Learning rate
    early_stopping_patience : int
        Early stopping patience
    loss_fn : str
        Loss function
        
    Returns
    -------
    model : nn.Module
        Trained model
    history : dict
        Training history
    """
    trainer = LSTMTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        loss_fn=loss_fn
    )
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        verbose=True
    )
    
    return trainer.model, history


if __name__ == '__main__':
    # Test training
    from src.models.lstm import VolatilityLSTM, LSTMDataset, get_device
    
    print("Testing LSTM training...")
    
    # Create dummy data
    n_samples = 1000
    n_features = 10
    seq_length = 60
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.rand(n_samples).astype(np.float32)
    
    # Create datasets
    train_dataset = LSTMDataset(X[:800], y[:800], sequence_length=seq_length)
    val_dataset = LSTMDataset(X[800:], y[800:], sequence_length=seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = VolatilityLSTM(
        input_size=n_features,
        hidden_size=32,
        num_layers=2,
        dropout=0.2
    )
    
    # Train
    device = get_device()
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=20,
        learning_rate=0.001
    )
    
    print("\n✓ Training test successful!")