"""
Helper utilities for the project.
"""

import numpy as np
import pandas as pd
import random
import torch
from pathlib import Path
from typing import Optional


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def create_directory_structure(base_path: str = '.'):
    """
    Create project directory structure.
    
    Parameters
    ----------
    base_path : str
        Base path for project
    """
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'reports/figures',
        'models/checkpoints'
    ]
    
    for directory in directories:
        Path(base_path) / directory
        (Path(base_path) / directory).mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created successfully!")


def save_results(results: dict, filepath: str):
    """
    Save results dictionary to file.
    
    Parameters
    ----------
    results : dict
        Results dictionary
    filepath : str
        Output file path
    """
    import pickle
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Results saved to: {filepath}")


def load_results(filepath: str) -> dict:
    """
    Load results dictionary from file.
    
    Parameters
    ----------
    filepath : str
        Input file path
        
    Returns
    -------
    dict
        Results dictionary
    """
    import pickle
    
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Results loaded from: {filepath}")
    return results


def print_section_header(text: str, width: int = 70):
    """
    Print formatted section header.
    
    Parameters
    ----------
    text : str
        Header text
    width : int
        Total width of header
    """
    print("\n" + "=" * width)
    print(f" {text} ".center(width))
    print("=" * width + "\n")


def print_subsection_header(text: str, width: int = 70):
    """
    Print formatted subsection header.
    
    Parameters
    ----------
    text : str
        Header text
    width : int
        Total width of header
    """
    print("\n" + "-" * width)
    print(f" {text} ")
    print("-" * width)


def format_metrics_table(metrics: dict) -> str:
    """
    Format metrics dictionary as table.
    
    Parameters
    ----------
    metrics : dict
        Metrics dictionary
        
    Returns
    -------
    str
        Formatted table string
    """
    lines = []
    lines.append("=" * 50)
    lines.append(f"{'Metric':<30} {'Value':>15}")
    lines.append("-" * 50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if abs(value) < 1:
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = f"{value:.2f}"
        else:
            formatted_value = str(value)
        
        lines.append(f"{key:<30} {formatted_value:>15}")
    
    lines.append("=" * 50)
    
    return "\n".join(lines)


def log_experiment(message: str, log_file: str = 'experiment.log'):
    """
    Log message to file with timestamp.
    
    Parameters
    ----------
    message : str
        Message to log
    log_file : str
        Log file path
    """
    from datetime import datetime
    
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_message = f"[{timestamp}] {message}\n"
    
    with open(log_file, 'a') as f:
        f.write(log_message)


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        """
        Initialize timer.
        
        Parameters
        ----------
        name : str
            Name of operation being timed
        """
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timer."""
        import time
        self.start_time = time.time()
        print(f"\n{self.name} started...")
        return self
    
    def __exit__(self, *args):
        """Stop timer and print elapsed time."""
        import time
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        
        if elapsed < 60:
            print(f"{self.name} completed in {elapsed:.2f} seconds")
        elif elapsed < 3600:
            print(f"{self.name} completed in {elapsed/60:.2f} minutes")
        else:
            print(f"{self.name} completed in {elapsed/3600:.2f} hours")


if __name__ == '__main__':
    # Test helpers
    print("Testing helper utilities...")
    
    # Test random seed
    set_random_seed(42)
    
    # Test directory creation
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    create_directory_structure(temp_dir)
    
    # Test timer
    with Timer("Test operation"):
        import time
        time.sleep(1)
    
    # Clean up
    shutil.rmtree(temp_dir)
    
    # Test formatting
    metrics = {
        'RMSE': 0.1234,
        'MAE': 0.0987,
        'Sharpe': 1.56
    }
    print("\nFormatted metrics:")
    print(format_metrics_table(metrics))
    
    print("\n✓ Helper utilities test successful!")