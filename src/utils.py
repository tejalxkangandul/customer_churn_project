"""
Utility Module
Provides helper functions for the project
"""

import logging
import yaml
from pathlib import Path


def setup_logging(log_level: str = 'INFO', log_file: str = None) -> None:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
        log_file: Optional file path for logging
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[logging.StreamHandler()]
        )


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: dict, config_path: str) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_directories(paths: list) -> None:
    """
    Create directories if they don't exist
    
    Args:
        paths: List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_file_paths(base_path: str, extensions: list = None) -> list:
    """
    Get all file paths with specific extensions from a directory
    
    Args:
        base_path: Base directory path
        extensions: List of file extensions to match
        
    Returns:
        list: List of file paths
    """
    base_path = Path(base_path)
    files = []
    
    for file in base_path.rglob('*'):
        if file.is_file():
            if extensions is None or file.suffix in extensions:
                files.append(str(file))
    
    return files
