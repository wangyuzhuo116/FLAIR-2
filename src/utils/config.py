"""Configuration utilities for FLAIR-2 project."""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class with attribute-style access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        """Allow dictionary-style access."""
        return getattr(self, key)
    
    def __repr__(self):
        """String representation."""
        items = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"Config({', '.join(items)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary.
        
        Returns:
            Configuration as dictionary
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config YAML file
        
    Returns:
        Config object
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Handle base config inheritance
    if '_base_' in config_dict:
        base_path = config_path.parent / config_dict.pop('_base_')
        base_config = load_config(str(base_path))
        base_dict = base_config.to_dict()
        
        # Merge configs (config_dict overrides base)
        config_dict = merge_configs(base_dict, config_dict)
    
    return Config(config_dict)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override config into base config.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: Config, save_path: str):
    """Save configuration to YAML file.
    
    Args:
        config: Config object to save
        save_path: Path to save config YAML
    """
    config_dict = config.to_dict()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
