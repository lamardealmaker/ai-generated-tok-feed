from typing import Dict, Any, Optional
import os
from pathlib import Path
import yaml

class DirectorConfig:
    """Configuration manager for Director integration."""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path('config/director')
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.configs: Dict[str, Any] = {}
        
    def load_config(self, name: str) -> Dict[Any, Any]:
        """Load a specific configuration file."""
        if name in self.configs:
            return self.configs[name]
            
        config_path = self.config_dir / f"{name}.yaml"
        if not config_path.exists():
            return {}
            
        with open(config_path, 'r') as f:
            self.configs[name] = yaml.safe_load(f)
            return self.configs[name]
            
    def save_config(self, name: str, config: Dict[Any, Any]) -> None:
        """Save a configuration to file."""
        config_path = self.config_dir / f"{name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        self.configs[name] = config
        
    def get_value(self, name: str, key: str, default: Any = None) -> Any:
        """Get a specific configuration value."""
        config = self.load_config(name)
        return config.get(key, default)
        
    def set_value(self, name: str, key: str, value: Any) -> None:
        """Set a specific configuration value."""
        config = self.load_config(name)
        config[key] = value
        self.save_config(name, config)
