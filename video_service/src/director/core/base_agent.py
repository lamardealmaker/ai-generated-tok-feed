from typing import Dict, Any

class BaseAgent:
    """Base class for all Director agents."""
    
    def __init__(self):
        """Initialize the base agent."""
        pass
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate agent configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration is valid
        """
        return True
