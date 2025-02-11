from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Base class for all Director tools."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the base tool with optional configuration."""
        self.config = config or {}
        self.validate_config()
        
    def validate_config(self) -> None:
        """Validate tool configuration."""
        pass
        
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get tool capabilities and requirements."""
        pass
        
    def cleanup(self) -> None:
        """Clean up any resources used by the tool."""
        pass
