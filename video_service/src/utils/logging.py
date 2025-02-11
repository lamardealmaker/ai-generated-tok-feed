import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path("/tmp/video_service/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    return logger
