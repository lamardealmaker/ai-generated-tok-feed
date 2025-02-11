from typing import Dict, Any, Optional
from datetime import datetime
import uuid

class Task:
    """Represents a video generation task."""
    
    def __init__(self, session_id: str):
        """Initialize a new task.
        
        Args:
            session_id: Associated session ID
        """
        self.id = str(uuid.uuid4())
        self.session_id = session_id
        self.status = "processing"
        self.progress = 0.0
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = datetime.utcnow()
        self.updated_at = self.created_at
        
    def update_progress(self, progress: float, status: Optional[str] = None):
        """Update task progress.
        
        Args:
            progress: Progress value between 0 and 1
            status: Optional new status
        """
        self.progress = max(0.0, min(1.0, progress))
        if status:
            self.status = status
        self.updated_at = datetime.utcnow()
        
    def complete(self, result: Dict[str, Any]):
        """Mark task as complete with result.
        
        Args:
            result: Task result data
        """
        self.status = "completed"
        self.progress = 1.0
        self.result = result
        self.updated_at = datetime.utcnow()
        
    def fail(self, error: str):
        """Mark task as failed with error.
        
        Args:
            error: Error message
        """
        self.status = "failed"
        self.error = error
        self.updated_at = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
