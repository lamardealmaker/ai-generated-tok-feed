from typing import Dict, Optional, Any
import uuid
from datetime import datetime

class Session:
    """Represents a video generation session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.utcnow()
        self.status = "initialized"
        self.data: Dict[str, Any] = {}
        self.progress = 0
        self.error: Optional[str] = None
        
    def update_progress(self, progress: int, status: Optional[str] = None) -> None:
        """Update session progress."""
        self.progress = progress
        if status:
            self.status = status
            
    def set_error(self, error: str) -> None:
        """Set error state for session."""
        self.status = "error"
        self.error = error
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "data": self.data
        }

class SessionManager:
    """Manages video generation sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        
    def create_session(self) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session = Session(session_id)
        self.sessions[session_id] = session
        return session
        
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self.sessions.get(session_id)
        
    def update_session(self, session_id: str, progress: int, status: Optional[str] = None) -> None:
        """Update session progress."""
        session = self.get_session(session_id)
        if session:
            session.update_progress(progress, status)
            
    def remove_session(self, session_id: str) -> None:
        """Remove a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> None:
        """Remove sessions older than max_age_hours."""
        now = datetime.utcnow()
        to_remove = []
        
        for session_id, session in self.sessions.items():
            age = now - session.created_at
            if age.total_seconds() > max_age_hours * 3600:
                to_remove.append(session_id)
                
        for session_id in to_remove:
            self.remove_session(session_id)
