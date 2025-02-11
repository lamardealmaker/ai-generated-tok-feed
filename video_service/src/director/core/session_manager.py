from typing import Dict, Any, Optional
import uuid
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager
from src.utils.logging import setup_logger

class SessionManager:
    """Manages video generation sessions and state tracking."""
    
    def __init__(self):
        self.logger = setup_logger("session_manager")
        self.logger.info("Initializing SessionManager...")
        
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.cleanup_interval = timedelta(hours=1)
        self._cleanup_task = None
        
        self.logger.info("SessionManager initialized successfully")
        
    async def start(self):
        """Start the session manager and cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop the session manager and cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            
    @asynccontextmanager
    async def session(self, session_id: Optional[str] = None) -> str:
        """
        Context manager for video generation sessions.
        
        Args:
            session_id: Optional session ID to use
            
        Returns:
            Session ID string
        """
        session_id = session_id or str(uuid.uuid4())
        try:
            await self.create_session(session_id)
            yield session_id
        finally:
            await self.end_session(session_id)
            
    async def create_session(self, session_id: str) -> None:
        """Create a new video generation session."""
        self.logger.info(f"Creating new session with ID: {session_id}")
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")
            
        self.sessions[session_id] = {
            "created_at": datetime.utcnow(),
            "status": "initializing",
            "progress": 0.0,
            "message": "Session created",
            "artifacts": {},
            "errors": []
        }
        
    async def end_session(self, session_id: str) -> None:
        """End a video generation session."""
        if session_id in self.sessions:
            self.sessions[session_id]["status"] = "completed"
            self.sessions[session_id]["ended_at"] = datetime.utcnow()
            
    async def update_progress(
        self,
        session_id: str,
        progress: float,
        message: Optional[str] = None
    ) -> None:
        """Update session progress."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        self.sessions[session_id].update({
            "progress": progress,
            "message": message or self.sessions[session_id]["message"],
            "updated_at": datetime.utcnow()
        })
        
    async def add_artifact(
        self,
        session_id: str,
        artifact_type: str,
        artifact_data: Any
    ) -> None:
        """Add an artifact to the session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        self.sessions[session_id]["artifacts"][artifact_type] = {
            "data": artifact_data,
            "created_at": datetime.utcnow()
        }
        
    async def add_error(
        self,
        session_id: str,
        error: Exception
    ) -> None:
        """Add an error to the session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        self.sessions[session_id]["errors"].append({
            "error": str(error),
            "type": type(error).__name__,
            "timestamp": datetime.utcnow()
        })
        
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session information."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        return self.sessions[session_id]
        
    async def _cleanup_loop(self):
        """Background task to clean up old sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval.total_seconds())
                await self._cleanup_old_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup failed: {str(e)}")
                
    async def _cleanup_old_sessions(self):
        """Clean up completed sessions older than cleanup_interval."""
        now = datetime.utcnow()
        to_remove = []
        
        for session_id, session in self.sessions.items():
            if (
                session["status"] == "completed"
                and now - session["ended_at"] > self.cleanup_interval
            ):
                to_remove.append(session_id)
                
        for session_id in to_remove:
            del self.sessions[session_id]
