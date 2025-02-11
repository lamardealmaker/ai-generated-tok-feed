# Video Service Core Codebase

## Statistics (Updated 2025-02-10)
- Total Core Python Files: 18
- Total Lines of Core Code: 4,756
- Focus: Real Estate Video Generation
- Last Updated: 2025-02-10 21:52:08

## Core Components

### Main Entry Points
1. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/generate_property_video.py
   - Main script for property video generation
   - Handles property data input
   - Coordinates video generation pipeline

2. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/main.py
   - Application entry point
   - API server initialization
   - Configuration loading

### Core Engine
3. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/core/reasoning_engine.py
   - Central orchestrator
   - Manages video generation workflow
   - Coordinates between agents
   - Progress tracking (0-100%)

4. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/core/engine.py
   - Base engine functionality
   - Component initialization
   - Error handling

5. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/core/session.py
   - Session management
   - State tracking
   - Resource cleanup

6. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/core/task.py
   - Task tracking
   - Progress monitoring
   - Status updates

### Key Agents
7. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/agents/script_agent.py
   - Script generation
   - Property feature highlighting
   - Dynamic content creation

8. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/agents/effect_agent.py
   - Visual effects management
   - Effect selection
   - Style application

9. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/agents/pacing_agent.py
   - Video timing optimization
   - Scene duration control
   - Transition timing

10. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/agents/music_agent.py
    - Music selection
    - Audio mixing
    - Mood matching

11. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/agents/style_agent.py
    - Style management
    - Theme consistency
    - Brand alignment

### Essential Tools
12. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/tools/video_renderer.py
    - Video rendering
    - Frame composition
    - Output generation

13. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/tools/video_db_tool.py
    - Video processing
    - Effect application
    - Frame manipulation

14. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/tools/music_tool.py
    - Music processing
    - Audio alignment
    - Volume control

15. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/tools/voiceover_tool.py
    - Voice generation
    - Audio synchronization
    - Speech processing

### Effects
16. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/effects/text_overlay.py
    - Text effects
    - Caption generation
    - Font styling

17. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/effects/visual_effects.py
    - Visual effects
    - Color grading
    - Filter application

18. /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/effects/transitions.py
    - Transition effects
    - Scene transitions
    - Smooth animations

## Recent Updates
1. Video Generation Pipeline
   - Improved script generation
   - Enhanced effect application
   - Optimized rendering process

2. Audio System
   - Better voiceover integration
   - Improved music selection
   - Enhanced audio mixing

3. Visual Effects
   - New transition effects
   - Improved text overlays
   - Enhanced visual filters

## File Contents
### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/generate_property_video.py
```python
import asyncio
import os
from dotenv import load_dotenv
from src.director.core.reasoning_engine import ReasoningEngine
from src.utils.logging import setup_logger

# Load environment variables
load_dotenv()

# Set up logger
logger = setup_logger("property_video_gen")

async def generate_property_video(property_id: str, style: str = "luxury"):
    try:
        # Initialize the engine
        engine = ReasoningEngine()
        
        # Real property data
        listing_data = {
            "id": property_id,
            "title": "Modern Waterfront Paradise",
            "price": 2850000,
            "description": """
            Discover luxury living at its finest in this stunning waterfront estate. 
            This architectural masterpiece seamlessly blends indoor and outdoor living, 
            offering breathtaking water views from nearly every room. The home features 
            soaring ceilings, walls of glass, and premium finishes throughout.
            """,
            "features": [
                "Panoramic Water Views",
                "Chef's Kitchen with Premium Appliances",
                "Primary Suite with Private Terrace",
                "Indoor-Outdoor Entertainment Spaces",
                "Private Dock",
                "Wine Cellar",
                "Smart Home Technology",
                "3-Car Garage"
            ],
            "location": "Miami Beach",
            "bedrooms": 6,
            "bathrooms": 7.5,
            "sqft": 6800,
            "year_built": 2023,
            "lot_size": 15000
        }
        
        # Property images (using test images for now)
        image_urls = [
            "gs://project3-ziltok.firebasestorage.app/test_properties/house1.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/house2.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/house3.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/interior1.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/interior2.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/backyard.jpg"
        ]
        
        # Progress callback
        def progress_callback(progress: float, message: str):
            logger.info(f"Progress: {progress*100:.1f}% - {message}")
        
        logger.info(f"Generating video for property {property_id}...")
        
        # Generate video
        await engine.generate_video(
            listing_data=listing_data,
            image_urls=image_urls,
            style=style,
            duration=30.0,  # 30 second video
            progress_callback=progress_callback
        )
        
        logger.info("Video generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Video generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # You can change the style to "modern" or "minimal" for different looks
        asyncio.run(generate_property_video("waterfront-estate-1", style="luxury"))
    except KeyboardInterrupt:
        logger.info("Video generation interrupted by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/main.py
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Real Estate TikTok Video Generator",
    description="Director-based service for generating TikTok-style real estate videos",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from api.routes import app as api_app

# Include API routes
app.include_router(api_app.router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/core/reasoning_engine.py
```python
from typing import Dict, Any, List, Optional, Callable
import asyncio
from datetime import datetime
import uuid
import os
from src.utils.logging import setup_logger
from ..agents.script_agent import ScriptAgent
from ..agents.pacing_agent import PacingAgent
from ..agents.effect_agent import EffectAgent
from ..tools.video_db_tool import VideoDBTool
from ..tools.video_renderer import VideoRenderer
from ..tools.music_tool import MusicTool
from ..tools.voiceover_tool import VoiceoverTool
from .task import Task

class ReasoningEngine:
    """Core reasoning and workflow orchestration engine."""
    
    def __init__(self):
        self.logger = setup_logger("reasoning_engine")
        self.logger.info("Initializing ReasoningEngine...")
        
        self.script_agent = ScriptAgent()
        self.pacing_agent = PacingAgent()
        self.effect_agent = EffectAgent()
        self.video_db_tool = VideoDBTool()
        self.video_renderer = VideoRenderer()
        self.music_tool = MusicTool()
        self.voiceover_tool = VoiceoverTool()
        
        # Create audio directory if it doesn't exist
        self.audio_dir = os.path.join(os.getcwd(), "assets/audio")
        os.makedirs(os.path.join(self.audio_dir, "voiceover"), exist_ok=True)
        os.makedirs(os.path.join(self.audio_dir, "music"), exist_ok=True)
        
        self.logger.info("All components initialized successfully")
        
        self.workflow_hooks: Dict[str, List[Callable]] = {
            "pre_generation": [],
            "post_script": [],
            "post_effects": [],
            "post_generation": []
        }
        
    async def generate_video(
        self,
        listing_data: Dict[str, Any],
        image_urls: List[str],
        style: str = "modern",
        duration: float = 30.0,
        session_id: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Task:
        """
        Generate a complete real estate video.
        
        Args:
            property_data: Property listing data
            style: Video style to use
            duration: Target video duration
            session_id: Optional session ID for tracking
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing video metadata and output paths
        """
        try:
            # Run pre-generation hooks
            await self._run_hooks("pre_generation", listing_data)
            
            # 1. Generate script
            self.logger.info(f"Generating script for property {listing_data.get('id', 'unknown')}")
            if progress_callback:
                progress_callback(0.1, "Generating script")
                
            script = self.script_agent.generate_script(
                property_data=listing_data,
                style=style,
                duration=duration
            )
            
            # Run post-script hooks
            await self._run_hooks("post_script", script)
            
            # 2. Analyze and optimize pacing
            if progress_callback:
                progress_callback(0.2, "Optimizing pacing")
                
            content_analysis = self.pacing_agent.analyze_content(script["segments"])
            pacing = self.pacing_agent.get_optimal_pacing(
                content_analysis=content_analysis,
                target_duration=duration,
                style=style
            )
            
            segments = self.pacing_agent.apply_pacing(
                segments=script["segments"],
                pacing=pacing,
                target_duration=duration
            )
            
            # 3. Generate effects and transitions
            if progress_callback:
                progress_callback(0.3, "Generating effects")
                
            # Get base effects for the style
            base_effects = self.video_db_tool.get_effects(
                template_name=f"{style}_standard",
                intensity=0.8  # Slightly reduced intensity for better results
            )
            
            # Get transitions (one less than segments since we don't need transition for first)
            transitions = self.video_db_tool.get_transitions(
                template_name=f"{style}_standard",
                count=len(segments) - 1,
                variation=0.2
            )
            
            # Process each segment with effects and transitions
            processed_segments = []
            for i, segment in enumerate(segments):
                # Don't add transition for first segment
                transition = None if i == 0 else transitions[i-1]
                
                # Process segment
                processed = self.video_db_tool.process_segment(
                    segment=segment,
                    effects=base_effects,
                    transition=transition
                )
                processed_segments.append(processed)
            
            # Replace segments with processed ones
            segments = processed_segments
                
            # Run post-effects hooks
            await self._run_hooks("post_effects", segments)
            
            # 4. Generate voiceovers for each segment
            if progress_callback:
                progress_callback(0.4, "Generating voiceover")
                
            for segment in segments:
                if "text" in segment:
                    voiceover = self.voiceover_tool.generate_voiceover(
                        text=segment["text"],
                        style=style,
                        output_format="mp3"
                    )
                    segment["voiceover_path"] = voiceover["file_path"]
            
            # 5. Select music
            if progress_callback:
                progress_callback(0.5, "Selecting music")
                
            music = self.music_tool.select_music(
                template_name=f"{style}_balanced_standard",
                duration=duration
            )
            
            # 5. Process video
            if progress_callback:
                progress_callback(0.5, "Processing video")
                
            # 6. Finalize video
            if progress_callback:
                progress_callback(0.9, "Finalizing video")
                
            result = {
                "session_id": session_id,
                "style": style,
                "duration": duration,
                "segments": segments,  # Use processed segments
                "music": music,
                "generated_at": datetime.utcnow().isoformat(),
                "property_id": listing_data.get("id")
            }
            
            # Add image URLs to segments
            for i, segment in enumerate(segments):  # Use processed segments
                # Cycle through images if we have more segments than images
                image_index = i % len(image_urls)
                segment["image_url"] = image_urls[image_index]
            
            # Render final video
            if progress_callback:
                progress_callback(0.9, "Rendering video")
                
            output_path = self.video_renderer.render_video(
                segments=segments,  # Use processed segments
                music=music
            )
            
            result["output_path"] = output_path
            
            # Run post-generation hooks
            await self._run_hooks("post_generation", result)
            
            if progress_callback:
                progress_callback(1.0, "Complete")
                
            # Create and complete task
            task = Task(session_id or str(uuid.uuid4()))
            task.complete(result)
            return task
            
        except Exception as e:
            self.logger.error(f"Video generation failed: {str(e)}")
            # Create failed task
            task = Task(session_id or str(uuid.uuid4()))
            task.fail(str(e))
            return task
            
    def register_hook(
        self,
        hook_type: str,
        callback: Callable
    ) -> None:
        """Register a workflow hook."""
        if hook_type not in self.workflow_hooks:
            raise ValueError(f"Invalid hook type: {hook_type}")
        self.workflow_hooks[hook_type].append(callback)
        
    async def _run_hooks(
        self,
        hook_type: str,
        data: Any
    ) -> None:
        """Run all registered hooks of a given type."""
        for hook in self.workflow_hooks[hook_type]:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(data)
                else:
                    hook(data)
            except Exception as e:
                logger.error(f"Hook execution failed: {str(e)}")
                # Continue with other hooks
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/core/engine.py
```python
from typing import Dict, Any, Optional
import os
from pathlib import Path
import yaml

class ReasoningEngine:
    """
    Wrapper for Director's reasoning engine, customized for real estate video generation.
    """
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.session_manager = None  # Will be initialized with SessionManager
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[Any, Any]:
        """Load engine configuration from YAML file."""
        if not config_path:
            config_path = os.getenv('DIRECTOR_CONFIG_PATH', 'config/director/engine.yaml')
        
        config_path = Path(config_path)
        if not config_path.exists():
            return self._create_default_config(config_path)
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self, config_path: Path) -> Dict[Any, Any]:
        """Create default configuration if none exists."""
        default_config = {
            'llm': {
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 2000
            },
            'video': {
                'max_duration': 60,  # seconds
                'resolution': '1080p',
                'fps': 30
            },
            'agents': {
                'script_generator': {
                    'enabled': True,
                    'style': 'tiktok'
                },
                'voice_generator': {
                    'enabled': True,
                    'provider': 'elevenlabs'
                },
                'music_selector': {
                    'enabled': True,
                    'style': 'upbeat'
                }
            }
        }
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save default config
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
            
        return default_config
    
    def initialize(self, session_manager: 'SessionManager') -> None:
        """Initialize the engine with a session manager."""
        self.session_manager = session_manager
        
    async def process_request(self, session_id: str, request_data: Dict[Any, Any]) -> Dict[Any, Any]:
        """
        Process a video generation request.
        
        Args:
            session_id: Unique session identifier
            request_data: Video generation parameters
            
        Returns:
            Dict containing processing results
        """
        if not self.session_manager:
            raise RuntimeError("Engine not initialized with SessionManager")
            
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"No session found for ID: {session_id}")
            
        # TODO: Implement actual processing logic using Director's components
        return {
            "status": "processing",
            "session_id": session_id,
            "message": "Video generation started"
        }
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/core/session.py
```python
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
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/core/task.py
```python
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
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/agents/script_agent.py
```python
from typing import Dict, Any, List, Optional
import json
from ..core.base_agent import BaseAgent

class ScriptAgent(BaseAgent):
    """Agent for generating dynamic video scripts using LLMs."""
    
    def __init__(self):
        self.script_templates = {
            "modern": {
                "hooks": [
                    "WAIT UNTIL YOU SEE THIS {location} DREAM HOME! 🏠✨",
                    "THIS {style} HOME IN {location} WILL BLOW YOUR MIND! 🔥",
                    "INSANE {price_range} PROPERTY ALERT! 🚨 {location} GEM! 💎"
                ],
                "features": [
                    "OBSESSED with this {feature}! 😍",
                    "THE MOST INCREDIBLE {feature} for {use_case}! 🤯",
                    "YOU WON'T BELIEVE this {feature}! ✨"
                ],
                "closings": [
                    "RUN don't walk! DM now! 🏃‍♂️",
                    "This won't last! Save this! ⭐️",
                    "Tag someone who needs to see this! 🔥"
                ]
            },
            "luxury": {
                "hooks": [
                    "EXCLUSIVE FIRST LOOK: ${price_range} {location} MANSION 👑",
                    "INSIDE THE MOST STUNNING {location} ESTATE YET! ✨",
                    "THIS {style} MASTERPIECE WILL LEAVE YOU SPEECHLESS! 🏰"
                ],
                "features": [
                    "LOOK AT THIS INCREDIBLE {feature}! Perfect for {use_case} 🤩",
                    "THE MOST INSANE {feature} you'll ever see! 🔥",
                    "I CAN'T BELIEVE this {feature}! Absolutely unreal! ✨"
                ],
                "closings": [
                    "Serious inquiries only - DM for private tour 🔑",
                    "Save this dream home! You won't regret it! ⭐️",
                    "Like & Share if this is your dream home! 🏠"
                ]
            },
            "minimal": {
                "hooks": [
                    "THIS {location} GEM IS EVERYTHING! ✨",
                    "PERFECTLY DESIGNED {style} HOME! 🏠",
                    "THE MOST AESTHETIC {location} FIND! 🎯"
                ],
                "features": [
                    "OBSESSED with this {feature} for {use_case}! 😍",
                    "Clean lines define this {feature}",
                    "Smart design in the {feature}"
                ],
                "closings": [
                    "Schedule your visit",
                    "Available for viewing",
                    "Connect for details"
                ]
            }
        }
        
    def _analyze_property(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze property data to determine key selling points."""
        analysis = {
            "price_range": self._get_price_range(property_data["price"]),
            "key_features": [],
            "unique_selling_points": [],
            "target_audience": [],
            "style_keywords": []
        }
        
        # Analyze price and location
        if property_data["price"] > 1000000:
            analysis["target_audience"].extend(["luxury-buyers", "investors"])
            analysis["style_keywords"].extend(["luxury", "premium", "exclusive"])
        else:
            analysis["target_audience"].extend(["homeowners", "families"])
            analysis["style_keywords"].extend(["modern", "comfortable", "practical"])
            
        # Analyze features
        # Check bedrooms
        if property_data.get("bedrooms", 0) >= 4:
            analysis["key_features"].append(("spacious layout", "family living"))
            analysis["unique_selling_points"].append("Large family home")
            
        # Check bathrooms
        if property_data.get("bathrooms", 0) >= 2.5:
            analysis["key_features"].append(("multiple bathrooms", "convenience"))
            
        # Check square footage
        sqft = property_data.get("sqft", 0)
        if sqft > 3000:
            analysis["key_features"].append(("expansive space", "luxury living"))
        elif sqft > 2000:
            analysis["key_features"].append(("generous space", "comfortable living"))
            
        # Check lot size
        if property_data.get("lot_size", 0) > 10000:  # 10,000 sqft
            analysis["key_features"].append(("large lot", "outdoor living"))
            
        # Process custom features
        for feature in property_data.get("features", []):
            feature_lower = feature.lower()
            if "kitchen" in feature_lower:
                analysis["key_features"].append((feature, "modern living"))
            elif "floor" in feature_lower:
                analysis["key_features"].append((feature, "design"))
            elif "yard" in feature_lower or "outdoor" in feature_lower:
                analysis["key_features"].append((feature, "outdoor living"))
            else:
                analysis["key_features"].append((feature, "lifestyle"))
            
        return analysis
        
    def _get_price_range(self, price: float) -> str:
        """Convert price to descriptive range."""
        if price < 300000:
            return "starter"
        elif price < 600000:
            return "mid-range"
        elif price < 1000000:
            return "upscale"
        else:
            return "luxury"
            
    MIN_DURATION = 10.0  # TikTok minimum video duration
    MAX_DURATION = 60.0  # TikTok maximum video duration
    
    def generate_script(
        self,
        property_data: Dict[str, Any],
        style: str = "modern",
        duration: float = 30.0,
        target_segments: int = 5
    ) -> Dict[str, Any]:
        """
        Generate a dynamic video script based on property data.
        
        Args:
            property_data: Dictionary containing property information
            style: Script style to use
            duration: Target video duration
            target_segments: Number of segments to generate
            
        Returns:
            Dictionary containing script segments and metadata
        """
        # Validate duration
        if duration < self.MIN_DURATION:
            duration = self.MIN_DURATION
        elif duration > self.MAX_DURATION:
            duration = self.MAX_DURATION
            
        # Adjust segments based on duration
        if duration <= 15.0:
            target_segments = min(target_segments, 3)  # Shorter videos need fewer segments
        elif duration >= 45.0:
            target_segments = 7  # Longer videos need more segments
            # Add extra segments for very long videos
            if duration >= 55.0:
                target_segments = 8
            
        analysis = self._analyze_property(property_data)
        templates = self.script_templates[style]
        
        # Calculate timing
        segment_duration = duration / target_segments
        current_time = 0.0
        
        script = {
            "style": style,
            "duration": duration,
            "segments": [],
            "metadata": {
                "price_range": analysis["price_range"],
                "target_audience": analysis["target_audience"],
                "key_features": analysis["key_features"]
            }
        }
        
        # Generate hook
        hook_text = templates["hooks"][0].format(
            location=property_data["location"],
            style=property_data.get("style", "modern"),
            price_range=analysis["price_range"]
        )
        script["segments"].append({
            "type": "hook",
            "text": hook_text,
            "start_time": current_time,
            "duration": segment_duration,
            "focus": "exterior"
        })
        current_time += segment_duration
        
        # Generate feature segments
        # Calculate how many feature segments we need
        feature_count = target_segments - 2  # -2 for hook and closing
        
        # If we need more features than available, duplicate some
        features = analysis["key_features"]
        while len(features) < feature_count:
            features.extend(analysis["key_features"])
        features = features[:feature_count]
        
        # Generate feature segments
        for feature, use_case in features:
            feature_text = templates["features"][0].format(
                feature=feature,
                use_case=use_case
            )
            script["segments"].append({
                "type": "feature",
                "text": feature_text,
                "start_time": current_time,
                "duration": segment_duration,
                "focus": feature.replace(" ", "_")
            })
            current_time += segment_duration
            
        # Generate closing
        closing_text = templates["closings"][0]
        script["segments"].append({
            "type": "closing",
            "text": closing_text,
            "start_time": current_time,
            "duration": segment_duration,
            "focus": "exterior"
        })
        
        return script
        
    def adjust_script_pacing(
        self,
        script: Dict[str, Any],
        engagement_data: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Adjust script pacing based on engagement data.
        
        Args:
            script: Original script
            engagement_data: Optional dictionary with engagement metrics
            
        Returns:
            Adjusted script
        """
        if not engagement_data:
            return script
            
        adjusted = script.copy()
        segments = adjusted["segments"]
        
        # Calculate engagement-based duration adjustments
        total_engagement = sum(engagement_data.values())
        for segment in segments:
            segment_type = segment["type"]
            if segment_type in engagement_data:
                engagement_ratio = engagement_data[segment_type] / total_engagement
                # Adjust duration based on engagement (±20% max)
                base_duration = segment["duration"]
                adjustment = (engagement_ratio - 1/len(engagement_data)) * 0.2
                segment["duration"] = base_duration * (1 + adjustment)
                
        # Normalize durations to match target duration
        total_duration = sum(s["duration"] for s in segments)
        scale_factor = script["duration"] / total_duration
        for segment in segments:
            segment["duration"] *= scale_factor
            
        # Update start times
        current_time = 0.0
        for segment in segments:
            segment["start_time"] = current_time
            current_time += segment["duration"]
            
        return adjusted
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/agents/effect_agent.py
```python
from typing import Dict, Any, List, Optional
import random
from ..core.base_agent import BaseAgent

class EffectAgent(BaseAgent):
    """Agent for generating and combining video effects."""
    
    def __init__(self):
        self.effect_library = {
            "color": {
                "modern": [
                    {"name": "vibrant", "contrast": 1.2, "saturation": 1.3, "brightness": 1.1},
                    {"name": "cool", "contrast": 1.1, "saturation": 0.9, "temperature": -10},
                    {"name": "warm", "contrast": 1.1, "saturation": 1.1, "temperature": 10}
                ],
                "luxury": [
                    {"name": "cinematic", "contrast": 1.3, "saturation": 0.9, "shadows": 0.8},
                    {"name": "rich", "contrast": 1.2, "saturation": 1.1, "highlights": 0.9},
                    {"name": "moody", "contrast": 1.4, "saturation": 0.8, "shadows": 0.7}
                ],
                "minimal": [
                    {"name": "clean", "contrast": 1.1, "saturation": 0.9, "brightness": 1.05},
                    {"name": "bright", "contrast": 1.0, "saturation": 0.8, "brightness": 1.2},
                    {"name": "pure", "contrast": 1.05, "saturation": 0.85, "highlights": 1.1}
                ]
            },
            "overlay": {
                "modern": [
                    {"name": "light_leak", "opacity": 0.2, "blend": "screen"},
                    {"name": "grain", "opacity": 0.1, "blend": "overlay"},
                    {"name": "vignette", "opacity": 0.15, "blend": "multiply"}
                ],
                "luxury": [
                    {"name": "film_grain", "opacity": 0.08, "blend": "overlay"},
                    {"name": "soft_light", "opacity": 0.15, "blend": "soft_light"},
                    {"name": "cinematic_bars", "opacity": 1.0, "blend": "normal"}
                ],
                "minimal": [
                    {"name": "subtle_grain", "opacity": 0.05, "blend": "overlay"},
                    {"name": "soft_vignette", "opacity": 0.1, "blend": "multiply"},
                    {"name": "clean_overlay", "opacity": 0.08, "blend": "screen"}
                ]
            },
            "filter": {
                "modern": [
                    {"name": "sharpen", "amount": 0.3, "radius": 1.0},
                    {"name": "clarity", "amount": 0.4, "radius": 1.5},
                    {"name": "dehaze", "amount": 0.2}
                ],
                "luxury": [
                    {"name": "bloom", "amount": 0.3, "radius": 2.0},
                    {"name": "glow", "amount": 0.25, "radius": 1.8},
                    {"name": "soft_focus", "amount": 0.2, "radius": 1.5}
                ],
                "minimal": [
                    {"name": "clarity", "amount": 0.2, "radius": 1.0},
                    {"name": "sharpen", "amount": 0.15, "radius": 0.8},
                    {"name": "clean", "amount": 0.1, "radius": 1.0}
                ]
            }
        }
        
        self.combination_rules = {
            "modern": {
                "max_overlays": 2,
                "required_effects": ["color", "filter"],
                "optional_effects": ["overlay"],
                "compatibility": {
                    "light_leak": ["grain"],
                    "vignette": ["grain", "light_leak"]
                }
            },
            "luxury": {
                "max_overlays": 2,
                "required_effects": ["color", "filter", "overlay"],
                "optional_effects": [],
                "compatibility": {
                    "film_grain": ["soft_light"],
                    "cinematic_bars": ["film_grain", "soft_light"]
                }
            },
            "minimal": {
                "max_overlays": 1,
                "required_effects": ["color"],
                "optional_effects": ["filter", "overlay"],
                "compatibility": {
                    "subtle_grain": ["soft_vignette"],
                    "clean_overlay": ["subtle_grain"]
                }
            }
        }
        
    def generate_effect_combination(
        self,
        style: str,
        content_type: str,
        intensity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate a compatible combination of effects.
        
        Args:
            style: Video style
            content_type: Type of content (exterior, interior, etc.)
            intensity: Effect intensity multiplier
            
        Returns:
            Dictionary of combined effects
        """
        rules = self.combination_rules[style]
        effects = {
            "color": None,
            "filter": None,
            "overlay": []
        }
        
        # Apply required effects
        for effect_type in rules["required_effects"]:
            if effect_type == "overlay":
                continue  # Handled separately
            effects[effect_type] = self._select_effect(
                effect_type, style, intensity
            )
            
        # Apply optional effects
        for effect_type in rules["optional_effects"]:
            if random.random() < 0.7:  # 70% chance to apply optional effect
                if effect_type == "overlay":
                    continue  # Handled separately
                effects[effect_type] = self._select_effect(
                    effect_type, style, intensity
                )
                
        # Handle overlays with compatibility rules
        available_overlays = self.effect_library["overlay"][style]
        selected_overlays = []
        
        while (len(selected_overlays) < rules["max_overlays"] and 
               len(available_overlays) > 0):
            overlay = random.choice(available_overlays)
            available_overlays.remove(overlay)
            
            # Check compatibility
            compatible = True
            for selected in selected_overlays:
                if (selected["name"] in rules["compatibility"] and
                    overlay["name"] not in rules["compatibility"][selected["name"]]):
                    compatible = False
                    break
                    
            if compatible:
                # Adjust opacity based on intensity
                adjusted = overlay.copy()
                adjusted["opacity"] *= intensity
                selected_overlays.append(adjusted)
                
        effects["overlay"] = selected_overlays
        
        # Add content-specific adjustments
        self._adjust_for_content(effects, content_type)
        
        return effects
        
    def _select_effect(
        self,
        effect_type: str,
        style: str,
        intensity: float
    ) -> Dict[str, Any]:
        """Select and adjust an effect based on style and intensity."""
        options = self.effect_library[effect_type][style]
        effect = random.choice(options).copy()
        
        # Adjust effect parameters based on intensity
        if effect_type == "color":
            for key in ["contrast", "saturation", "brightness"]:
                if key in effect:
                    deviation = abs(1 - effect[key])
                    effect[key] = 1 + (deviation * intensity * 
                                     (1 if effect[key] > 1 else -1))
        elif effect_type == "filter":
            effect["amount"] *= intensity
            
        return effect
        
    def _adjust_for_content(
        self,
        effects: Dict[str, Any],
        content_type: str
    ) -> None:
        """Adjust effects based on content type."""
        if content_type == "exterior":
            # Enhance contrast and saturation for exterior shots
            if effects["color"]:
                effects["color"]["contrast"] *= 1.1
                effects["color"]["saturation"] *= 1.1
        elif content_type == "interior":
            # Boost brightness for interior shots
            if effects["color"]:
                effects["color"]["brightness"] = max(
                    1.1,
                    effects["color"].get("brightness", 1.0)
                )
        elif content_type == "feature":
            # Enhance clarity for feature shots
            if effects["filter"]:
                effects["filter"]["amount"] *= 1.2
                
    def blend_effects(
        self,
        effects_a: Dict[str, Any],
        effects_b: Dict[str, Any],
        blend_factor: float
    ) -> Dict[str, Any]:
        """
        Blend two effect combinations.
        
        Args:
            effects_a: First effect combination
            effects_b: Second effect combination
            blend_factor: Blend factor (0-1, 0=full A, 1=full B)
            
        Returns:
            Blended effect combination
        """
        blended = {
            "color": {},
            "filter": {},
            "overlay": []
        }
        
        # Blend color effects
        if effects_a.get("color") and effects_b.get("color"):
            for key in ["contrast", "saturation", "brightness", "temperature"]:
                if key in effects_a["color"] and key in effects_b["color"]:
                    blended["color"][key] = (
                        effects_a["color"][key] * (1 - blend_factor) +
                        effects_b["color"][key] * blend_factor
                    )
                    
        # Blend filter effects
        if effects_a.get("filter") and effects_b.get("filter"):
            for key in ["amount", "radius"]:
                if key in effects_a["filter"] and key in effects_b["filter"]:
                    blended["filter"][key] = (
                        effects_a["filter"][key] * (1 - blend_factor) +
                        effects_b["filter"][key] * blend_factor
                    )
                    
        # Select overlays based on blend factor
        all_overlays = (effects_a.get("overlay", []) + 
                       effects_b.get("overlay", []))
        if all_overlays:
            # Prioritize overlays from the dominant effect set
            dominant = effects_a if blend_factor < 0.5 else effects_b
            secondary = effects_b if blend_factor < 0.5 else effects_a
            
            blended["overlay"] = (
                dominant.get("overlay", [])[:1] +
                secondary.get("overlay", [])[:1]
            )
            
        return blended
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/agents/pacing_agent.py
```python
from typing import Dict, Any, List, Optional
import numpy as np
from ..core.base_agent import BaseAgent

class PacingAgent(BaseAgent):
    """Agent for adapting video pacing based on content and engagement."""
    
    def __init__(self):
        self.pacing_profiles = {
            "rapid": {
                "segment_duration": 2.5,
                "transition_duration": 0.3,
                "text_duration": 2.0,
                "music_bpm_range": (120, 140)
            },
            "balanced": {
                "segment_duration": 3.5,
                "transition_duration": 0.5,
                "text_duration": 3.0,
                "music_bpm_range": (100, 120)
            },
            "relaxed": {
                "segment_duration": 4.5,
                "transition_duration": 0.7,
                "text_duration": 4.0,
                "music_bpm_range": (80, 100)
            }
        }
        
        self.content_weights = {
            "exterior": 1.0,
            "interior": 1.2,
            "feature": 1.5,
            "amenity": 1.3,
            "view": 1.4
        }
        
    def analyze_content(
        self,
        segments: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Analyze content to determine optimal pacing.
        
        Args:
            segments: List of video segments with content type
            
        Returns:
            Dictionary with content analysis metrics
        """
        analysis = {
            "complexity": 0.0,
            "visual_interest": 0.0,
            "information_density": 0.0
        }
        
        for segment in segments:
            content_type = segment.get("content_type", "interior")
            weight = self.content_weights.get(content_type, 1.0)
            
            # Calculate complexity based on content
            if "text" in segment:
                analysis["complexity"] += len(segment["text"].split()) * 0.1 * weight
            
            # Calculate visual interest
            if "features" in segment:
                analysis["visual_interest"] += len(segment["features"]) * 0.2 * weight
                
            # Calculate information density
            if "data_points" in segment:
                analysis["information_density"] += len(segment["data_points"]) * 0.15 * weight
                
        # Normalize metrics
        total_segments = len(segments)
        for key in analysis:
            analysis[key] /= total_segments
            
        return analysis
        
    def get_optimal_pacing(
        self,
        content_analysis: Dict[str, float],
        target_duration: float,
        style: str = "modern"
    ) -> Dict[str, Any]:
        """
        Determine optimal pacing based on content analysis.
        
        Args:
            content_analysis: Content analysis metrics
            target_duration: Target video duration
            style: Video style
            
        Returns:
            Pacing configuration
        """
        # Calculate base pacing profile
        complexity_score = content_analysis["complexity"]
        if complexity_score > 0.7:
            base_profile = "relaxed"
        elif complexity_score < 0.3:
            base_profile = "rapid"
        else:
            base_profile = "balanced"
            
        profile = self.pacing_profiles[base_profile].copy()
        
        # Adjust for style
        if style == "luxury":
            profile["segment_duration"] *= 1.2
            profile["transition_duration"] *= 1.3
        elif style == "minimal":
            profile["segment_duration"] *= 0.9
            profile["transition_duration"] *= 0.8
            
        # Adjust for content metrics
        visual_score = content_analysis["visual_interest"]
        info_score = content_analysis["information_density"]
        
        profile["segment_duration"] *= 1 + (info_score - 0.5) * 0.3
        profile["transition_duration"] *= 1 + (visual_score - 0.5) * 0.2
        
        return profile
        
    def apply_pacing(
        self,
        segments: List[Dict[str, Any]],
        pacing: Dict[str, Any],
        target_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Apply pacing configuration to video segments.
        
        Args:
            segments: List of video segments
            pacing: Pacing configuration
            target_duration: Target video duration
            
        Returns:
            List of segments with adjusted timing
        """
        adjusted_segments = []
        current_time = 0.0
        
        # Calculate initial durations
        base_durations = []
        for segment in segments:
            content_type = segment.get("content_type", "interior")
            weight = self.content_weights.get(content_type, 1.0)
            duration = pacing["segment_duration"] * weight
            base_durations.append(duration)
            
        # Scale durations to match target
        total_duration = sum(base_durations)
        scale_factor = (target_duration - 
                       (len(segments) - 1) * pacing["transition_duration"]) / total_duration
        
        # Apply scaled durations
        for segment, base_duration in zip(segments, base_durations):
            adjusted_segment = segment.copy()
            adjusted_segment["duration"] = base_duration * scale_factor
            adjusted_segment["start_time"] = current_time
            
            # Add transition if not last segment
            if len(adjusted_segments) < len(segments) - 1:
                adjusted_segment["transition_duration"] = pacing["transition_duration"]
                current_time += adjusted_segment["duration"] + pacing["transition_duration"]
            else:
                current_time += adjusted_segment["duration"]
                
            adjusted_segments.append(adjusted_segment)
            
        return adjusted_segments
        
    def optimize_engagement(
        self,
        segments: List[Dict[str, Any]],
        engagement_data: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Optimize pacing based on engagement data.
        
        Args:
            segments: List of video segments
            engagement_data: Optional engagement metrics
            
        Returns:
            Optimized segments
        """
        if not engagement_data:
            return segments
            
        optimized = []
        total_duration = sum(s["duration"] for s in segments)
        
        # Calculate engagement-based duration adjustments
        for segment in segments:
            adjusted = segment.copy()
            content_type = segment.get("content_type", "interior")
            
            if content_type in engagement_data:
                # Adjust duration based on engagement (±25% max)
                engagement_score = engagement_data[content_type]
                adjustment = np.clip(engagement_score - 0.5, -0.25, 0.25)
                adjusted["duration"] *= (1 + adjustment)
                
            optimized.append(adjusted)
            
        # Normalize durations
        current_total = sum(s["duration"] for s in optimized)
        scale_factor = total_duration / current_total
        
        current_time = 0.0
        for segment in optimized:
            segment["duration"] *= scale_factor
            segment["start_time"] = current_time
            current_time += segment["duration"]
            
        return optimized
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/agents/music_agent.py
```python
from typing import Dict, Any, List, Optional
import os
from pathlib import Path
import json
import random

class MusicAgent:
    """
    Agent for selecting and managing background music.
    """
    
    def __init__(self, music_dir: Optional[str] = None):
        self.music_dir = Path(music_dir or 'assets/audio/music')
        self.music_dir.mkdir(parents=True, exist_ok=True)
        self._load_music_catalog()
        
    def _load_music_catalog(self) -> None:
        """Load music catalog from JSON file or create default."""
        catalog_file = self.music_dir / 'catalog.json'
        
        if catalog_file.exists():
            with open(catalog_file, 'r') as f:
                self.catalog = json.load(f)
        else:
            self.catalog = self._create_default_catalog()
            with open(catalog_file, 'w') as f:
                json.dump(self.catalog, f, indent=2)
                
    def _create_default_catalog(self) -> Dict[str, Any]:
        """Create default music catalog."""
        return {
            'genres': {
                'upbeat_electronic': {
                    'mood': 'energetic',
                    'tempo': 'fast',
                    'tracks': [
                        {
                            'title': 'Modern Momentum',
                            'duration': 60,
                            'file': 'modern_momentum.mp3',
                            'tags': ['modern', 'upbeat', 'electronic']
                        }
                    ]
                },
                'corporate': {
                    'mood': 'professional',
                    'tempo': 'medium',
                    'tracks': [
                        {
                            'title': 'Business Progress',
                            'duration': 60,
                            'file': 'business_progress.mp3',
                            'tags': ['corporate', 'professional', 'clean']
                        }
                    ]
                },
                'ambient': {
                    'mood': 'luxury',
                    'tempo': 'slow',
                    'tracks': [
                        {
                            'title': 'Elegant Atmosphere',
                            'duration': 60,
                            'file': 'elegant_atmosphere.mp3',
                            'tags': ['ambient', 'luxury', 'sophisticated']
                        }
                    ]
                }
            },
            'moods': ['energetic', 'professional', 'luxury'],
            'tempos': ['slow', 'medium', 'fast']
        }
        
    async def select_music(self,
                         style: str,
                         duration: int,
                         mood: Optional[str] = None) -> Dict[str, Any]:
        """
        Select appropriate background music.
        
        Args:
            style: Video style
            duration: Required duration in seconds
            mood: Optional specific mood
            
        Returns:
            Selected track information
        """
        # Map style to genre
        style_genre_map = {
            'energetic': 'upbeat_electronic',
            'professional': 'corporate',
            'luxury': 'ambient'
        }
        
        genre = style_genre_map.get(style, 'upbeat_electronic')
        genre_info = self.catalog['genres'][genre]
        
        # Filter tracks by duration
        suitable_tracks = [
            track for track in genre_info['tracks']
            if track['duration'] >= duration
        ]
        
        if not suitable_tracks:
            # If no tracks match duration, use any track from genre
            suitable_tracks = genre_info['tracks']
            
        # Select random track from suitable ones
        selected_track = random.choice(suitable_tracks)
        
        return {
            'track': selected_track,
            'genre': genre,
            'mood': genre_info['mood'],
            'tempo': genre_info['tempo']
        }
        
    async def process_music(self,
                          track_path: str,
                          duration: int,
                          volume: float = 0.5) -> str:
        """
        Process music track for video.
        
        Args:
            track_path: Path to music track
            duration: Required duration
            volume: Background music volume (0-1)
            
        Returns:
            Path to processed music file
        """
        # TODO: Implement music processing
        # - Trim to duration
        # - Adjust volume
        # - Add fade in/out
        return track_path
        
    def get_available_tracks(self, 
                           genre: Optional[str] = None, 
                           mood: Optional[str] = None,
                           tempo: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of available tracks with optional filters."""
        tracks = []
        
        for genre_name, genre_info in self.catalog['genres'].items():
            if genre and genre != genre_name:
                continue
                
            if mood and genre_info['mood'] != mood:
                continue
                
            if tempo and genre_info['tempo'] != tempo:
                continue
                
            tracks.extend(genre_info['tracks'])
            
        return tracks
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/agents/style_agent.py
```python
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

class StyleAgent:
    """
    Agent for managing and varying video styles.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default style configuration."""
        return {
            'styles': {
                'energetic': {
                    'video': {
                        'transitions': ['fast_fade', 'zoom_blur', 'slide'],
                        'effects': ['zoom', 'shake', 'flash'],
                        'pacing': 'fast',
                        'color_grade': 'vibrant'
                    },
                    'text': {
                        'font': 'Impact',
                        'animation': 'pop',
                        'color_scheme': ['#FF4B4B', '#FFFFFF'],
                        'placement': 'dynamic'
                    },
                    'audio': {
                        'music_style': 'upbeat_electronic',
                        'voice_style': 'friendly',
                        'sound_effects': True
                    }
                },
                'professional': {
                    'video': {
                        'transitions': ['smooth_fade', 'dissolve', 'wipe'],
                        'effects': ['pan', 'tilt', 'focus'],
                        'pacing': 'medium',
                        'color_grade': 'natural'
                    },
                    'text': {
                        'font': 'Helvetica',
                        'animation': 'fade',
                        'color_scheme': ['#2C3E50', '#ECF0F1'],
                        'placement': 'structured'
                    },
                    'audio': {
                        'music_style': 'corporate',
                        'voice_style': 'professional',
                        'sound_effects': False
                    }
                },
                'luxury': {
                    'video': {
                        'transitions': ['cinematic_fade', 'blur_dissolve', 'elegant_slide'],
                        'effects': ['depth_blur', 'light_rays', 'film_grain'],
                        'pacing': 'slow',
                        'color_grade': 'cinematic'
                    },
                    'text': {
                        'font': 'Didot',
                        'animation': 'elegant',
                        'color_scheme': ['#B8860B', '#FFFFFF'],
                        'placement': 'minimal'
                    },
                    'audio': {
                        'music_style': 'ambient',
                        'voice_style': 'luxury',
                        'sound_effects': False
                    }
                }
            },
            'variations': {
                'color_grades': ['natural', 'vibrant', 'cinematic', 'moody', 'warm'],
                'text_animations': ['fade', 'pop', 'slide', 'elegant', 'dynamic'],
                'transition_types': ['fade', 'dissolve', 'slide', 'zoom', 'blur']
            }
        }
        
    def get_style(self, style_name: str) -> Dict[str, Any]:
        """Get complete style configuration."""
        return self.config['styles'].get(style_name, self.config['styles']['professional'])
        
    def generate_variation(self,
                         base_style: str,
                         variation_strength: float = 0.3) -> Dict[str, Any]:
        """
        Generate a variation of a base style.
        
        Args:
            base_style: Name of base style
            variation_strength: How much to vary (0-1)
            
        Returns:
            Modified style configuration
        """
        base_config = self.get_style(base_style).copy()
        
        if variation_strength <= 0:
            return base_config
            
        # Vary video effects
        if random.random() < variation_strength:
            base_config['video']['effects'] = random.sample(
                base_config['video']['effects'],
                k=max(1, len(base_config['video']['effects']) - 1)
            )
            
        # Vary color grade
        if random.random() < variation_strength:
            base_config['video']['color_grade'] = random.choice(
                self.config['variations']['color_grades']
            )
            
        # Vary text animation
        if random.random() < variation_strength:
            base_config['text']['animation'] = random.choice(
                self.config['variations']['text_animations']
            )
            
        return base_config
        
    def adapt_style_to_content(self,
                             content_type: str,
                             property_value: float,
                             target_market: str) -> str:
        """
        Adapt style based on content and target market.
        
        Args:
            content_type: Type of property
            property_value: Property value
            target_market: Target market description
            
        Returns:
            Recommended style name
        """
        # Luxury properties
        if property_value > 1000000 or 'luxury' in target_market.lower():
            return 'luxury'
            
        # Young/urban market
        if any(term in target_market.lower() for term in ['young', 'urban', 'professional']):
            return 'energetic'
            
        # Default to professional
        return 'professional'
        
    def get_style_elements(self,
                         style_name: str,
                         element_type: str) -> Dict[str, Any]:
        """Get specific elements of a style."""
        style = self.get_style(style_name)
        return style.get(element_type, {})
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/tools/video_renderer.py
```python
from typing import Dict, Any, List, Optional
import os
import logging
import moviepy.editor as mp
from moviepy.editor import VideoFileClip, ImageClip, TextClip, CompositeVideoClip, AudioFileClip, CompositeAudioClip
import firebase_admin
from firebase_admin import storage
import tempfile
import uuid

logger = logging.getLogger(__name__)

class VideoRenderer:
    """Tool for rendering video segments into a final video."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Firebase if not already initialized
        try:
            self.bucket = storage.bucket()
        except ValueError:
            cred = firebase_admin.credentials.Certificate("config/firebase-service-account.json")
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'project3-ziltok.firebasestorage.app'
            })
            self.bucket = storage.bucket()
    
    def download_image(self, image_url: str) -> str:
        """Download image from Firebase Storage."""
        if not image_url.startswith("gs://"):
            return image_url
            
        # Extract blob path from gs:// URL
        blob_path = image_url.replace("gs://project3-ziltok.firebasestorage.app/", "")
        blob = self.bucket.blob(blob_path)
        
        # Create temp file
        _, temp_path = tempfile.mkstemp(suffix=os.path.splitext(blob_path)[1])
        
        # Download to temp file
        blob.download_to_filename(temp_path)
        return temp_path
    
    def apply_effects(self, clip: VideoFileClip, effects: Dict[str, Any], is_first: bool = False) -> VideoFileClip:
        """Apply video effects to a clip."""
        # Skip effects for first clip
        if is_first:
            return clip
            
        # Apply color adjustments
        if "color" in effects:
            color = effects["color"]
            if "contrast" in color:
                clip = clip.fx(mp.vfx.colorx, color["contrast"])
            if "brightness" in color:
                clip = clip.fx(mp.vfx.colorx, color["brightness"])
            if "saturation" in color:
                clip = clip.fx(mp.vfx.colorx, color["saturation"])
        
        # Apply filters
        if "filters" in effects:
            for filter_name in effects["filters"]:
                if filter_name == "vibrant":
                    clip = clip.fx(mp.vfx.colorx, 1.2)
                elif filter_name == "warm":
                    clip = clip.fx(mp.vfx.colorx, 1.1)
                elif filter_name == "cinematic":
                    clip = clip.fx(mp.vfx.lum_contrast)
        
        # Apply transitions if specified
        if "transition" in effects:
            transition = effects["transition"]
            if transition["name"] == "fade":
                clip = clip.crossfadein(transition["duration"])
            elif transition["name"] == "slide":
                # TODO: Implement slide transition
                pass
            
        return clip
    
    def create_text_overlay(self, text: str, duration: float, size: int = 30) -> TextClip:
        """Create a text overlay clip."""
        return TextClip(text, fontsize=size, color='white', font='Arial', 
                       size=(900, None), method='caption', align='center',  # Wider text for vertical video
                       stroke_color='black', stroke_width=2)\
               .set_duration(duration)\
               .set_position(('center', 'bottom'))\
               .crossfadein(0.3)\
               .crossfadeout(0.3)
    
    def render_video(self, segments: List[Dict[str, Any]], music: Dict[str, Any], 
                    output_path: Optional[str] = None) -> str:
        """
        Render the final video from segments and music.
        
        Args:
            segments: List of video segments with effects
            music: Music configuration
            output_path: Optional custom output path
            
        Returns:
            Path to the rendered video file
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"video_{uuid.uuid4()}.mp4")
            
        clips = []
        
        # Process each segment
        for segment in segments:
            # Download and create optimized image clip
            image_path = self.download_image(segment["image_url"])
            
            # Create clip with TikTok resolution (1080x1920)
            clip = ImageClip(image_path)
            
            # Calculate dimensions to fill TikTok aspect ratio while maintaining center focus
            img_w, img_h = clip.size
            target_ratio = 9/16  # TikTok aspect ratio
            current_ratio = img_w/img_h
            
            if current_ratio > target_ratio:  # Image is too wide
                # Resize based on height
                new_h = 1920
                new_w = int(img_w * (new_h/img_h))
                clip = clip.resize((new_w, new_h))
                # Crop to width
                x_center = new_w/2
                clip = clip.crop(x1=int(x_center-540), x2=int(x_center+540))  # 1080px wide
            else:  # Image is too tall or perfect
                # Resize based on width
                new_w = 1080
                new_h = int(img_h * (new_w/img_w))
                clip = clip.resize((new_w, new_h))
                # Crop to height if needed
                if new_h > 1920:
                    y_center = new_h/2
                    clip = clip.crop(y1=int(y_center-960), y2=int(y_center+960))  # 1920px tall
            
            # Set duration
            clip = clip.set_duration(segment["duration"])
            
            # Apply effects (pass is_first flag)
            if "effects" in segment:
                is_first = (segment == segments[0])
                clip = self.apply_effects(clip, segment["effects"], is_first)
            
            # Add text overlay
            if "text" in segment:
                text_clip = self.create_text_overlay(segment["text"], segment["duration"])
                clip = CompositeVideoClip([clip, text_clip])
            
            clips.append(clip)
        
        # Concatenate all clips with no transitions
        final_clip = mp.concatenate_videoclips(clips, method="compose")
        
        # Get total duration from segments
        total_duration = sum(segment["duration"] for segment in segments)
        # Ensure final clip duration matches segments
        final_clip = final_clip.set_duration(total_duration)
        
        # Process audio tracks
        audio_tracks = []
        
        # Add voiceovers for each segment
        current_time = 0
        for segment in segments:
            if "voiceover_path" in segment:
                try:
                    voiceover = AudioFileClip(segment["voiceover_path"])
                    # Set the start time for this voiceover
                    voiceover = voiceover.set_start(current_time)
                    audio_tracks.append(voiceover)
                except Exception as e:
                    self.logger.warning(f"Failed to load voiceover: {e}")
            current_time += segment["duration"]
        
        # Add background music if specified
        if music and "path" in music:
            try:
                background_music = AudioFileClip(music["path"])
                # Trim or loop music to match video duration
                total_duration = sum(segment["duration"] for segment in segments)
                if background_music.duration > total_duration:
                    background_music = background_music.subclip(0, total_duration)
                elif background_music.duration < total_duration:
                    # Calculate how many loops we need
                    loops_needed = int(total_duration / background_music.duration) + 1
                    loops = [background_music] * loops_needed
                    background_music = concatenate_audioclips(loops)
                    background_music = background_music.subclip(0, total_duration)
                if music.get("fade_out"):
                    background_music = background_music.audio_fadeout(2)
                # Lower the volume of background music
                background_music = background_music.volumex(0.3)
                audio_tracks.append(background_music)
            except Exception as e:
                self.logger.warning(f"Failed to load background music: {e}")
        
        # Combine all audio tracks if we have any
        if audio_tracks:
            try:
                final_audio = CompositeAudioClip(audio_tracks)
                final_clip = final_clip.set_audio(final_audio)
            except Exception as e:
                self.logger.error(f"Failed to combine audio tracks: {e}")
        
        # Write final video with optimized settings
        # Use more compatible encoding settings
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            preset='medium',  # Better compatibility
            threads=4,
            bitrate='4000k',
            ffmpeg_params=[
                "-profile:v", "baseline",
                "-level", "3.0",
                "-pix_fmt", "yuv420p",  # Required for QuickTime
                "-movflags", "+faststart"  # Enable streaming
            ]
        )
        
        # Cleanup temp files
        for clip in clips:
            clip.close()
        
        return output_path
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/tools/video_db_tool.py
```python
from typing import Dict, Any, List, Optional
import random

class VideoDBTool:
    """Tool for video processing and transitions using VideoDBTool."""
    
    def __init__(self):
        self.transition_library = {
            "modern": {
                "standard": [
                    {"name": "slide", "duration": 0.5, "direction": "left"},
                    {"name": "fade", "duration": 0.4, "ease": "smooth"},
                    {"name": "zoom", "duration": 0.6, "scale": 1.2}
                ],
                "dynamic": [
                    {"name": "swipe", "duration": 0.4, "direction": "up"},
                    {"name": "reveal", "duration": 0.5, "direction": "right"},
                    {"name": "blur", "duration": 0.6, "intensity": 0.8}
                ]
            },
            "luxury": {
                "elegant": [
                    {"name": "dissolve", "duration": 0.8, "ease": "smooth"},
                    {"name": "glide", "duration": 0.7, "direction": "up"},
                    {"name": "fade", "duration": 0.6, "ease": "elegant"}
                ],
                "premium": [
                    {"name": "reveal", "duration": 0.9, "direction": "diagonal"},
                    {"name": "sweep", "duration": 0.8, "direction": "right"},
                    {"name": "blur", "duration": 0.7, "intensity": 0.6}
                ]
            },
            "minimal": {
                "clean": [
                    {"name": "fade", "duration": 0.4, "ease": "linear"},
                    {"name": "slide", "duration": 0.5, "direction": "left"},
                    {"name": "dissolve", "duration": 0.4, "ease": "smooth"}
                ],
                "simple": [
                    {"name": "cut", "duration": 0.3},
                    {"name": "fade", "duration": 0.5, "ease": "linear"},
                    {"name": "slide", "duration": 0.4, "direction": "up"}
                ]
            }
        }
        
        self.effect_library = {
            "modern": {
                "color": {
                    "contrast": 1.1,
                    "saturation": 1.2,
                    "brightness": 1.05
                },
                "filters": ["vibrant", "sharp"],
                "overlays": ["grain", "light_leak"]
            },
            "luxury": {
                "color": {
                    "contrast": 1.15,
                    "saturation": 0.95,
                    "brightness": 1.0
                },
                "filters": ["cinematic", "warm"],
                "overlays": ["film", "subtle_flare"]
            },
            "minimal": {
                "color": {
                    "contrast": 1.05,
                    "saturation": 0.9,
                    "brightness": 1.1
                },
                "filters": ["clean", "crisp"],
                "overlays": ["none"]
            }
        }
    
    def get_transitions(
        self,
        template_name: str,
        count: int = 3,
        variation: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Get a list of transitions for a video.
        
        Args:
            template_name: Name of the template being used
            count: Number of transitions to return
            variation: Amount of random variation to apply (0-1)
            
        Returns:
            List of transition configurations
        """
        style = template_name.split('_')[0]
        
        if style not in self.transition_library:
            style = "modern"  # fallback
            
        # Get all transitions for this style
        available_transitions = []
        for category in self.transition_library[style].values():
            available_transitions.extend(category)
            
        # Select random transitions
        selected = []
        for _ in range(count):
            transition = random.choice(available_transitions).copy()
            
            # Add variation
            if "duration" in transition:
                base_duration = transition["duration"]
                min_duration = base_duration * (1 - variation)
                max_duration = base_duration * (1 + variation)
                transition["duration"] = random.uniform(min_duration, max_duration)
                
            selected.append(transition)
            
        return selected
    
    def get_effects(
        self,
        template_name: str,
        intensity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Get video effects configuration.
        
        Args:
            template_name: Name of the template being used
            intensity: Intensity of effects (0-1)
            
        Returns:
            Effects configuration
        """
        style = template_name.split('_')[0]
        
        if style not in self.effect_library:
            style = "modern"  # fallback
            
        effects = self.effect_library[style].copy()
        
        # Adjust effect intensity
        color = effects["color"].copy()
        for key in color:
            deviation = abs(1 - color[key])
            color[key] = 1 + (deviation * intensity * (1 if color[key] > 1 else -1))
            
        effects["color"] = color
        effects["intensity"] = intensity
        
        return effects
    
    def process_segment(
        self,
        segment: Dict[str, Any],
        effects: Dict[str, Any],
        transition: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a video segment with effects and transition.
        
        Args:
            segment: Segment configuration
            effects: Effects to apply
            transition: Optional transition configuration
            
        Returns:
            Processed segment configuration
        """
        processed = segment.copy()
        processed["effects"] = effects
        if transition:
            processed["transition"] = transition
        return processed
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/tools/music_tool.py
```python
from typing import Dict, Any, List, Optional
import random
import os
from ..core.base_tool import BaseTool

class MusicTool(BaseTool):
    """Tool for selecting and managing background music for videos."""
    
    def __init__(self):
        super().__init__()
        self.music_dir = os.path.join(os.getcwd(), "assets/audio/music")
        os.makedirs(self.music_dir, exist_ok=True)
        
        self.music_library = {
            "modern": {
                "upbeat": [
                    {"name": "Urban Energy", "duration": 60, "bpm": 128, "key": "C", "mood": "energetic", "path": os.path.join(self.music_dir, "urban_energy.mp3")},
                    {"name": "City Pulse", "duration": 45, "bpm": 120, "key": "G", "mood": "dynamic", "path": os.path.join(self.music_dir, "city_pulse.mp3")},
                    {"name": "Modern Flow", "duration": 55, "bpm": 125, "key": "Am", "mood": "groovy", "path": os.path.join(self.music_dir, "modern_flow.mp3")}
                ],
                "chill": [
                    {"name": "Lofi Dreams", "duration": 50, "bpm": 90, "key": "Dm", "mood": "relaxed", "path": os.path.join(self.music_dir, "lofi_dreams.mp3")},
                    {"name": "Urban Calm", "duration": 65, "bpm": 85, "key": "Em", "mood": "peaceful", "path": os.path.join(self.music_dir, "urban_calm.mp3")},
                    {"name": "City Nights", "duration": 48, "bpm": 95, "key": "F", "mood": "smooth", "path": os.path.join(self.music_dir, "city_nights.mp3")}
                ]
            },
            "luxury": {
                "elegant": [
                    {"name": "Golden Hour", "duration": 55, "bpm": 110, "key": "D", "mood": "sophisticated", "path": os.path.join(self.music_dir, "golden_hour.mp3")},
                    {"name": "Luxury Suite", "duration": 62, "bpm": 105, "key": "Em", "mood": "refined", "path": os.path.join(self.music_dir, "luxury_suite.mp3")},
                    {"name": "Elite Status", "duration": 58, "bpm": 108, "key": "Bm", "mood": "prestigious", "path": os.path.join(self.music_dir, "elite_status.mp3")}
                ],
                "ambient": [
                    {"name": "Velvet Dreams", "duration": 70, "bpm": 80, "key": "C", "mood": "atmospheric", "path": os.path.join(self.music_dir, "velvet_dreams.mp3")},
                    {"name": "Premium Space", "duration": 65, "bpm": 75, "key": "G", "mood": "ethereal", "path": os.path.join(self.music_dir, "premium_space.mp3")},
                    {"name": "High End", "duration": 60, "bpm": 85, "key": "Am", "mood": "luxurious", "path": os.path.join(self.music_dir, "high_end.mp3")}
                ]
            },
            "minimal": {
                "ambient": [
                    {"name": "Pure Space", "duration": 45, "bpm": 70, "key": "C", "mood": "minimal", "path": os.path.join(self.music_dir, "pure_space.mp3")},
                    {"name": "Clean Lines", "duration": 50, "bpm": 65, "key": "Em", "mood": "spacious", "path": os.path.join(self.music_dir, "clean_lines.mp3")},
                    {"name": "Simple Beauty", "duration": 55, "bpm": 72, "key": "G", "mood": "clean", "path": os.path.join(self.music_dir, "simple_beauty.mp3")}
                ],
                "soft": [
                    {"name": "Gentle Flow", "duration": 58, "bpm": 80, "key": "Am", "mood": "subtle", "path": os.path.join(self.music_dir, "gentle_flow.mp3")},
                    {"name": "Soft Touch", "duration": 52, "bpm": 75, "key": "Dm", "mood": "delicate", "path": os.path.join(self.music_dir, "soft_touch.mp3")},
                    {"name": "Light Air", "duration": 48, "bpm": 78, "key": "F", "mood": "airy", "path": os.path.join(self.music_dir, "light_air.mp3")}
                ]
            }
        }
        
        # Mapping of template pacing to music categories
        self.pacing_music_map = {
            "rapid": ["upbeat", "elegant"],
            "balanced": ["upbeat", "elegant", "ambient"],
            "relaxed": ["chill", "ambient", "soft"]
        }
    
    def select_music(
        self,
        template_name: str,
        duration: float,
        mood_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Select appropriate background music for a video.
        
        Args:
            template_name: Name of the template being used
            duration: Desired duration of the video
            mood_override: Optional override for mood selection
            
        Returns:
            Dictionary containing selected music track information
        """
        # For now, use a default track for testing
        track = {
            "name": "Lofi Dreams",
            "duration": 50,
            "bpm": 90,
            "key": "Dm",
            "mood": "relaxed",
            "path": os.path.join(self.music_dir, "lofi_dreams.mp3")
        }
        
        # Calculate number of loops needed
        loop_count = int(duration / track["duration"]) + 1
        
        # Add fade out if looping
        fade_out = loop_count > 1
        
        return {
            **track,
            "loop_count": loop_count,
            "fade_out": fade_out
        }
        
        if not matching_tracks:
            raise ValueError(f"No suitable music found for template {template_name}")
        
        # Select random track from matches
        selected_track = random.choice(matching_tracks)
        
        # Add any necessary adjustments for duration
        result = selected_track.copy()
        result["loop_count"] = max(1, int(duration / selected_track["duration"]))
        result["fade_out"] = True if result["loop_count"] > 1 else False
        
        return result
    
    def get_available_moods(self, template_name: str) -> List[str]:
        """Get available music moods for a template."""
        style = template_name.split('_')[0]
        moods = set()
        
        if style in self.music_library:
            for category in self.music_library[style].values():
                for track in category:
                    moods.add(track["mood"])
        
        return sorted(list(moods))
    
    def get_track_by_name(self, track_name: str) -> Optional[Dict[str, Any]]:
        """Get track information by name."""
        for style in self.music_library.values():
            for category in style.values():
                for track in category:
                    if track["name"] == track_name:
                        return track
        return None
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Get tool capabilities and requirements."""
        return {
            "name": "music_tool",
            "description": "Selects and manages background music for videos",
            "supported_styles": list(self.music_library.keys()),
            "features": [
                "style_based_selection",
                "mood_matching",
                "tempo_matching",
                "duration_adjustment"
            ]
        }
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/tools/voiceover_tool.py
```python
from typing import Dict, Any, Optional
import os
import logging
from ..core.base_tool import BaseTool

logger = logging.getLogger(__name__)

class VoiceoverTool(BaseTool):
    """Tool for generating voiceovers using ElevenLabs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        super().__init__(config)
        if not self.api_key:
            logger.warning("ELEVENLABS_API_KEY not found in environment")
        
        # Load configuration from environment
        self.model = os.getenv("ELEVENLABS_MODEL", "eleven_monolingual_v1")
        self.default_stability = float(os.getenv("ELEVENLABS_VOICE_STABILITY", "0.75"))
        self.default_similarity = float(os.getenv("ELEVENLABS_VOICE_SIMILARITY", "0.75"))
            
        self.voice_profiles = {
            "luxury": {
                "voice_id": "ErXwobaYiN019PkySvjV",  # Antoni - Professional male voice
                "settings": {
                    "stability": self.default_stability,
                    "similarity_boost": self.default_similarity,
                    "style": 0.5,
                    "use_speaker_boost": True
                }
            },
            "friendly": {
                "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Bella - Female voice
                "settings": {
                    "stability": self.default_stability,
                    "similarity_boost": self.default_similarity,
                    "style": 0.3,
                    "use_speaker_boost": True
                }
            },
            "professional": {
                "voice_id": "VR6AewLTigWG4xSOukaG",  # Adam - Professional male voice
                "settings": {
                    "stability": self.default_stability,
                    "similarity_boost": self.default_similarity,
                    "style": 0.4,
                    "use_speaker_boost": True
                }
            }
        }
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Get tool capabilities and requirements."""
        return {
            "name": "voiceover_tool",
            "description": "Generates professional voiceovers using ElevenLabs API",
            "requires_api_key": True,
            "supported_voices": list(self.voice_profiles.keys()),
            "max_text_length": 5000,
            "supported_formats": ["mp3", "wav"]
        }
        
    def generate_voiceover(
        self,
        text: str,
        style: str = "professional",
        output_format: str = "mp3"
    ) -> Dict[str, Any]:
        """
        Generate a voiceover for the given text using ElevenLabs API.
        
        Args:
            text: Text to convert to speech
            style: Voice style to use
            output_format: Output audio format
            
        Returns:
            Dictionary containing voiceover metadata and file path
        """
        if not self.api_key:
            raise ValueError("ElevenLabs API key not configured")
            
        if style not in self.voice_profiles:
            raise ValueError(f"Unknown voice style: {style}")
            
        if len(text) > self.get_capabilities()["max_text_length"]:
            raise ValueError("Text exceeds maximum length")
            
        try:
            import requests
            import uuid
            
            # Get voice profile
            profile = self.voice_profiles[style]
            
            # Prepare request
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{profile['voice_id']}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": self.model,
                "voice_settings": profile["settings"]
            }
            
            # Make API request
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                # Save audio file
                output_dir = os.path.join(os.getcwd(), "assets/audio/voiceover")
                os.makedirs(output_dir, exist_ok=True)
                
                file_name = f"voiceover_{uuid.uuid4()}.{output_format}"
                file_path = os.path.join(output_dir, file_name)
                
                with open(file_path, "wb") as f:
                    f.write(response.content)
                
                return {
                    "status": "success",
                    "file_path": file_path,
                    "text": text,
                    "voice_id": profile["voice_id"],
                    "settings": profile["settings"]
                }
            else:
                error_msg = f"ElevenLabs API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
        except Exception as e:
            logger.error(f"Voiceover generation failed: {str(e)}")
            raise
            
    def validate_config(self) -> None:
        """Validate tool configuration."""
        if not self.api_key and not self.config.get("mock_mode", False):
            logger.warning("No API key provided and mock mode is disabled")
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/effects/text_overlay.py
```python
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path
import json

class TextStyle:
    """Text styling configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.font_face = config.get('font_face', cv2.FONT_HERSHEY_DUPLEX)
        self.font_scale = config.get('font_scale', 1.0)
        self.color = config.get('color', (255, 255, 255))
        self.thickness = config.get('thickness', 2)
        self.background_color = config.get('background_color')
        self.padding = config.get('padding', 10)
        self.shadow = config.get('shadow', True)
        self.shadow_color = config.get('shadow_color', (0, 0, 0))
        self.animation = config.get('animation', 'fade')

class TextOverlay:
    """Dynamic text overlay system for videos."""
    
    def __init__(self, style: Optional[Dict[str, Any]] = None):
        self.style = TextStyle(style or {})
        self._load_fonts()
        
    def _load_fonts(self):
        """Load custom fonts if available."""
        # TODO: Implement custom font loading
        pass
        
    async def render_text(self,
                         frame: np.ndarray,
                         text: str,
                         position: Tuple[float, float],
                         progress: float = 1.0,
                         style: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Render text overlay on frame.
        
        Args:
            frame: Input video frame
            text: Text to overlay
            position: Position as (x, y) in relative coordinates (0-1)
            progress: Animation progress (0-1)
            style: Optional style override
            
        Returns:
            Frame with text overlay
        """
        text_style = TextStyle(style) if style else self.style
        frame_h, frame_w = frame.shape[:2]
        
        # Convert relative position to absolute pixels
        x = int(position[0] * frame_w)
        y = int(position[1] * frame_h)
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            text_style.font_face,
            text_style.font_scale,
            text_style.thickness
        )
        
        # Apply animation
        if text_style.animation == 'fade':
            alpha = progress
        elif text_style.animation == 'slide':
            x = int(x + (frame_w * (1 - progress)))
        elif text_style.animation == 'pop':
            text_style.font_scale *= progress
            
        # Draw background if specified
        if text_style.background_color:
            bg_x1 = x - text_style.padding
            bg_y1 = y - text_h - text_style.padding
            bg_x2 = x + text_w + text_style.padding
            bg_y2 = y + text_style.padding
            
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (bg_x1, bg_y1),
                (bg_x2, bg_y2),
                text_style.background_color,
                -1
            )
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
        # Draw shadow if enabled
        if text_style.shadow:
            shadow_offset = max(1, int(text_style.thickness / 2))
            cv2.putText(
                frame,
                text,
                (x + shadow_offset, y + shadow_offset),
                text_style.font_face,
                text_style.font_scale,
                text_style.shadow_color,
                text_style.thickness
            )
            
        # Draw text
        cv2.putText(
            frame,
            text,
            (x, y),
            text_style.font_face,
            text_style.font_scale,
            text_style.color,
            text_style.thickness
        )
        
        return frame

class TextTemplate:
    """Predefined text overlay templates."""
    
    def __init__(self, template_path: Optional[str] = None):
        self.templates = self._load_templates(template_path)
        
    def _load_templates(self, template_path: Optional[str]) -> Dict[str, Any]:
        """Load text templates from JSON file."""
        if not template_path:
            return self._default_templates()
            
        template_path = Path(template_path)
        if template_path.exists():
            with open(template_path, 'r') as f:
                return json.load(f)
                
        return self._default_templates()
        
    def _default_templates(self) -> Dict[str, Any]:
        """Default text overlay templates."""
        return {
            'price': {
                'style': {
                    'font_scale': 1.5,
                    'color': (0, 255, 0),
                    'background_color': (0, 0, 0),
                    'animation': 'pop'
                },
                'position': (0.1, 0.9)
            },
            'feature': {
                'style': {
                    'font_scale': 1.0,
                    'color': (255, 255, 255),
                    'background_color': None,
                    'animation': 'slide'
                },
                'position': (0.05, 0.8)
            },
            'call_to_action': {
                'style': {
                    'font_scale': 1.2,
                    'color': (255, 255, 0),
                    'background_color': (0, 0, 0),
                    'animation': 'fade'
                },
                'position': (0.5, 0.5)
            }
        }
        
    def get_template(self, name: str) -> Dict[str, Any]:
        """Get template by name."""
        if name not in self.templates:
            raise ValueError(f"Unknown template: {name}")
        return self.templates[name]
        
    def list_templates(self) -> List[str]:
        """List available templates."""
        return list(self.templates.keys())
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/effects/visual_effects.py
```python
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from pathlib import Path

class VisualEffect:
    """Base class for visual effects."""
    
    def __init__(self, intensity: float = 1.0):
        self.intensity = max(0.0, min(1.0, intensity))
        
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply effect to frame."""
        raise NotImplementedError

class ColorEnhance(VisualEffect):
    """Enhance colors in frame."""
    
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Enhance saturation
        hsv[:, :, 1] = hsv[:, :, 1] * (1 + self.intensity * 0.5)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back to BGR
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

class Sharpness(VisualEffect):
    """Adjust image sharpness."""
    
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]]) * self.intensity
        return cv2.filter2D(frame, -1, kernel)

class Vignette(VisualEffect):
    """Add vignette effect."""
    
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        rows, cols = frame.shape[:2]
        
        # Generate vignette mask
        X_resultant, Y_resultant = np.meshgrid(np.linspace(-1, 1, cols),
                                             np.linspace(-1, 1, rows))
        mask = np.sqrt(X_resultant**2 + Y_resultant**2)
        mask = 1 - mask
        mask = np.clip(mask + (1 - self.intensity), 0, 1)
        
        # Expand mask to match frame channels
        mask = np.dstack([mask] * 3)
        
        return (frame * mask).astype(np.uint8)

class FilmGrain(VisualEffect):
    """Add film grain effect."""
    
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, 25 * self.intensity, frame.shape).astype(np.uint8)
        return cv2.add(frame, noise)

class CinematicBars(VisualEffect):
    """Add cinematic bars effect."""
    
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        height = frame.shape[0]
        bar_height = int(height * 0.1 * self.intensity)
        
        frame[:bar_height] = 0  # Top bar
        frame[-bar_height:] = 0  # Bottom bar
        return frame

class EffectLibrary:
    """Library of available visual effects."""
    
    def __init__(self):
        self.effects = {
            'color_enhance': ColorEnhance,
            'sharpness': Sharpness,
            'vignette': Vignette,
            'film_grain': FilmGrain,
            'cinematic_bars': CinematicBars
        }
        
    def get_effect(self, name: str, intensity: float = 1.0) -> VisualEffect:
        """Get effect by name."""
        if name not in self.effects:
            raise ValueError(f"Unknown effect: {name}")
            
        return self.effects[name](intensity)
        
    def list_effects(self) -> List[str]:
        """List available effects."""
        return list(self.effects.keys())
        
    def get_effect_info(self, name: str) -> Dict[str, Any]:
        """Get information about an effect."""
        if name not in self.effects:
            raise ValueError(f"Unknown effect: {name}")
            
        effect_class = self.effects[name]
        return {
            'name': name,
            'description': effect_class.__doc__,
            'parameters': {
                'intensity': 'Effect intensity (0-1)'
            }
        }

class EffectChain:
    """Chain multiple visual effects together."""
    
    def __init__(self):
        self.effects: List[VisualEffect] = []
        self.library = EffectLibrary()
        
    def add_effect(self, name: str, intensity: float = 1.0) -> 'EffectChain':
        """Add effect to chain."""
        effect = self.library.get_effect(name, intensity)
        self.effects.append(effect)
        return self
        
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply all effects in chain."""
        result = frame.copy()
        for effect in self.effects:
            result = await effect.apply(result)
        return result
        
    def clear(self) -> None:
        """Clear all effects from chain."""
        self.effects.clear()
```

### /Users/Learn/Desktop/gauntlet-projects/real_estate_tok/video_service/src/director/effects/transitions.py
```python
from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path
import cv2

class TransitionEffect:
    """Base class for video transitions."""
    
    def __init__(self, duration: float = 0.5):
        self.duration = duration
        
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        """Apply transition effect between two frames."""
        raise NotImplementedError

class CrossDissolve(TransitionEffect):
    """Smooth fade transition between frames."""
    
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        return cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)

class SlideTransition(TransitionEffect):
    """Slide one frame over another."""
    
    def __init__(self, direction: str = 'left', duration: float = 0.5):
        super().__init__(duration)
        self.direction = direction
        
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        height, width = frame1.shape[:2]
        result = np.zeros_like(frame1)
        
        if self.direction == 'left':
            offset = int(width * progress)
            result[:, :width-offset] = frame1[:, offset:]
            result[:, width-offset:] = frame2[:, :offset]
        elif self.direction == 'right':
            offset = int(width * progress)
            result[:, offset:] = frame1[:, :width-offset]
            result[:, :offset] = frame2[:, width-offset:]
        
        return result

class ZoomTransition(TransitionEffect):
    """Zoom in/out transition effect."""
    
    def __init__(self, zoom_in: bool = True, duration: float = 0.5):
        super().__init__(duration)
        self.zoom_in = zoom_in
        
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        if self.zoom_in:
            scale = 1 + progress
            frame1_scaled = cv2.resize(frame1, None, fx=scale, fy=scale)
            h1, w1 = frame1_scaled.shape[:2]
            y1 = int((h1 - frame1.shape[0]) / 2)
            x1 = int((w1 - frame1.shape[1]) / 2)
            frame1_crop = frame1_scaled[y1:y1+frame1.shape[0], x1:x1+frame1.shape[1]]
            return cv2.addWeighted(frame1_crop, 1 - progress, frame2, progress, 0)
        else:
            scale = 2 - progress
            frame2_scaled = cv2.resize(frame2, None, fx=scale, fy=scale)
            h2, w2 = frame2_scaled.shape[:2]
            y2 = int((h2 - frame2.shape[0]) / 2)
            x2 = int((w2 - frame2.shape[1]) / 2)
            frame2_crop = frame2_scaled[y2:y2+frame2.shape[0], x2:x2+frame2.shape[1]]
            return cv2.addWeighted(frame1, 1 - progress, frame2_crop, progress, 0)

class WhipPanTransition(TransitionEffect):
    """Fast whip pan transition effect."""
    
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        if progress < 0.5:
            # Apply motion blur to first frame
            kernel_size = int((progress * 2) * 50)
            if kernel_size > 0:
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
                kernel = kernel / kernel_size
                frame1 = cv2.filter2D(frame1, -1, kernel)
            return frame1
        else:
            # Apply motion blur to second frame
            kernel_size = int((1 - progress) * 2 * 50)
            if kernel_size > 0:
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
                kernel = kernel / kernel_size
                frame2 = cv2.filter2D(frame2, -1, kernel)
            return frame2

class TransitionLibrary:
    """Library of available transition effects."""
    
    def __init__(self):
        self.transitions = {
            'dissolve': CrossDissolve,
            'slide_left': lambda: SlideTransition('left'),
            'slide_right': lambda: SlideTransition('right'),
            'zoom_in': lambda: ZoomTransition(True),
            'zoom_out': lambda: ZoomTransition(False),
            'whip_pan': WhipPanTransition
        }
        
    def get_transition(self, name: str, **kwargs) -> TransitionEffect:
        """Get transition effect by name."""
        if name not in self.transitions:
            raise ValueError(f"Unknown transition effect: {name}")
            
        transition_class = self.transitions[name]
        if callable(transition_class):
            return transition_class()
        return transition_class(**kwargs)
        
    def list_transitions(self) -> List[str]:
        """List available transition effects."""
        return list(self.transitions.keys())
        
    def get_transition_info(self, name: str) -> Dict[str, Any]:
        """Get information about a transition effect."""
        if name not in self.transitions:
            raise ValueError(f"Unknown transition effect: {name}")
            
        transition_class = self.transitions[name]
        return {
            'name': name,
            'description': transition_class.__doc__,
            'parameters': {
                'duration': 'Duration of transition in seconds'
            }
        }
```

