from typing import Dict, Any, List, Optional
import random
from ..core.base_tool import BaseTool

class MusicLibraryTool(BaseTool):
    """Tool for music selection and management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.music_library = {
            "luxury": {
                "tracks": [
                    {
                        "id": "lux_001",
                        "name": "Elegant Journey",
                        "duration": 120,
                        "tempo": "moderate",
                        "mood": "sophisticated",
                        "key": "C major"
                    },
                    {
                        "id": "lux_002",
                        "name": "Premium Lifestyle",
                        "duration": 180,
                        "tempo": "slow",
                        "mood": "upscale",
                        "key": "G major"
                    }
                ],
                "transitions": [
                    {"name": "smooth_fade", "duration": 2.0},
                    {"name": "orchestral_swell", "duration": 3.0}
                ]
            },
            "modern": {
                "tracks": [
                    {
                        "id": "mod_001",
                        "name": "Urban Pulse",
                        "duration": 90,
                        "tempo": "fast",
                        "mood": "energetic",
                        "key": "D minor"
                    },
                    {
                        "id": "mod_002",
                        "name": "City Vibes",
                        "duration": 150,
                        "tempo": "moderate",
                        "mood": "trendy",
                        "key": "A minor"
                    }
                ],
                "transitions": [
                    {"name": "quick_cut", "duration": 0.5},
                    {"name": "beat_sync", "duration": 1.0}
                ]
            }
        }
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Get tool capabilities and requirements."""
        return {
            "name": "music_library_tool",
            "description": "Manages music selection and background tracks",
            "supported_styles": list(self.music_library.keys()),
            "features": [
                "track_selection",
                "tempo_matching",
                "mood_analysis",
                "transition_generation"
            ]
        }
        
    def select_track(
        self,
        style: str,
        duration: float,
        mood: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Select an appropriate music track.
        
        Args:
            style: Music style to use
            duration: Target duration
            mood: Optional mood preference
            
        Returns:
            Selected track information
        """
        if style not in self.music_library:
            raise ValueError(f"Unknown music style: {style}")
            
        available_tracks = self.music_library[style]["tracks"]
        
        # Filter by mood if specified
        if mood:
            available_tracks = [
                t for t in available_tracks
                if mood.lower() in t["mood"].lower()
            ]
            
        if not available_tracks:
            raise ValueError(f"No tracks found for style={style}, mood={mood}")
            
        # Select best matching track
        track = random.choice(available_tracks)
        
        return {
            "track": track,
            "start_time": 0.0,
            "end_time": min(duration, track["duration"]),
            "fade_in": 1.0,
            "fade_out": 2.0
        }
        
    def generate_transition(
        self,
        style: str,
        duration: float
    ) -> Dict[str, Any]:
        """
        Generate a music transition.
        
        Args:
            style: Music style to use
            duration: Transition duration
            
        Returns:
            Transition configuration
        """
        if style not in self.music_library:
            raise ValueError(f"Unknown music style: {style}")
            
        transitions = self.music_library[style]["transitions"]
        transition = random.choice(transitions)
        
        return {
            "name": transition["name"],
            "duration": min(duration, transition["duration"]),
            "fade_curve": "smooth"
        }
        
    def analyze_mood(self, property_data: Dict[str, Any]) -> str:
        """Analyze property data to determine appropriate music mood."""
        price = property_data.get("price", 0)
        features = property_data.get("features", [])
        
        if price > 1000000 or "luxury" in " ".join(features).lower():
            return "sophisticated"
        elif "modern" in " ".join(features).lower():
            return "trendy"
        else:
            return "neutral"
