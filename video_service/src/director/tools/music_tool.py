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
