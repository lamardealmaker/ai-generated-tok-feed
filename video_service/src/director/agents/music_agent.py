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
