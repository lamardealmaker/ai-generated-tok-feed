from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class PacingPoint:
    """Represents a point in the video with specific pacing."""
    time: float
    intensity: float
    duration: float

class PacingCurve:
    """Generates dynamic pacing curves for video timing."""
    
    def __init__(self, duration: float, base_intensity: float = 0.5):
        self.duration = duration
        self.base_intensity = base_intensity
        self.points: List[PacingPoint] = []
        
    def add_point(self, time: float, intensity: float, duration: float) -> None:
        """Add a pacing point."""
        self.points.append(PacingPoint(time, intensity, duration))
        # Sort points by time
        self.points.sort(key=lambda p: p.time)
        
    def get_intensity(self, time: float) -> float:
        """Get intensity at specific time."""
        if not self.points:
            return self.base_intensity
            
        # Find surrounding points
        prev_point = None
        next_point = None
        
        for point in self.points:
            if point.time <= time:
                prev_point = point
            else:
                next_point = point
                break
                
        if not prev_point:
            return self.points[0].intensity
        if not next_point:
            return prev_point.intensity
            
        # Interpolate between points
        t = (time - prev_point.time) / (next_point.time - prev_point.time)
        return prev_point.intensity * (1 - t) + next_point.intensity * t

class PacingEngine:
    """Engine for managing video pacing."""
    
    def __init__(self):
        self.pacing_presets = self._create_presets()
        
    def _create_presets(self) -> Dict[str, Dict[str, Any]]:
        """Create pacing presets."""
        return {
            'energetic': {
                'base_intensity': 0.7,
                'points': [
                    {'time': 0, 'intensity': 0.8, 'duration': 3},
                    {'time': 15, 'intensity': 0.9, 'duration': 2},
                    {'time': 30, 'intensity': 1.0, 'duration': 1},
                    {'time': 45, 'intensity': 0.9, 'duration': 2}
                ],
                'transition_frequency': 'high',
                'effect_intensity': 'high'
            },
            'balanced': {
                'base_intensity': 0.5,
                'points': [
                    {'time': 0, 'intensity': 0.6, 'duration': 4},
                    {'time': 20, 'intensity': 0.7, 'duration': 3},
                    {'time': 40, 'intensity': 0.6, 'duration': 3}
                ],
                'transition_frequency': 'medium',
                'effect_intensity': 'medium'
            },
            'cinematic': {
                'base_intensity': 0.3,
                'points': [
                    {'time': 0, 'intensity': 0.4, 'duration': 5},
                    {'time': 25, 'intensity': 0.5, 'duration': 4},
                    {'time': 50, 'intensity': 0.4, 'duration': 4}
                ],
                'transition_frequency': 'low',
                'effect_intensity': 'low'
            }
        }
        
    def create_pacing_curve(self, 
                          preset: str,
                          duration: float,
                          variation: float = 0.2) -> PacingCurve:
        """
        Create pacing curve from preset.
        
        Args:
            preset: Preset name
            duration: Video duration
            variation: Amount of random variation (0-1)
        """
        if preset not in self.pacing_presets:
            raise ValueError(f"Unknown preset: {preset}")
            
        preset_data = self.pacing_presets[preset]
        base_intensity = preset_data['base_intensity']
        
        # Add variation to base intensity
        if variation > 0:
            base_intensity += np.random.uniform(-variation, variation)
            base_intensity = max(0.1, min(1.0, base_intensity))
            
        curve = PacingCurve(duration, base_intensity)
        
        # Add points with variation
        for point in preset_data['points']:
            time_ratio = point['time'] / 60  # Original points based on 60s
            time = time_ratio * duration
            
            # Add random variation to each point
            intensity = point['intensity']
            if variation > 0:
                variation_scale = np.random.uniform(1 - variation, 1 + variation)
                intensity *= variation_scale
                intensity = max(0.1, min(1.0, intensity))
                
            curve.add_point(time, intensity, point['duration'])
            
        return curve
        
    def get_segment_duration(self,
                           base_duration: float,
                           intensity: float) -> float:
        """
        Calculate segment duration based on intensity.
        
        Args:
            base_duration: Base segment duration
            intensity: Current pacing intensity
            
        Returns:
            Adjusted duration
        """
        # Higher intensity = shorter duration
        return base_duration * (1.5 - intensity)
        
    def get_transition_duration(self,
                              base_duration: float,
                              intensity: float) -> float:
        """
        Calculate transition duration based on intensity.
        
        Args:
            base_duration: Base transition duration
            intensity: Current pacing intensity
            
        Returns:
            Adjusted duration
        """
        # Higher intensity = faster transitions
        return base_duration * (2 - intensity)
        
    def get_effect_intensity(self,
                           base_intensity: float,
                           pacing_intensity: float) -> float:
        """
        Calculate effect intensity based on pacing.
        
        Args:
            base_intensity: Base effect intensity
            pacing_intensity: Current pacing intensity
            
        Returns:
            Adjusted intensity
        """
        # Higher pacing = stronger effects
        return base_intensity * (0.5 + pacing_intensity * 0.5)
