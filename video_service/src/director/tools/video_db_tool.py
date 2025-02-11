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
