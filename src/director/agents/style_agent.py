from typing import Dict, Any, List, Optional
import json
import random
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
                },
                "tiktok": {
                    "video": {
                        "transitions": ["quick_slide", "jump_cut", "glitch"],
                        "effects": ["pop", "shake", "flash"],
                        "pacing": "rapid",
                        "color_grade": "vibrant"
                    },
                    "text": {
                        "font": "Impact",
                        "animation": "pop",
                        "color_scheme": ["#FF0000", "#FFFFFF"],
                        "placement": "dynamic"
                    },
                    "audio": {
                        "music_style": "upbeat_electronic",
                        "voice_style": "energetic",
                        "sound_effects": True
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
            
        if random.random() < variation_strength:
            base_config['video']['effects'] = random.sample(
                base_config['video']['effects'],
                k=max(1, len(base_config['video']['effects']) - 1)
            )
            
        if random.random() < variation_strength:
            base_config['video']['color_grade'] = random.choice(
                self.config['variations']['color_grades']
            )
            
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
        if property_value > 1000000 or 'luxury' in target_market.lower():
            return 'luxury'
            
        if any(term in target_market.lower() for term in ['young', 'urban', 'professional']):
            return 'energetic'
            
        return 'professional'
        
    def get_style_elements(self,
                         style_name: str,
                         element_type: str) -> Dict[str, Any]:
        """Get specific elements of a style."""
        style = self.get_style(style_name)
        return style.get(element_type, {})