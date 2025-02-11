from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import random

class VideoTemplate:
    """Template for video structure and styling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.name = config.get('name', 'default')
        self.duration = config.get('duration', 60)
        self.segments = config.get('segments', [])
        self.transitions = config.get('transitions', [])
        self.effects = config.get('effects', [])
        self.text_styles = config.get('text_styles', {})
        self.music_style = config.get('music_style', {})
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            'name': self.name,
            'duration': self.duration,
            'segments': self.segments,
            'transitions': self.transitions,
            'effects': self.effects,
            'text_styles': self.text_styles,
            'music_style': self.music_style
        }

class TemplateLibrary:
    """Library of video templates."""
    
    def __init__(self, template_path: Optional[str] = None):
        self.templates = self._load_templates(template_path)
        
    def _load_templates(self, template_path: Optional[str]) -> Dict[str, VideoTemplate]:
        """Load templates from JSON file."""
        if not template_path:
            return self._default_templates()
            
        template_path = Path(template_path)
        if template_path.exists():
            with open(template_path, 'r') as f:
                templates_data = json.load(f)
                return {
                    name: VideoTemplate(data)
                    for name, data in templates_data.items()
                }
                
        return self._default_templates()
        
    def _default_templates(self) -> Dict[str, VideoTemplate]:
        """Default video templates with extensive variations."""
        templates = {}
        
        # Base styles
        styles = {
            'modern': {
                'transitions': ['slide_left', 'slide_right', 'zoom_in', 'whip_pan'],
                'effects': ['color_enhance', 'sharpness'],
                'text_animations': ['pop', 'slide'],
                'music_genres': ['electronic', 'pop', 'upbeat'],
                'color_schemes': ['vibrant', 'neon', 'pastel']
            },
            'luxury': {
                'transitions': ['dissolve', 'zoom_out', 'fade'],
                'effects': ['cinematic_bars', 'film_grain', 'vignette'],
                'text_animations': ['fade', 'elegant_slide'],
                'music_genres': ['ambient', 'classical', 'jazz'],
                'color_schemes': ['gold', 'monochrome', 'earth']
            },
            'minimal': {
                'transitions': ['cut', 'dissolve', 'slide_minimal'],
                'effects': ['subtle_sharpen', 'light_contrast'],
                'text_animations': ['fade', 'minimal_pop'],
                'music_genres': ['minimal', 'acoustic', 'soft'],
                'color_schemes': ['clean', 'subtle', 'neutral']
            },
            'urban': {
                'transitions': ['glitch', 'swipe', 'bounce'],
                'effects': ['rgb_split', 'grain', 'contrast'],
                'text_animations': ['glitch', 'bounce', 'swipe'],
                'music_genres': ['hip_hop', 'electronic', 'urban'],
                'color_schemes': ['street', 'neon', 'bold']
            },
            'cinematic': {
                'transitions': ['cinematic_fade', 'letterbox', 'pan'],
                'effects': ['anamorphic', 'film_look', 'lens_blur'],
                'text_animations': ['cinematic_fade', 'smooth_reveal'],
                'music_genres': ['orchestral', 'soundtrack', 'epic'],
                'color_schemes': ['film', 'dramatic', 'moody']
            }
        }
        
        # Pacing variations
        pacings = {
            'rapid': {'segment_multiplier': 0.7, 'transition_duration': 0.3},
            'balanced': {'segment_multiplier': 1.0, 'transition_duration': 0.7},
            'relaxed': {'segment_multiplier': 1.3, 'transition_duration': 1.0}
        }
        
        # Segment structures
        structures = {
            'standard': [
                {'type': 'intro', 'duration': 5},
                {'type': 'features', 'duration': 30},
                {'type': 'highlights', 'duration': 20},
                {'type': 'outro', 'duration': 5}
            ],
            'story': [
                {'type': 'hook', 'duration': 3},
                {'type': 'context', 'duration': 12},
                {'type': 'reveal', 'duration': 35},
                {'type': 'close', 'duration': 10}
            ],
            'showcase': [
                {'type': 'aerial', 'duration': 10},
                {'type': 'exterior', 'duration': 15},
                {'type': 'interior', 'duration': 25},
                {'type': 'details', 'duration': 10}
            ],
            'feature_focus': [
                {'type': 'intro', 'duration': 5},
                {'type': 'feature_1', 'duration': 20},
                {'type': 'feature_2', 'duration': 20},
                {'type': 'feature_3', 'duration': 15}
            ],
            'quick': [
                {'type': 'hook', 'duration': 3},
                {'type': 'highlights', 'duration': 42},
                {'type': 'call_to_action', 'duration': 15}
            ]
        }
        
        # Generate templates by combining styles, pacings, and structures
        for style_name, style in styles.items():
            for pacing_name, pacing in pacings.items():
                for structure_name, base_segments in structures.items():
                    # Create variations of segments with pacing
                    segments = []
                    for segment in base_segments:
                        segments.append({
                            'type': segment['type'],
                            'duration': segment['duration'] * pacing['segment_multiplier']
                        })
                    
                    # Create template variations with different combinations
                    for color_scheme in style['color_schemes']:
                        # Ensure consistent naming format
                        safe_structure_name = structure_name.replace('_', '')
                        template_name = f'{style_name}_{pacing_name}_{safe_structure_name}_{color_scheme}'
                        
                        # Select random variations while maintaining style consistency
                        import random
                        transitions = [
                            {'type': t, 'duration': pacing['transition_duration']}
                            for t in random.sample(style['transitions'], 
                                                min(3, len(style['transitions'])))
                        ]
                        
                        effects = [
                            {'name': e, 'intensity': random.uniform(0.3, 0.8)}
                            for e in random.sample(style['effects'],
                                                min(2, len(style['effects'])))
                        ]
                        
                        templates[template_name] = VideoTemplate({
                            'name': template_name,
                            'duration': 60,
                            'segments': segments,
                            'transitions': transitions,
                            'effects': effects,
                            'text_styles': {
                                'title': {
                                    'font_scale': 1.2,
                                    'animation': random.choice(style['text_animations'])
                                },
                                'feature': {
                                    'font_scale': 1.0,
                                    'animation': random.choice(style['text_animations'])
                                }
                            },
                            'music_style': {
                                'genre': random.choice(style['music_genres']),
                                'tempo': pacing_name
                            }
                        })
        
        return templates
        
    def get_template(self, name: str) -> VideoTemplate:
        """Get template by name."""
        if name not in self.templates:
            raise ValueError(f"Unknown template: {name}")
        return self.templates[name]
        
    def list_templates(self) -> List[str]:
        """List available templates."""
        return list(self.templates.keys())
        
    def get_template_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a template."""
        template = self.get_template(name)
        return template.to_dict()
        
    def save_templates(self, path: str) -> None:
        """Save templates to JSON file."""
        templates_data = {
            name: template.to_dict()
            for name, template in self.templates.items()
        }
        
        with open(path, 'w') as f:
            json.dump(templates_data, f, indent=2)
            
    def add_template(self, name: str, config: Dict[str, Any]) -> None:
        """Add new template to library."""
        self.templates[name] = VideoTemplate(config)
