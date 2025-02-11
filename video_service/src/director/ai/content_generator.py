from typing import Dict, Any, List, Optional
from .llm_manager import LLMManager

class ContentGenerator:
    """
    Generates video content using LLM-driven decisions.
    """
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        
    async def generate_video_content(self,
                                   listing_data: Dict[str, Any],
                                   duration: int = 60) -> Dict[str, Any]:
        """
        Generate complete video content including script, style, and segments.
        
        Args:
            listing_data: Property listing information
            duration: Video duration in seconds
            
        Returns:
            Dictionary containing all content elements:
            - script: Generated script
            - style: Selected style
            - segments: Timed content segments
            - directions: Production directions
        """
        # Select appropriate style based on property details
        style_preset = await self.llm_manager.select_video_style(listing_data)
        
        # Generate script with selected style
        script_data = await self.llm_manager.generate_video_script(
            listing_data,
            style_preset,
            duration
        )
        
        # Enhance the content with additional elements
        return {
            'script': script_data['script'],
            'style': {
                'preset': style_preset,
                'settings': self.llm_manager.get_style_preset(style_preset)
            },
            'segments': script_data['segments'],
            'directions': {
                'style': script_data['style_directions'],
                'transitions': self._generate_transitions(script_data['segments']),
                'effects': self._generate_effects(style_preset),
                'music': self._select_music_style(style_preset)
            }
        }
        
    def _generate_transitions(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate transition effects between segments."""
        transitions = []
        for i in range(len(segments) - 1):
            transitions.append({
                'time': segments[i+1]['time'],
                'type': 'smooth_fade',  # Default transition
                'duration': 0.5
            })
        return transitions
        
    def _generate_effects(self, style_preset: str) -> List[Dict[str, Any]]:
        """Generate video effects based on style."""
        effects = []
        if style_preset == 'energetic':
            effects = [
                {'type': 'zoom', 'intensity': 'medium'},
                {'type': 'text_pop', 'intensity': 'high'},
                {'type': 'color_enhance', 'intensity': 'medium'}
            ]
        elif style_preset == 'luxury':
            effects = [
                {'type': 'smooth_pan', 'intensity': 'low'},
                {'type': 'cinematic_fade', 'intensity': 'medium'},
                {'type': 'depth_blur', 'intensity': 'low'}
            ]
        return effects
        
    def _select_music_style(self, style_preset: str) -> Dict[str, Any]:
        """Select music style based on video style."""
        music_styles = {
            'energetic': {
                'genre': 'upbeat_electronic',
                'tempo': 'fast',
                'intensity': 'high'
            },
            'professional': {
                'genre': 'corporate',
                'tempo': 'medium',
                'intensity': 'low'
            },
            'luxury': {
                'genre': 'ambient',
                'tempo': 'slow',
                'intensity': 'medium'
            }
        }
        return music_styles.get(style_preset, music_styles['energetic'])
