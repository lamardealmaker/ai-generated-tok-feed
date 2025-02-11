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
