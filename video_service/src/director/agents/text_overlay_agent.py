from typing import Dict, Any, List
from ..core.base_agent import BaseAgent

class TextOverlayAgent(BaseAgent):
    """Agent for generating and styling text overlays in TikTok-style videos."""
    
    def __init__(self):
        self.style_presets = {
            "modern": {
                "title": {
                    "font": "Helvetica Neue",
                    "weight": "bold",
                    "size": "large",
                    "color": "#FFFFFF",
                    "shadow": True,
                    "animation": "slide_up"
                },
                "price": {
                    "font": "Helvetica Neue",
                    "weight": "bold",
                    "size": "xlarge",
                    "color": "#00FF88",
                    "shadow": True,
                    "animation": "pop"
                },
                "features": {
                    "font": "Helvetica Neue",
                    "weight": "medium",
                    "size": "medium",
                    "color": "#FFFFFF",
                    "shadow": True,
                    "animation": "fade_in"
                }
            },
            "luxury": {
                "title": {
                    "font": "Didot",
                    "weight": "bold",
                    "size": "large",
                    "color": "#FFFFFF",
                    "shadow": False,
                    "animation": "elegant_fade"
                },
                "price": {
                    "font": "Didot",
                    "weight": "bold",
                    "size": "xlarge",
                    "color": "#FFD700",
                    "shadow": False,
                    "animation": "elegant_reveal"
                },
                "features": {
                    "font": "Didot",
                    "weight": "medium",
                    "size": "medium",
                    "color": "#FFFFFF",
                    "shadow": False,
                    "animation": "smooth_fade"
                }
            },
            "minimal": {
                "title": {
                    "font": "SF Pro Display",
                    "weight": "regular",
                    "size": "large",
                    "color": "#FFFFFF",
                    "shadow": False,
                    "animation": "minimal_fade"
                },
                "price": {
                    "font": "SF Pro Display",
                    "weight": "light",
                    "size": "xlarge",
                    "color": "#FFFFFF",
                    "shadow": False,
                    "animation": "minimal_slide"
                },
                "features": {
                    "font": "SF Pro Display",
                    "weight": "light",
                    "size": "medium",
                    "color": "#FFFFFF",
                    "shadow": False,
                    "animation": "minimal_reveal"
                }
            }
        }
        
        self.animations = {
            "slide_up": {"duration": 0.5, "ease": "cubic-bezier(0.4, 0, 0.2, 1)"},
            "pop": {"duration": 0.3, "ease": "cubic-bezier(0.34, 1.56, 0.64, 1)"},
            "fade_in": {"duration": 0.4, "ease": "cubic-bezier(0.4, 0, 0.2, 1)"},
            "elegant_fade": {"duration": 0.8, "ease": "cubic-bezier(0.4, 0, 0.2, 1)"},
            "elegant_reveal": {"duration": 1.0, "ease": "cubic-bezier(0.4, 0, 0.2, 1)"},
            "smooth_fade": {"duration": 0.6, "ease": "cubic-bezier(0.4, 0, 0.2, 1)"},
            "minimal_fade": {"duration": 0.4, "ease": "linear"},
            "minimal_slide": {"duration": 0.5, "ease": "cubic-bezier(0.4, 0, 0.2, 1)"},
            "minimal_reveal": {"duration": 0.3, "ease": "cubic-bezier(0.4, 0, 0.2, 1)"}
        }
    
    def get_style_for_template(self, template_name: str) -> Dict[str, Any]:
        """Get text style configuration for a template."""
        style_name = template_name.split('_')[0]  # e.g., 'modern' from 'modern_rapid_standard'
        return self.style_presets.get(style_name, self.style_presets["modern"])
    
    def generate_overlay_config(
        self,
        template_name: str,
        text_content: Dict[str, str],
        timing: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Generate text overlay configuration for a video.
        
        Args:
            template_name: Name of the template being used
            text_content: Dictionary of text content (title, price, features, etc.)
            timing: Dictionary of timing information for each text element
            
        Returns:
            List of overlay configurations with styling and animation
        """
        style = self.get_style_for_template(template_name)
        overlays = []
        
        for text_type, content in text_content.items():
            if text_type in style:
                text_style = style[text_type]
                animation = self.animations[text_style["animation"]]
                
                overlay = {
                    "content": content,
                    "type": text_type,
                    "style": {
                        "font_family": text_style["font"],
                        "font_weight": text_style["weight"],
                        "font_size": text_style["size"],
                        "color": text_style["color"],
                        "text_shadow": text_style["shadow"]
                    },
                    "animation": {
                        "type": text_style["animation"],
                        "duration": animation["duration"],
                        "easing": animation["ease"]
                    },
                    "timing": {
                        "start": timing.get(text_type, 0),
                        "duration": timing.get(f"{text_type}_duration", 3.0)
                    }
                }
                overlays.append(overlay)
        
        return overlays
