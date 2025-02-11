from typing import Dict, Any, List
import random
from ..core.base_agent import BaseAgent

class StyleVariationAgent(BaseAgent):
    """Agent for generating TikTok-style variations."""
    
    def __init__(self):
        super().__init__()
        self.style_templates = {
            "energetic": {
                "transitions": ["quick_cut", "whip_pan", "zoom_blur"],
                "effects": ["shake", "glitch", "color_pulse"],
                "music_style": "upbeat",
                "text_style": "bold",
                "duration_range": (15, 30)
            },
            "cinematic": {
                "transitions": ["smooth_fade", "dolly_zoom", "parallax"],
                "effects": ["film_grain", "letterbox", "color_grade"],
                "music_style": "dramatic",
                "text_style": "elegant",
                "duration_range": (30, 60)
            },
            "minimal": {
                "transitions": ["simple_cut", "fade", "slide"],
                "effects": ["subtle_zoom", "soft_light", "clean"],
                "music_style": "ambient",
                "text_style": "modern",
                "duration_range": (20, 45)
            }
        }
        
    def generate_variation(
        self,
        base_style: str,
        target_audience: List[str],
        duration: float
    ) -> Dict[str, Any]:
        """
        Generate a TikTok-style variation.
        
        Args:
            base_style: Base style to build from
            target_audience: List of target audience segments
            duration: Target duration
            
        Returns:
            Style variation configuration
        """
        if base_style not in self.style_templates:
            raise ValueError(f"Unknown base style: {base_style}")
            
        base_template = self.style_templates[base_style]
        
        # Adjust style based on target audience
        variation = {
            "style": base_style,
            "transitions": list(base_template["transitions"]),
            "effects": list(base_template["effects"]),
            "music_style": base_template["music_style"],
            "text_style": base_template["text_style"]
        }
        
        # Audience-specific adjustments
        if "young_professionals" in target_audience:
            variation["effects"].extend(["dynamic_text", "emoji_pop"])
            variation["music_style"] = "trendy"
        elif "families" in target_audience:
            variation["effects"].extend(["warm_glow", "family_friendly"])
            variation["music_style"] = "uplifting"
        elif "luxury_buyers" in target_audience:
            variation["effects"].extend(["premium_grade", "elegant_text"])
            variation["music_style"] = "sophisticated"
            
        # Duration-based adjustments
        if duration <= 15:
            # Short-form optimizations
            variation["transitions"] = [t for t in variation["transitions"] if "quick" in t or "fast" in t]
            variation["effects"].append("attention_grab")
        elif duration >= 45:
            # Long-form enhancements
            variation["transitions"].extend(["story_transition", "mood_shift"])
            variation["effects"].append("narrative_enhance")
            
        return variation
        
    def optimize_engagement(
        self,
        variation: Dict[str, Any],
        platform_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Optimize variation based on platform engagement metrics.
        
        Args:
            variation: Current style variation
            platform_metrics: Engagement metrics from the platform
            
        Returns:
            Optimized variation
        """
        optimized = dict(variation)
        
        # Adjust based on view duration metrics
        if platform_metrics.get("avg_view_duration", 0) < 0.5:
            # Users dropping off quickly, make it more engaging
            optimized["effects"].extend(["pattern_interrupt", "hook_enhance"])
            optimized["transitions"] = [t for t in optimized["transitions"] if "slow" not in t]
            
        # Adjust based on engagement rate
        if platform_metrics.get("engagement_rate", 0) < 0.1:
            # Low engagement, add more interactive elements
            optimized["effects"].extend(["text_callout", "highlight_zoom"])
            
        # Adjust based on completion rate
        if platform_metrics.get("completion_rate", 0) < 0.7:
            # Users not finishing, optimize for retention
            optimized["effects"].append("retention_hook")
            optimized["transitions"].append("curiosity_gap")
            
        return optimized
        
    def get_trending_elements(self) -> Dict[str, List[str]]:
        """Get currently trending style elements."""
        # In a real implementation, this would fetch from a trends API
        return {
            "transitions": ["creative_zoom", "glitch_transition"],
            "effects": ["viral_text_style", "trending_filter"],
            "music_styles": ["viral_beats", "trending_sounds"],
            "hooks": ["question_hook", "statistic_hook"]
        }
