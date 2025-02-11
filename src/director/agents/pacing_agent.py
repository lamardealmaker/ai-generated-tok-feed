from typing import Dict, Any, List, Optional
import numpy as np
from ..core.base_agent import BaseAgent

class PacingAgent(BaseAgent):
    """Agent for adapting video pacing based on content and engagement."""
    
    def __init__(self):
        self.pacing_profiles = {
            "rapid": {
                "segment_duration": 2.5,
                "transition_duration": 0.3,
                "text_duration": 2.0,
                "music_bpm_range": (120, 140)
            },
            "balanced": {
                "segment_duration": 3.5,
                "transition_duration": 0.5,
                "text_duration": 3.0,
                "music_bpm_range": (100, 120)
            },
            "relaxed": {
                "segment_duration": 4.5,
                "transition_duration": 0.7,
                "text_duration": 4.0,
                "music_bpm_range": (80, 100)
            }
        }
        
        self.content_weights = {
            "exterior": 1.0,
            "interior": 1.2,
            "feature": 1.5,
            "amenity": 1.3,
            "view": 1.4
        }
        
    def analyze_content(
        self,
        segments: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Analyze content to determine optimal pacing.
        
        Args:
            segments: List of video segments with content type
            
        Returns:
            Dictionary with content analysis metrics
        """
        analysis = {
            "complexity": 0.0,
            "visual_interest": 0.0,
            "information_density": 0.0
        }
        
        for segment in segments:
            content_type = segment.get("content_type", "interior")
            weight = self.content_weights.get(content_type, 1.0)
            
            if "text" in segment:
                analysis["complexity"] += len(segment["text"].split()) * 0.1 * weight
            
            if "features" in segment:
                analysis["visual_interest"] += len(segment["features"]) * 0.2 * weight
                
            if "data_points" in segment:
                analysis["information_density"] += len(segment["data_points"]) * 0.15 * weight
                
        total_segments = len(segments)
        for key in analysis:
            analysis[key] /= total_segments
            
        return analysis
        
    def get_optimal_pacing(
        self,
        content_analysis: Dict[str, float],
        target_duration: float,
        style: str = "modern"
    ) -> Dict[str, Any]:
        """
        Determine optimal pacing based on content analysis.
        
        Args:
            content_analysis: Content analysis metrics
            target_duration: Target video duration
            style: Video style
            
        Returns:
            Pacing configuration
        """
        complexity_score = content_analysis["complexity"]
        if complexity_score > 0.7:
            base_profile = "relaxed"
        elif complexity_score < 0.3:
            base_profile = "rapid"
        else:
            base_profile = "balanced"
            
        profile = self.pacing_profiles[base_profile].copy()
        
        if style == "luxury":
            profile["segment_duration"] *= 1.2
            profile["transition_duration"] *= 1.3
        elif style == "minimal":
            profile["segment_duration"] *= 0.9
            profile["transition_duration"] *= 0.8
            
        visual_score = content_analysis["visual_interest"]
        info_score = content_analysis["information_density"]
        
        profile["segment_duration"] *= 1 + (info_score - 0.5) * 0.3
        profile["transition_duration"] *= 1 + (visual_score - 0.5) * 0.2
        
        return profile
        
    def apply_pacing(
        self,
        segments: List[Dict[str, Any]],
        pacing: Dict[str, Any],
        target_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Apply pacing configuration to video segments.
        
        Args:
            segments: List of video segments
            pacing: Pacing configuration
            target_duration: Target video duration
            
        Returns:
            List of segments with adjusted timing
        """
        adjusted_segments = []
        current_time = 0.0
        
        base_durations = []
        for segment in segments:
            content_type = segment.get("content_type", "interior")
            weight = self.content_weights.get(content_type, 1.0)
            duration = pacing["segment_duration"] * weight
            base_durations.append(duration)
            
        total_duration = sum(base_durations)
        scale_factor = (target_duration - (len(segments) - 1) * pacing["transition_duration"]) / total_duration
        
        for segment, base_duration in zip(segments, base_durations):
            adjusted_segment = segment.copy()
            adjusted_segment["duration"] = base_duration * scale_factor
            adjusted_segment["start_time"] = current_time
            
            if len(adjusted_segments) < len(segments) - 1:
                adjusted_segment["transition_duration"] = pacing["transition_duration"]
                current_time += adjusted_segment["duration"] + pacing["transition_duration"]
            else:
                current_time += adjusted_segment["duration"]
                
            adjusted_segments.append(adjusted_segment)
            
        return adjusted_segments
        
    def optimize_engagement(
        self,
        segments: List[Dict[str, Any]],
        engagement_data: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Optimize pacing based on engagement data.
        
        Args:
            segments: List of video segments
            engagement_data: Optional engagement metrics
            
        Returns:
            Optimized segments
        """
        if not engagement_data:
            return segments
            
        optimized = []
        total_duration = sum(s["duration"] for s in segments)
        
        for segment in segments:
            adjusted = segment.copy()
            content_type = segment.get("content_type", "interior")
            
            if content_type in engagement_data:
                engagement_score = engagement_data[content_type]
                adjustment = np.clip(engagement_score - 0.5, -0.25, 0.25)
                adjusted["duration"] *= (1 + adjustment)
                
            optimized.append(adjusted)
            
        current_total = sum(s["duration"] for s in optimized)
        scale_factor = total_duration / current_total
        
        current_time = 0.0
        for segment in optimized:
            segment["duration"] *= scale_factor
            segment["start_time"] = current_time
            current_time += segment["duration"]
            
        return optimized