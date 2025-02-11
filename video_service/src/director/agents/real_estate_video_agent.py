from typing import Dict, Any, List
from ..core.base_agent import BaseAgent
from .script_agent import ScriptAgent
from .pacing_agent import PacingAgent
from .effect_agent import EffectAgent

class RealEstateVideoAgent(BaseAgent):
    """Agent for property-specific video generation."""
    
    def __init__(self):
        super().__init__()
        self.script_agent = ScriptAgent()
        self.pacing_agent = PacingAgent()
        self.effect_agent = EffectAgent()
        
    def analyze_property(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze property data to determine key selling points and features.
        
        Args:
            property_data: Property listing data
            
        Returns:
            Analysis results including key features, target audience, etc.
        """
        analysis = {
            "key_features": [],
            "target_audience": [],
            "price_segment": "",
            "unique_selling_points": []
        }
        
        # Analyze price segment
        price = property_data.get("price", 0)
        if price > 1000000:
            analysis["price_segment"] = "luxury"
        elif price > 500000:
            analysis["price_segment"] = "premium"
        else:
            analysis["price_segment"] = "standard"
            
        # Extract key features
        features = property_data.get("features", [])
        amenities = property_data.get("amenities", [])
        all_features = features + amenities
        
        # Prioritize features based on price segment
        if analysis["price_segment"] == "luxury":
            priority_features = ["pool", "spa", "view", "smart_home"]
        else:
            priority_features = ["location", "space", "updates", "parking"]
            
        for feature in all_features:
            if any(p in feature.lower() for p in priority_features):
                analysis["key_features"].append(feature)
                
        # Determine target audience
        if property_data.get("bedrooms", 0) >= 4:
            analysis["target_audience"].append("families")
        if "modern" in " ".join(features).lower():
            analysis["target_audience"].append("young_professionals")
        if "retirement" in " ".join(features).lower():
            analysis["target_audience"].append("retirees")
            
        # Identify unique selling points
        if property_data.get("year_built", 0) > 2020:
            analysis["unique_selling_points"].append("new_construction")
        if property_data.get("lot_size", 0) > 10000:
            analysis["unique_selling_points"].append("large_lot")
        if "view" in " ".join(features).lower():
            analysis["unique_selling_points"].append("scenic_view")
            
        return analysis
        
    def generate_video_plan(
        self,
        property_data: Dict[str, Any],
        style: str = "modern",
        duration: float = 30.0
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive video plan for the property.
        
        Args:
            property_data: Property listing data
            style: Video style preference
            duration: Target video duration
            
        Returns:
            Video plan including script, pacing, and effects
        """
        # Analyze property
        analysis = self.analyze_property(property_data)
        
        # Generate script
        script = self.script_agent.generate_script(
            property_data=property_data,
            style=style,
            duration=duration
        )
        
        # Optimize pacing
        content_analysis = self.pacing_agent.analyze_content(script["segments"])
        pacing = self.pacing_agent.get_optimal_pacing(
            content_analysis=content_analysis,
            target_duration=duration,
            style=style
        )
        
        # Apply pacing
        segments = self.pacing_agent.apply_pacing(
            segments=script["segments"],
            pacing=pacing,
            target_duration=duration
        )
        
        # Generate effects
        effects = []
        for segment in segments:
            segment_effects = self.effect_agent.generate_effect_combination(
                style=style,
                content_type=segment.get("content_type", "interior")
            )
            effects.append(segment_effects)
            
        return {
            "analysis": analysis,
            "script": script,
            "segments": segments,
            "effects": effects,
            "style": style,
            "duration": duration
        }
