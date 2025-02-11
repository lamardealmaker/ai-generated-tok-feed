from typing import Dict, Any, List, Optional
import json
import random
from ..core.base_agent import BaseAgent

class ScriptAgent(BaseAgent):
    """Agent for generating dynamic video scripts using LLMs."""
    
    def __init__(self):
        self.script_templates = {
            "modern": {
                "hooks": [
                    "WAIT UNTIL YOU SEE THIS {location} DREAM HOME! 🏠✨",
                    "THIS {style} HOME IN {location} WILL BLOW YOUR MIND! 🔥",
                    "INSANE {price_range} PROPERTY ALERT! 🚨 {location} GEM! 💎"
                ],
                "features": [
                    "OBSESSED with this {feature}! 😍",
                    "THE MOST INCREDIBLE {feature} for {use_case}! 🤯",
                    "YOU WON'T BELIEVE this {feature}! ✨"
                ],
                "closings": [
                    "RUN don't walk! DM now! 🏃‍♂️",
                    "This won't last! Save this! ⭐️",
                    "Tag someone who needs to see this! 🔥"
                ]
            },
            "luxury": {
                "hooks": [
                    "EXCLUSIVE FIRST LOOK: ${price_range} {location} MANSION 👑",
                    "INSIDE THE MOST STUNNING {location} ESTATE YET! ✨",
                    "THIS {style} MASTERPIECE WILL LEAVE YOU SPEECHLESS! 🏰"
                ],
                "features": [
                    "LOOK AT THIS INCREDIBLE {feature}! Perfect for {use_case} 🤩",
                    "THE MOST INSANE {feature} you'll ever see! 🔥",
                    "I CAN'T BELIEVE this {feature}! Absolutely unreal! ✨"
                ],
                "closings": [
                    "Serious inquiries only - DM for private tour 🔑",
                    "Save this dream home! You won't regret it! ⭐️",
                    "Like & Share if this is your dream home! 🏠"
                ]
            },
            "minimal": {
                "hooks": [
                    "THIS {location} GEM IS EVERYTHING! ✨",
                    "PERFECTLY DESIGNED {style} HOME! 🏠",
                    "THE MOST AESTHETIC {location} FIND! 🎯"
                ],
                "features": [
                    "OBSESSED with this {feature} for {use_case}! 😍",
                    "Clean lines define this {feature}",
                    "Smart design in the {feature}"
                ],
                "closings": [
                    "Schedule your visit",
                    "Available for viewing",
                    "Connect for details"
                ]
            }
        }
        
    def _analyze_property(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze property data to determine key selling points."""
        analysis = {
            "price_range": self._get_price_range(property_data["price"]),
            "key_features": [],
            "unique_selling_points": [],
            "target_audience": [],
            "style_keywords": []
        }
        
        if property_data["price"] > 1000000:
            analysis["target_audience"].extend(["luxury-buyers", "investors"])
            analysis["style_keywords"].extend(["luxury", "premium", "exclusive"])
        else:
            analysis["target_audience"].extend(["homeowners", "families"])
            analysis["style_keywords"].extend(["modern", "comfortable", "practical"])
            
        if property_data.get("bedrooms", 0) >= 4:
            analysis["key_features"].append(("spacious layout", "family living"))
            analysis["unique_selling_points"].append("Large family home")
            
        if property_data.get("bathrooms", 0) >= 2.5:
            analysis["key_features"].append(("multiple bathrooms", "convenience"))
            
        sqft = property_data.get("sqft", 0)
        if sqft > 3000:
            analysis["key_features"].append(("expansive space", "luxury living"))
        elif sqft > 2000:
            analysis["key_features"].append(("generous space", "comfortable living"))
            
        if property_data.get("lot_size", 0) > 10000:
            analysis["key_features"].append(("large lot", "outdoor living"))
            
        for feature in property_data.get("features", []):
            feature_lower = feature.lower()
            if "kitchen" in feature_lower:
                analysis["key_features"].append((feature, "modern living"))
            elif "floor" in feature_lower:
                analysis["key_features"].append((feature, "design"))
            elif "yard" in feature_lower or "outdoor" in feature_lower:
                analysis["key_features"].append((feature, "outdoor living"))
            else:
                analysis["key_features"].append((feature, "lifestyle"))
            
        return analysis
        
    def _get_price_range(self, price: float) -> str:
        if price < 300000:
            return "starter"
        elif price < 600000:
            return "mid-range"
        elif price < 1000000:
            return "upscale"
        else:
            return "luxury"
            
    MIN_DURATION = 10.0
    MAX_DURATION = 60.0
    
    def generate_script(
        self,
        property_data: Dict[str, Any],
        style: str = "modern",
        duration: float = 30.0,
        target_segments: int = 5
    ) -> Dict[str, Any]:
        if duration < self.MIN_DURATION:
            duration = self.MIN_DURATION
        elif duration > self.MAX_DURATION:
            duration = self.MAX_DURATION
            
        if duration <= 15.0:
            target_segments = min(target_segments, 3)
        elif duration >= 45.0:
            target_segments = 7
            if duration >= 55.0:
                target_segments = 8
            
        analysis = self._analyze_property(property_data)
        templates = self.script_templates[style]
        
        segment_duration = duration / target_segments
        current_time = 0.0
        
        script = {
            "style": style,
            "duration": duration,
            "segments": [],
            "metadata": {
                "price_range": analysis["price_range"],
                "target_audience": analysis["target_audience"],
                "key_features": analysis["key_features"]
            }
        }
        
        hook_text = random.choice(templates["hooks"]).format(
            location=property_data["location"],
            style=property_data.get("style", "modern"),
            price_range=analysis["price_range"]
        )
        script["segments"].append({
            "type": "hook",
            "text": hook_text,
            "start_time": current_time,
            "duration": segment_duration,
            "focus": "exterior"
        })
        current_time += segment_duration
        
        feature_count = target_segments - 2
        features = analysis["key_features"]
        while len(features) < feature_count:
            features.extend(analysis["key_features"])
        features = features[:feature_count]
        
        for feature, use_case in features:
            feature_text = random.choice(templates["features"]).format(
                feature=feature,
                use_case=use_case
            )
            script["segments"].append({
                "type": "feature",
                "text": feature_text,
                "start_time": current_time,
                "duration": segment_duration,
                "focus": feature.replace(" ", "_")
            })
            current_time += segment_duration
            
        closing_text = random.choice(templates["closings"])
        script["segments"].append({
            "type": "closing",
            "text": closing_text,
            "start_time": current_time,
            "duration": segment_duration,
            "focus": "exterior"
        })
        
        return script