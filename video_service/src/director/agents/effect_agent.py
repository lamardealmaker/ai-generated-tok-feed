from typing import Dict, Any, List, Optional
import random
from ..core.base_agent import BaseAgent

class EffectAgent(BaseAgent):
    """Agent for generating and combining video effects."""
    
    def __init__(self):
        self.effect_library = {
            "color": {
                "modern": [
                    {"name": "vibrant", "contrast": 1.2, "saturation": 1.3, "brightness": 1.1},
                    {"name": "cool", "contrast": 1.1, "saturation": 0.9, "temperature": -10},
                    {"name": "warm", "contrast": 1.1, "saturation": 1.1, "temperature": 10}
                ],
                "luxury": [
                    {"name": "cinematic", "contrast": 1.3, "saturation": 0.9, "shadows": 0.8},
                    {"name": "rich", "contrast": 1.2, "saturation": 1.1, "highlights": 0.9},
                    {"name": "moody", "contrast": 1.4, "saturation": 0.8, "shadows": 0.7}
                ],
                "minimal": [
                    {"name": "clean", "contrast": 1.1, "saturation": 0.9, "brightness": 1.05},
                    {"name": "bright", "contrast": 1.0, "saturation": 0.8, "brightness": 1.2},
                    {"name": "pure", "contrast": 1.05, "saturation": 0.85, "highlights": 1.1}
                ]
            },
            "overlay": {
                "modern": [
                    {"name": "light_leak", "opacity": 0.2, "blend": "screen"},
                    {"name": "grain", "opacity": 0.1, "blend": "overlay"},
                    {"name": "vignette", "opacity": 0.15, "blend": "multiply"}
                ],
                "luxury": [
                    {"name": "film_grain", "opacity": 0.08, "blend": "overlay"},
                    {"name": "soft_light", "opacity": 0.15, "blend": "soft_light"},
                    {"name": "cinematic_bars", "opacity": 1.0, "blend": "normal"}
                ],
                "minimal": [
                    {"name": "subtle_grain", "opacity": 0.05, "blend": "overlay"},
                    {"name": "soft_vignette", "opacity": 0.1, "blend": "multiply"},
                    {"name": "clean_overlay", "opacity": 0.08, "blend": "screen"}
                ]
            },
            "filter": {
                "modern": [
                    {"name": "sharpen", "amount": 0.3, "radius": 1.0},
                    {"name": "clarity", "amount": 0.4, "radius": 1.5},
                    {"name": "dehaze", "amount": 0.2}
                ],
                "luxury": [
                    {"name": "bloom", "amount": 0.3, "radius": 2.0},
                    {"name": "glow", "amount": 0.25, "radius": 1.8},
                    {"name": "soft_focus", "amount": 0.2, "radius": 1.5}
                ],
                "minimal": [
                    {"name": "clarity", "amount": 0.2, "radius": 1.0},
                    {"name": "sharpen", "amount": 0.15, "radius": 0.8},
                    {"name": "clean", "amount": 0.1, "radius": 1.0}
                ]
            }
        }
        
        self.combination_rules = {
            "modern": {
                "max_overlays": 2,
                "required_effects": ["color", "filter"],
                "optional_effects": ["overlay"],
                "compatibility": {
                    "light_leak": ["grain"],
                    "vignette": ["grain", "light_leak"]
                }
            },
            "luxury": {
                "max_overlays": 2,
                "required_effects": ["color", "filter", "overlay"],
                "optional_effects": [],
                "compatibility": {
                    "film_grain": ["soft_light"],
                    "cinematic_bars": ["film_grain", "soft_light"]
                }
            },
            "minimal": {
                "max_overlays": 1,
                "required_effects": ["color"],
                "optional_effects": ["filter", "overlay"],
                "compatibility": {
                    "subtle_grain": ["soft_vignette"],
                    "clean_overlay": ["subtle_grain"]
                }
            }
        }
        
    def generate_effect_combination(
        self,
        style: str,
        content_type: str,
        intensity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate a compatible combination of effects.
        
        Args:
            style: Video style
            content_type: Type of content (exterior, interior, etc.)
            intensity: Effect intensity multiplier
            
        Returns:
            Dictionary of combined effects
        """
        rules = self.combination_rules[style]
        effects = {
            "color": None,
            "filter": None,
            "overlay": []
        }
        
        # Apply required effects
        for effect_type in rules["required_effects"]:
            if effect_type == "overlay":
                continue  # Handled separately
            effects[effect_type] = self._select_effect(
                effect_type, style, intensity
            )
            
        # Apply optional effects
        for effect_type in rules["optional_effects"]:
            if random.random() < 0.7:  # 70% chance to apply optional effect
                if effect_type == "overlay":
                    continue  # Handled separately
                effects[effect_type] = self._select_effect(
                    effect_type, style, intensity
                )
                
        # Handle overlays with compatibility rules
        available_overlays = self.effect_library["overlay"][style]
        selected_overlays = []
        
        while (len(selected_overlays) < rules["max_overlays"] and 
               len(available_overlays) > 0):
            overlay = random.choice(available_overlays)
            available_overlays.remove(overlay)
            
            # Check compatibility
            compatible = True
            for selected in selected_overlays:
                if (selected["name"] in rules["compatibility"] and
                    overlay["name"] not in rules["compatibility"][selected["name"]]):
                    compatible = False
                    break
                    
            if compatible:
                # Adjust opacity based on intensity
                adjusted = overlay.copy()
                adjusted["opacity"] *= intensity
                selected_overlays.append(adjusted)
                
        effects["overlay"] = selected_overlays
        
        # Add content-specific adjustments
        self._adjust_for_content(effects, content_type)
        
        return effects
        
    def _select_effect(
        self,
        effect_type: str,
        style: str,
        intensity: float
    ) -> Dict[str, Any]:
        """Select and adjust an effect based on style and intensity."""
        options = self.effect_library[effect_type][style]
        effect = random.choice(options).copy()
        
        # Adjust effect parameters based on intensity
        if effect_type == "color":
            for key in ["contrast", "saturation", "brightness"]:
                if key in effect:
                    deviation = abs(1 - effect[key])
                    effect[key] = 1 + (deviation * intensity * 
                                     (1 if effect[key] > 1 else -1))
        elif effect_type == "filter":
            effect["amount"] *= intensity
            
        return effect
        
    def _adjust_for_content(
        self,
        effects: Dict[str, Any],
        content_type: str
    ) -> None:
        """Adjust effects based on content type."""
        if content_type == "exterior":
            # Enhance contrast and saturation for exterior shots
            if effects["color"]:
                effects["color"]["contrast"] *= 1.1
                effects["color"]["saturation"] *= 1.1
        elif content_type == "interior":
            # Boost brightness for interior shots
            if effects["color"]:
                effects["color"]["brightness"] = max(
                    1.1,
                    effects["color"].get("brightness", 1.0)
                )
        elif content_type == "feature":
            # Enhance clarity for feature shots
            if effects["filter"]:
                effects["filter"]["amount"] *= 1.2
                
    def blend_effects(
        self,
        effects_a: Dict[str, Any],
        effects_b: Dict[str, Any],
        blend_factor: float
    ) -> Dict[str, Any]:
        """
        Blend two effect combinations.
        
        Args:
            effects_a: First effect combination
            effects_b: Second effect combination
            blend_factor: Blend factor (0-1, 0=full A, 1=full B)
            
        Returns:
            Blended effect combination
        """
        blended = {
            "color": {},
            "filter": {},
            "overlay": []
        }
        
        # Blend color effects
        if effects_a.get("color") and effects_b.get("color"):
            for key in ["contrast", "saturation", "brightness", "temperature"]:
                if key in effects_a["color"] and key in effects_b["color"]:
                    blended["color"][key] = (
                        effects_a["color"][key] * (1 - blend_factor) +
                        effects_b["color"][key] * blend_factor
                    )
                    
        # Blend filter effects
        if effects_a.get("filter") and effects_b.get("filter"):
            for key in ["amount", "radius"]:
                if key in effects_a["filter"] and key in effects_b["filter"]:
                    blended["filter"][key] = (
                        effects_a["filter"][key] * (1 - blend_factor) +
                        effects_b["filter"][key] * blend_factor
                    )
                    
        # Select overlays based on blend factor
        all_overlays = (effects_a.get("overlay", []) + 
                       effects_b.get("overlay", []))
        if all_overlays:
            # Prioritize overlays from the dominant effect set
            dominant = effects_a if blend_factor < 0.5 else effects_b
            secondary = effects_b if blend_factor < 0.5 else effects_a
            
            blended["overlay"] = (
                dominant.get("overlay", [])[:1] +
                secondary.get("overlay", [])[:1]
            )
            
        return blended
