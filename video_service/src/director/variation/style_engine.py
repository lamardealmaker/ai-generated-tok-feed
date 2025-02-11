from typing import Dict, Any, List, Optional, Tuple
import random
import numpy as np
from dataclasses import dataclass

@dataclass
class StyleComponent:
    """Component of a video style."""
    name: str
    weight: float
    attributes: Dict[str, Any]

class StyleCombination:
    """Combination of multiple style components."""
    
    def __init__(self, components: List[StyleComponent]):
        self.components = components
        self._normalize_weights()
        
    def _normalize_weights(self) -> None:
        """Normalize component weights to sum to 1."""
        total_weight = sum(c.weight for c in self.components)
        if total_weight > 0:
            for component in self.components:
                component.weight /= total_weight
                
    def get_weighted_value(self, attribute: str) -> Any:
        """Get weighted average of an attribute across components."""
        values = []
        weights = []
        
        for component in self.components:
            if attribute in component.attributes:
                values.append(component.attributes[attribute])
                weights.append(component.weight)
                
        if not values:
            return None
            
        # Handle different value types
        if isinstance(values[0], (int, float)):
            return np.average(values, weights=weights)
        elif isinstance(values[0], str):
            # For strings, return the value with highest weight
            max_weight_idx = np.argmax(weights)
            return values[max_weight_idx]
        elif isinstance(values[0], dict):
            # Merge dictionaries with weights
            result = {}
            for value, weight in zip(values, weights):
                for k, v in value.items():
                    if k not in result:
                        result[k] = v if isinstance(v, str) else 0
                    if isinstance(v, (int, float)):
                        if isinstance(result[k], (int, float)):
                            result[k] += v * weight
                    else:
                        # For non-numeric values (like colors), use weighted selection
                        if weight > result.get(f'{k}_weight', 0):
                            result[k] = v
                            result[f'{k}_weight'] = weight
            
            # Clean up temporary weight keys
            for k in list(result.keys()):
                if k.endswith('_weight'):
                    del result[k]
            return result
            
        return values[0]

class StyleEngine:
    """Engine for combining and varying video styles."""
    
    def __init__(self):
        self.base_styles = self._create_base_styles()
        
    def _create_base_styles(self) -> Dict[str, Dict[str, Any]]:
        """Create base video styles."""
        return {
            'modern': {
                'color_scheme': {
                    'primary': '#FF5733',
                    'secondary': '#33FF57',
                    'text': '#FFFFFF'
                },
                'transitions': {
                    'types': ['slide_left', 'slide_right', 'zoom_in'],
                    'duration': 0.5
                },
                'effects': {
                    'color_enhance': 0.7,
                    'sharpness': 0.5
                },
                'text': {
                    'font': 'Roboto',
                    'animation': 'slide'
                }
            },
            'luxury': {
                'color_scheme': {
                    'primary': '#FFD700',
                    'secondary': '#000000',
                    'text': '#FFFFFF'
                },
                'transitions': {
                    'types': ['dissolve', 'zoom_out'],
                    'duration': 1.0
                },
                'effects': {
                    'vignette': 0.3,
                    'film_grain': 0.2
                },
                'text': {
                    'font': 'Playfair Display',
                    'animation': 'fade'
                }
            },
            'minimalist': {
                'color_scheme': {
                    'primary': '#FFFFFF',
                    'secondary': '#000000',
                    'text': '#000000'
                },
                'transitions': {
                    'types': ['dissolve'],
                    'duration': 0.7
                },
                'effects': {
                    'sharpness': 0.3
                },
                'text': {
                    'font': 'Helvetica',
                    'animation': 'fade'
                }
            }
        }
        
    def create_style_combination(self,
                               styles: List[Tuple[str, float]],
                               variation: float = 0.2) -> StyleCombination:
        """
        Create combination of styles with weights.
        
        Args:
            styles: List of (style_name, weight) tuples
            variation: Amount of random variation (0-1)
        """
        components = []
        
        for style_name, weight in styles:
            if style_name not in self.base_styles:
                raise ValueError(f"Unknown style: {style_name}")
                
            # Deep copy attributes with variation
            attributes = {}
            base_attrs = self.base_styles[style_name]
            
            for key, value in base_attrs.items():
                if isinstance(value, (int, float)):
                    # Add random variation to numeric values
                    variation_amount = value * variation
                    attributes[key] = value + random.uniform(-variation_amount, variation_amount)
                elif isinstance(value, dict):
                    # Recursively handle nested dictionaries
                    attributes[key] = {}
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            variation_amount = v * variation
                            attributes[key][k] = v + random.uniform(-variation_amount, variation_amount)
                        else:
                            attributes[key][k] = v
                else:
                    attributes[key] = value
                    
            components.append(StyleComponent(style_name, weight, attributes))
            
        return StyleCombination(components)
        
    def interpolate_styles(self,
                          style1: StyleCombination,
                          style2: StyleCombination,
                          t: float) -> StyleCombination:
        """
        Interpolate between two style combinations.
        
        Args:
            style1: First style combination
            style2: Second style combination
            t: Interpolation factor (0-1)
        """
        # Combine components from both styles
        components = []
        
        # Add components from style1 with adjusted weights
        for comp in style1.components:
            components.append(StyleComponent(
                comp.name,
                comp.weight * (1 - t),
                comp.attributes
            ))
            
        # Add components from style2 with adjusted weights
        for comp in style2.components:
            components.append(StyleComponent(
                comp.name,
                comp.weight * t,
                comp.attributes
            ))
            
        return StyleCombination(components)
