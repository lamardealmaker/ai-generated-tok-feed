from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from pathlib import Path

class VisualEffect:
    """Base class for visual effects."""
    
    def __init__(self, intensity: float = 1.0):
        self.intensity = max(0.0, min(1.0, intensity))
        
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply effect to frame."""
        raise NotImplementedError

class ColorEnhance(VisualEffect):
    """Enhance colors in frame."""
    
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Enhance saturation
        hsv[:, :, 1] = hsv[:, :, 1] * (1 + self.intensity * 0.5)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back to BGR
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

class Sharpness(VisualEffect):
    """Adjust image sharpness."""
    
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]]) * self.intensity
        return cv2.filter2D(frame, -1, kernel)

class Vignette(VisualEffect):
    """Add vignette effect."""
    
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        rows, cols = frame.shape[:2]
        
        # Generate vignette mask
        X_resultant, Y_resultant = np.meshgrid(np.linspace(-1, 1, cols),
                                             np.linspace(-1, 1, rows))
        mask = np.sqrt(X_resultant**2 + Y_resultant**2)
        mask = 1 - mask
        mask = np.clip(mask + (1 - self.intensity), 0, 1)
        
        # Expand mask to match frame channels
        mask = np.dstack([mask] * 3)
        
        return (frame * mask).astype(np.uint8)

class FilmGrain(VisualEffect):
    """Add film grain effect."""
    
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, 25 * self.intensity, frame.shape).astype(np.uint8)
        return cv2.add(frame, noise)

class CinematicBars(VisualEffect):
    """Add cinematic bars effect."""
    
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        height = frame.shape[0]
        bar_height = int(height * 0.1 * self.intensity)
        
        frame[:bar_height] = 0  # Top bar
        frame[-bar_height:] = 0  # Bottom bar
        return frame

class EffectLibrary:
    """Library of available visual effects."""
    
    def __init__(self):
        self.effects = {
            'color_enhance': ColorEnhance,
            'sharpness': Sharpness,
            'vignette': Vignette,
            'film_grain': FilmGrain,
            'cinematic_bars': CinematicBars
        }
        
    def get_effect(self, name: str, intensity: float = 1.0) -> VisualEffect:
        """Get effect by name."""
        if name not in self.effects:
            raise ValueError(f"Unknown effect: {name}")
            
        return self.effects[name](intensity)
        
    def list_effects(self) -> List[str]:
        """List available effects."""
        return list(self.effects.keys())
        
    def get_effect_info(self, name: str) -> Dict[str, Any]:
        """Get information about an effect."""
        if name not in self.effects:
            raise ValueError(f"Unknown effect: {name}")
            
        effect_class = self.effects[name]
        return {
            'name': name,
            'description': effect_class.__doc__,
            'parameters': {
                'intensity': 'Effect intensity (0-1)'
            }
        }

class EffectChain:
    """Chain multiple visual effects together."""
    
    def __init__(self):
        self.effects: List[VisualEffect] = []
        self.library = EffectLibrary()
        
    def add_effect(self, name: str, intensity: float = 1.0) -> 'EffectChain':
        """Add effect to chain."""
        effect = self.library.get_effect(name, intensity)
        self.effects.append(effect)
        return self
        
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply all effects in chain."""
        result = frame.copy()
        for effect in self.effects:
            result = await effect.apply(result)
        return result
        
    def clear(self) -> None:
        """Clear all effects from chain."""
        self.effects.clear()
