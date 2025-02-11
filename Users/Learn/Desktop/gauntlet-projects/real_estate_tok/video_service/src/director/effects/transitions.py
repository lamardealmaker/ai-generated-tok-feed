from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path
import cv2

class TransitionEffect:
    """Base class for video transitions."""
    
    def __init__(self, duration: float = 0.5):
        self.duration = duration
        
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        """Apply transition effect between two frames."""
        raise NotImplementedError

class CrossDissolve(TransitionEffect):
    """Smooth fade transition between frames."""
    
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        return cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)

class SlideTransition(TransitionEffect):
    """Slide one frame over another."""
    
    def __init__(self, direction: str = 'left', duration: float = 0.5):
        super().__init__(duration)
        self.direction = direction
        
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        height, width = frame1.shape[:2]
        result = np.zeros_like(frame1)
        
        if self.direction == 'left':
            offset = int(width * progress)
            result[:, :width-offset] = frame1[:, offset:]
            result[:, width-offset:] = frame2[:, :offset]
        elif self.direction == 'right':
            offset = int(width * progress)
            result[:, offset:] = frame1[:, :width-offset]
            result[:, :offset] = frame2[:, width-offset:]
        
        return result

class ZoomTransition(TransitionEffect):
    """Zoom in/out transition effect."""
    
    def __init__(self, zoom_in: bool = True, duration: float = 0.5):
        super().__init__(duration)
        self.zoom_in = zoom_in
        
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        if self.zoom_in:
            scale = 1 + progress
            frame1_scaled = cv2.resize(frame1, None, fx=scale, fy=scale)
            h1, w1 = frame1_scaled.shape[:2]
            y1 = int((h1 - frame1.shape[0]) / 2)
            x1 = int((w1 - frame1.shape[1]) / 2)
            frame1_crop = frame1_scaled[y1:y1+frame1.shape[0], x1:x1+frame1.shape[1]]
            return cv2.addWeighted(frame1_crop, 1 - progress, frame2, progress, 0)
        else:
            scale = 2 - progress
            frame2_scaled = cv2.resize(frame2, None, fx=scale, fy=scale)
            h2, w2 = frame2_scaled.shape[:2]
            y2 = int((h2 - frame2.shape[0]) / 2)
            x2 = int((w2 - frame2.shape[1]) / 2)
            frame2_crop = frame2_scaled[y2:y2+frame2.shape[0], x2:x2+frame2.shape[1]]
            return cv2.addWeighted(frame1, 1 - progress, frame2_crop, progress, 0)

class WhipPanTransition(TransitionEffect):
    """Fast whip pan transition effect."""
    
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        if progress < 0.5:
            kernel_size = int((progress * 2) * 50)
            if kernel_size > 0:
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
                kernel = kernel / kernel_size
                frame1 = cv2.filter2D(frame1, -1, kernel)
            return frame1
        else:
            kernel_size = int((1 - progress) * 2 * 50)
            if kernel_size > 0:
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
                kernel = kernel / kernel_size
                frame2 = cv2.filter2D(frame2, -1, kernel)
            return frame2

class JumpCutTransition(TransitionEffect):
    """Instant jump cut transition effect with no intermediate blending."""
    
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        # Jump cut: immediately switch to the next frame.
        return frame2

class GlitchTransition(TransitionEffect):
    """Glitch effect transition that overlays random noise for a brief moment."""
    
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        import cv2
        import numpy as np
        if progress < 0.5:
            noise = np.random.randint(0, 50, frame1.shape, dtype=np.uint8)
            return cv2.add(frame1, noise)
        else:
            noise = np.random.randint(0, 50, frame2.shape, dtype=np.uint8)
            return cv2.add(frame2, noise)

class TransitionLibrary:
    """Library of available transition effects."""
    
    def __init__(self):
        self.transitions = {
            'dissolve': CrossDissolve,
            'slide_left': lambda: SlideTransition('left'),
            'slide_right': lambda: SlideTransition('right'),
            'zoom_in': lambda: ZoomTransition(True),
            'zoom_out': lambda: ZoomTransition(False),
            'whip_pan': WhipPanTransition,
            'jump_cut': JumpCutTransition,
            'glitch': GlitchTransition
        }
        
    def get_transition(self, name: str, **kwargs) -> TransitionEffect:
        """Get transition effect by name."""
        if name not in self.transitions:
            raise ValueError(f"Unknown transition effect: {name}")
            
        transition_class = self.transitions[name]
        if callable(transition_class):
            return transition_class()
        return transition_class(**kwargs)
        
    def list_transitions(self) -> List[str]:
        """List available transition effects."""
        return list(self.transitions.keys())
        
    def get_transition_info(self, name: str) -> Dict[str, Any]:
        """Get information about a transition effect."""
        if name not in self.transitions:
            raise ValueError(f"Unknown transition effect: {name}")
            
        transition_class = self.transitions[name]
        return {
            'name': name,
            'description': transition_class.__doc__,
            'parameters': {
                'duration': 'Duration of transition in seconds'
            }
        }