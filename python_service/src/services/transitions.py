from typing import Tuple, Optional, List
import numpy as np
from moviepy.editor import VideoClip, VideoFileClip
from moviepy.video.fx.resize import resize
from moviepy.video.fx.crop import crop

class TransitionEffects:
    @staticmethod
    def zoom_effect(clip: VideoClip, 
                   zoom_factor: float = 1.3, 
                   duration: float = 2.0) -> VideoClip:
        """Apply a smooth zoom effect to the clip"""
        w, h = clip.size
        
        def effect(get_frame, t):
            current_zoom = 1 + (zoom_factor - 1) * (t / duration)
            frame = get_frame(t)
            
            # Calculate new dimensions
            new_w = int(w * current_zoom)
            new_h = int(h * current_zoom)
            
            # Resize frame
            resized = resize(VideoClip(lambda t: frame), (new_w, new_h))
            
            # Crop to original size from center
            x1 = (new_w - w) // 2
            y1 = (new_h - h) // 2
            return crop(resized, x1=x1, y1=y1, width=w, height=h).get_frame(0)
            
        return VideoClip(lambda t: effect(clip.get_frame, t), 
                        duration=clip.duration)
    
    @staticmethod
    def ken_burns_effect(clip: VideoClip,
                        zoom_range: Tuple[float, float] = (1.0, 1.3),
                        pan_range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None) -> VideoClip:
        """Apply Ken Burns effect (slow zoom and pan)"""
        w, h = clip.size
        start_zoom, end_zoom = zoom_range
        
        # Default pan range if none provided
        if pan_range is None:
            pan_range = ((0, 0), (0.1, 0.1))  # Slight pan
            
        (start_x, start_y), (end_x, end_y) = pan_range
        
        def effect(get_frame, t):
            progress = t / clip.duration
            current_zoom = start_zoom + (end_zoom - start_zoom) * progress
            
            # Calculate pan offsets
            offset_x = (start_x + (end_x - start_x) * progress) * w
            offset_y = (start_y + (end_y - start_y) * progress) * h
            
            frame = get_frame(t)
            
            # Apply zoom
            new_w = int(w * current_zoom)
            new_h = int(h * current_zoom)
            resized = resize(VideoClip(lambda t: frame), (new_w, new_h))
            
            # Apply pan by adjusting crop position
            x1 = int((new_w - w) // 2 + offset_x)
            y1 = int((new_h - h) // 2 + offset_y)
            return crop(resized, x1=x1, y1=y1, width=w, height=h).get_frame(0)
            
        return VideoClip(lambda t: effect(clip.get_frame, t), 
                        duration=clip.duration)
    
    @staticmethod
    def slide_transition(clip1: VideoClip, 
                        clip2: VideoClip, 
                        direction: str = 'left',
                        duration: float = 1.0) -> VideoClip:
        """Create a sliding transition between two clips"""
        w, h = clip1.size
        
        def make_frame(t):
            if t < duration:
                progress = t / duration
                if direction == 'left':
                    x_offset = int(w * (1 - progress))
                    frame1 = clip1.get_frame(t)
                    frame2 = clip2.get_frame(t)
                    result = frame2.copy()
                    result[:, :w-x_offset] = frame1[:, x_offset:]
                elif direction == 'right':
                    x_offset = int(w * progress)
                    frame1 = clip1.get_frame(t)
                    frame2 = clip2.get_frame(t)
                    result = frame2.copy()
                    result[:, x_offset:] = frame1[:, :w-x_offset]
                return result
            else:
                return clip2.get_frame(t)
                
        return VideoClip(make_frame, duration=duration)
    
    @staticmethod
    def progress_bar(clip: VideoClip, 
                    color: Tuple[int, int, int] = (255, 255, 255),
                    height: int = 3,
                    opacity: float = 0.7) -> VideoClip:
        """Add a progress bar to the bottom of the clip"""
        w, h = clip.size
        
        def make_frame(t):
            frame = clip.get_frame(t)
            progress = t / clip.duration
            bar_width = int(w * progress)
            
            # Create progress bar
            bar = np.zeros((height, w, 3), dtype=np.uint8)
            bar[:, :bar_width] = color
            
            # Apply opacity
            bar = bar.astype(float) * opacity
            
            # Add bar to bottom of frame
            result = frame.copy()
            result[-height:] = bar
            return result
            
        return VideoClip(make_frame, duration=clip.duration)
