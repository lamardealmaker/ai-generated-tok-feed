from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import emoji
from moviepy.editor import VideoClip, ImageClip, CompositeVideoClip, ColorClip
from ..models.text_overlay import (
    TextOverlay, TextPosition, TextAlignment,
    AnimationType, TextStyle, Animation
)


class TextOverlayService:
    def __init__(self):
        self.default_font = "Arial"
        self.cache: Dict[str, ImageFont.FreeTypeFont] = {}
        
    def _get_font(self, family: str, size: int) -> ImageFont.FreeTypeFont:
        """Get font from cache or load it"""
        key = f"{family}_{size}"
        if key not in self.cache:
            try:
                self.cache[key] = ImageFont.truetype(family, size)
            except OSError:
                # Fallback to default font if specified font not found
                self.cache[key] = ImageFont.truetype(self.default_font, size)
        return self.cache[key]
    
    def _calculate_position(
        self,
        text_overlay: TextOverlay,
        frame_size: Tuple[int, int],
        text_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculate the position of text on frame"""
        frame_width, frame_height = frame_size
        text_width, text_height = text_size
        
        if text_overlay.custom_position:
            return text_overlay.custom_position
            
        # Calculate x position based on alignment
        if text_overlay.alignment == TextAlignment.LEFT:
            x = 20  # Padding from left
        elif text_overlay.alignment == TextAlignment.RIGHT:
            x = frame_width - text_width - 20  # Padding from right
        else:  # CENTER
            x = (frame_width - text_width) // 2
            
        # Calculate y position based on position type
        if text_overlay.position == TextPosition.TOP:
            y = 20  # Padding from top
        elif text_overlay.position == TextPosition.MIDDLE:
            y = (frame_height - text_height) // 2
        else:  # BOTTOM
            y = frame_height - text_height - 20  # Padding from bottom
            
        return (x, y)
    
    def _apply_animation(
        self,
        clip: VideoClip,
        animation: Animation,
        frame_size: Tuple[int, int],
        text_overlay: Optional[TextOverlay] = None
    ) -> VideoClip:
        """Apply animation effect to the clip"""
        w, h = frame_size
        duration = animation.duration
        
        if animation.type == AnimationType.NONE:
            return clip
            
        if animation.type == AnimationType.FADE:
            # Create a crossfade effect
            transparent = ColorClip(clip.size, col=(0, 0, 0), duration=clip.duration).set_opacity(0)
            return CompositeVideoClip([transparent, clip.crossfadein(duration)])
            
        if animation.type == AnimationType.SCALE:
            # Start small and scale up
            clip = clip.resize(lambda t: 1.0 if t > duration else max(0.1, t/duration))
            return clip
            
        if animation.type == AnimationType.SLIDE_LEFT:
            # Start from right and slide left
            return clip.set_position(lambda t: (w if t < duration else 0, clip.pos(t)[1]))
            
        if animation.type == AnimationType.SLIDE_RIGHT:
            # Start from left and slide right
            return clip.set_position(lambda t: (-w if t < duration else 0, clip.pos(t)[1]))
            
        if animation.type == AnimationType.SLIDE_UP:
            # Start from bottom and slide up
            return clip.set_position(lambda t: (clip.pos(t)[0], h if t < duration else 0))
            
        if animation.type == AnimationType.SLIDE_DOWN:
            # Start from top and slide down
            return clip.set_position(lambda t: (clip.pos(t)[0], -h if t < duration else 0))
            
        if animation.type == AnimationType.TYPE:
            if not text_overlay:
                return clip
                
            def make_frame(t):
                if t < duration:
                    progress = t / duration
                    char_count = int(len(text_overlay.text) * progress)
                    partial_text = text_overlay.text[:char_count]
                    return self._render_text_frame(
                        partial_text,
                        text_overlay.style,
                        frame_size,
                        text_overlay.position,
                        text_overlay.alignment
                    )
                return self._render_text_frame(
                    text_overlay.text,
                    text_overlay.style,
                    frame_size,
                    text_overlay.position,
                    text_overlay.alignment
                )
            return VideoClip(make_frame, duration=clip.duration)
            
        return clip
    
    def _render_text_frame(
        self,
        text: str,
        style: TextStyle,
        frame_size: Tuple[int, int],
        position: TextPosition,
        alignment: TextAlignment,
        emoji_scale: float = 1.0
    ) -> np.ndarray:
        """Render a single frame with text overlay"""
        # Create transparent frame
        frame = Image.new('RGBA', frame_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(frame)
        
        # Get font
        font = self._get_font(style.font_family, style.font_size)
        
        # Replace emoji shortcodes with actual emojis
        text = emoji.emojize(text, language='alias')
        
        # Calculate text size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Calculate position
        x, y = self._calculate_position(
            TextOverlay(
                text=text,
                position=position,
                alignment=alignment,
                style=style
            ),
            frame_size,
            (text_width, text_height)
        )
        
        # Draw background if specified
        if style.background_color and style.background_opacity > 0:
            bg_color = (*style.background_color, int(255 * style.background_opacity))
            padding = style.font_size // 4
            draw.rectangle(
                [
                    x - padding,
                    y - padding,
                    x + text_width + padding,
                    y + text_height + padding
                ],
                fill=bg_color
            )
        
        # Draw shadow if enabled
        if style.shadow:
            shadow_x = x + style.shadow_offset[0]
            shadow_y = y + style.shadow_offset[1]
            draw.text(
                (shadow_x, shadow_y),
                text,
                font=font,
                fill=style.shadow_color
            )
        
        # Draw stroke if specified
        if style.stroke_width > 0:
            for offset_x in range(-style.stroke_width, style.stroke_width + 1):
                for offset_y in range(-style.stroke_width, style.stroke_width + 1):
                    if offset_x == 0 and offset_y == 0:
                        continue
                    draw.text(
                        (x + offset_x, y + offset_y),
                        text,
                        font=font,
                        fill=style.stroke_color
                    )
        
        # Draw main text
        draw.text((x, y), text, font=font, fill=style.color)
        
        # Convert to numpy array
        return np.array(frame)
    
    def create_text_clip(
        self,
        text_overlay: TextOverlay,
        frame_size: Tuple[int, int],
        duration: float
    ) -> VideoClip:
        """Create a video clip with animated text overlay"""
        # Create the base frame with text
        base_frame = self._render_text_frame(
            text_overlay.text,
            text_overlay.style,
            frame_size,
            text_overlay.position,
            text_overlay.alignment,
            text_overlay.emoji_scale
        )
        
        # Create clip from the frame
        clip = ImageClip(base_frame, duration=duration)
        
        # Apply animation
        if text_overlay.animation.type != AnimationType.NONE:
            clip = self._apply_animation(
                clip,
                text_overlay.animation,
                frame_size,
                text_overlay
            )
        
        # Set start time
        if text_overlay.start_time > 0:
            clip = clip.set_start(text_overlay.start_time)
        
        # Set end time if specified
        if text_overlay.end_time is not None:
            clip = clip.set_end(text_overlay.end_time)
        
        return clip
    
    def apply_text_overlays(
        self,
        base_clip: VideoClip,
        text_overlays: List[TextOverlay]
    ) -> VideoClip:
        """Apply multiple text overlays to a video clip"""
        clips = [base_clip]
        
        for overlay in text_overlays:
            text_clip = self.create_text_clip(
                overlay,
                (base_clip.w, base_clip.h),
                base_clip.duration
            )
            clips.append(text_clip)
        
        # Composite all clips
        final_clip = CompositeVideoClip(clips)
        return final_clip
