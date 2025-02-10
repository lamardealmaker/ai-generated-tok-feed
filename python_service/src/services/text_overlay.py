from typing import Tuple, Dict
import moviepy.editor as mp
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import textwrap

class TextOverlay:
    def __init__(self):
        self.font_size = {
            'title': 70,
            'price': 60,
            'location': 50,
            'details': 40
        }
        self.colors = {
            'primary': 'white',
            'shadow': 'black',
            'price': '#2ECC71'  # Green color for price
        }
        self.shadow_offset = 3
        
    def _create_text_clip(
        self,
        text: str,
        font_size: int,
        color: str,
        position: Tuple[int, int],
        duration: float,
        with_shadow: bool = True
    ) -> mp.TextClip:
        """Create a text clip with optional shadow effect."""
        # Create main text
        txt_clip = mp.TextClip(
            text,
            font='Arial',
            fontsize=font_size,
            color=color
        )
        
        if with_shadow:
            # Create shadow text
            shadow = mp.TextClip(
                text,
                font='Arial',
                fontsize=font_size,
                color=self.colors['shadow']
            )
            
            # Position shadow slightly offset
            shadow = shadow.set_position((
                position[0] + self.shadow_offset,
                position[1] + self.shadow_offset
            ))
            
            # Composite shadow and main text
            txt_clip = mp.CompositeVideoClip([
                shadow.set_duration(duration),
                txt_clip.set_position(position).set_duration(duration)
            ])
        else:
            txt_clip = txt_clip.set_position(position).set_duration(duration)
            
        return txt_clip
        
    def create_property_details_overlay(
        self,
        properties: Dict,
        duration: float,
        size: Tuple[int, int] = (1080, 1920)
    ) -> mp.CompositeVideoClip:
        """
        Create text overlays for property details.
        
        Args:
            properties: Dictionary containing property details
            duration: Duration of the overlay in seconds
            size: Video dimensions (width, height)
            
        Returns:
            CompositeVideoClip with all text overlays
        """
        clips = []
        
        # Calculate positions (relative to video size)
        positions = {
            'title': (size[0] * 0.1, size[1] * 0.1),  # Top 10%
            'price': (size[0] * 0.1, size[1] * 0.2),  # Below title
            'location': (size[0] * 0.1, size[1] * 0.3),  # Below price
            'details': (size[0] * 0.1, size[1] * 0.4)  # Below location
        }
        
        # Add title
        if 'title' in properties:
            clips.append(self._create_text_clip(
                properties['title'],
                self.font_size['title'],
                self.colors['primary'],
                positions['title'],
                duration
            ))
            
        # Add price with special color
        if 'price' in properties:
            clips.append(self._create_text_clip(
                properties['price'],
                self.font_size['price'],
                self.colors['price'],
                positions['price'],
                duration
            ))
            
        # Add location
        if 'location' in properties:
            clips.append(self._create_text_clip(
                properties['location'],
                self.font_size['location'],
                self.colors['primary'],
                positions['location'],
                duration
            ))
            
        # Add additional details if present
        if 'details' in properties:
            details_text = properties['details']
            if isinstance(details_text, list):
                details_text = " • ".join(details_text)
                
            # Wrap text to avoid overflow
            wrapped_text = textwrap.fill(details_text, width=40)
            
            clips.append(self._create_text_clip(
                wrapped_text,
                self.font_size['details'],
                self.colors['primary'],
                positions['details'],
                duration
            ))
            
        # Create transparent background
        bg = mp.ColorClip(size, color=(0, 0, 0))
        bg = bg.set_opacity(0).set_duration(duration)
        
        # Combine all clips
        return mp.CompositeVideoClip([bg] + clips)
