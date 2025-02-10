from typing import List, Tuple, Dict
import moviepy.editor as mp
from PIL import Image
import io
import tempfile
import os
from urllib.parse import urlparse
import httpx
from .text_overlay import TextOverlay

class ImageProcessor:
    def __init__(self):
        self.slide_duration = 3  # seconds per slide
        self.transition_duration = 1  # seconds for transition
        self.target_size = (1080, 1920)  # vertical video format (9:16)
        self.text_overlay = TextOverlay()
    
    async def download_image(self, image_url: str) -> str:
        """Download image from URL and save to temporary file."""
        try:
            # Parse URL to get file extension
            parsed = urlparse(image_url)
            ext = os.path.splitext(parsed.path)[1] or '.jpg'
            
            # Create temporary file
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url)
                response.raise_for_status()
                temp.write(response.content)
                temp.close()
                return temp.name
        except Exception as e:
            raise Exception(f"Failed to download image: {str(e)}")

    def process_image(self, image_path: str) -> mp.VideoClip:
        """Process a single image into a video clip."""
        try:
            # Open and resize image while maintaining aspect ratio
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Calculate resize dimensions maintaining aspect ratio
                img_ratio = img.size[0] / img.size[1]
                target_ratio = self.target_size[0] / self.target_size[1]
                
                if img_ratio > target_ratio:
                    # Image is wider than target ratio
                    new_height = self.target_size[1]
                    new_width = int(new_height * img_ratio)
                else:
                    # Image is taller than target ratio
                    new_width = self.target_size[0]
                    new_height = int(new_width / img_ratio)
                
                # Resize image
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Create new blank image with target size
                new_img = Image.new('RGB', self.target_size, (0, 0, 0))
                
                # Paste resized image in center
                paste_x = (self.target_size[0] - new_width) // 2
                paste_y = (self.target_size[1] - new_height) // 2
                new_img.paste(img, (paste_x, paste_y))
                
                # Save to temporary file
                temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                new_img.save(temp.name, 'JPEG', quality=95)
                temp.close()
                
                # Create video clip
                clip = mp.ImageClip(temp.name, duration=self.slide_duration)
                os.unlink(temp.name)  # Clean up temp file
                
                return clip
                
        except Exception as e:
            raise Exception(f"Failed to process image: {str(e)}")
    
    async def create_slides(self, image_urls: List[str], properties: Dict) -> List[mp.VideoClip]:
        """Create video clips from a list of image URLs with property details overlay."""
        clips = []
        temp_files = []
        
        try:
            # Download and process each image
            for i, url in enumerate(image_urls):
                temp_file = await self.download_image(url)
                temp_files.append(temp_file)
                
                # Create base image clip
                base_clip = self.process_image(temp_file)
                
                # Create text overlay for this slide
                slide_properties = self._get_slide_properties(properties, i)
                text_overlay = self.text_overlay.create_property_details_overlay(
                    slide_properties,
                    self.slide_duration,
                    self.target_size
                )
                
                # Combine image and text
                final_clip = mp.CompositeVideoClip([
                    base_clip,
                    text_overlay
                ])
                
                clips.append(final_clip)
            
            return clips
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
    def _get_slide_properties(self, properties: Dict, slide_index: int) -> Dict:
        """Get properties to display for a specific slide."""
        # First slide shows all details
        if slide_index == 0:
            return properties
        
        # Other slides show minimal info
        return {
            'title': properties.get('title', ''),
            'price': properties.get('price', '')
        }
