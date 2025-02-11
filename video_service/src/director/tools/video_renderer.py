from typing import Dict, Any, List, Optional
import os
import logging
import moviepy.editor as mp
from moviepy.editor import VideoFileClip, ImageClip, TextClip, CompositeVideoClip, AudioFileClip, CompositeAudioClip
import firebase_admin
from firebase_admin import storage
import tempfile
import uuid

logger = logging.getLogger(__name__)

class VideoRenderer:
    """Tool for rendering video segments into a final video."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Firebase if not already initialized
        try:
            self.bucket = storage.bucket()
        except ValueError:
            cred = firebase_admin.credentials.Certificate("config/firebase-service-account.json")
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'project3-ziltok.firebasestorage.app'
            })
            self.bucket = storage.bucket()
    
    def download_image(self, image_url: str) -> str:
        """Download image from Firebase Storage."""
        if not image_url.startswith("gs://"):
            return image_url
            
        # Extract blob path from gs:// URL
        blob_path = image_url.replace("gs://project3-ziltok.firebasestorage.app/", "")
        blob = self.bucket.blob(blob_path)
        
        # Create temp file
        _, temp_path = tempfile.mkstemp(suffix=os.path.splitext(blob_path)[1])
        
        # Download to temp file
        blob.download_to_filename(temp_path)
        return temp_path
    
    def apply_effects(self, clip: VideoFileClip, effects: Dict[str, Any]) -> VideoFileClip:
        """Apply video effects to a clip."""
        # Apply basic effects
        if "filter" in effects:
            filter_effect = effects["filter"]
            if filter_effect["name"] in ["glow", "bloom"]:
                # Add a slight fade in/out for visual interest
                clip = clip.fadein(0.5).fadeout(0.5)
            elif filter_effect["name"] == "blur":
                # Add a mirror effect instead of blur
                clip = clip.fx(mp.vfx.mirror_x)
                
        # Add a slight zoom for all clips
        clip = clip.resize(1.1)
        
        return clip
    
    def create_text_overlay(self, text: str, duration: float, size: int = 30) -> TextClip:
        """Create a text overlay clip."""
        return TextClip(text, fontsize=size, color='white', font='Arial', 
                       size=(900, None), method='caption', align='center',  # Wider text for vertical video
                       stroke_color='black', stroke_width=2)\
               .set_duration(duration)\
               .set_position(('center', 'bottom'))\
               .crossfadein(0.3)\
               .crossfadeout(0.3)
    
    def render_video(self, segments: List[Dict[str, Any]], music: Dict[str, Any], 
                    output_path: Optional[str] = None) -> str:
        """
        Render the final video from segments and music.
        
        Args:
            segments: List of video segments with effects
            music: Music configuration
            output_path: Optional custom output path
            
        Returns:
            Path to the rendered video file
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"video_{uuid.uuid4()}.mp4")
            
        clips = []
        
        # Process each segment
        for segment in segments:
            # Download and create optimized image clip
            image_path = self.download_image(segment["image_url"])
            
            # Create clip with TikTok resolution (1080x1920)
            clip = ImageClip(image_path)
            
            # Calculate dimensions to fill TikTok aspect ratio while maintaining center focus
            img_w, img_h = clip.size
            target_ratio = 9/16  # TikTok aspect ratio
            current_ratio = img_w/img_h
            
            if current_ratio > target_ratio:  # Image is too wide
                # Resize based on height
                new_h = 1920
                new_w = int(img_w * (new_h/img_h))
                clip = clip.resize((new_w, new_h))
                # Crop to width
                x_center = new_w/2
                clip = clip.crop(x1=int(x_center-540), x2=int(x_center+540))  # 1080px wide
            else:  # Image is too tall or perfect
                # Resize based on width
                new_w = 1080
                new_h = int(img_h * (new_w/img_w))
                clip = clip.resize((new_w, new_h))
                # Crop to height if needed
                if new_h > 1920:
                    y_center = new_h/2
                    clip = clip.crop(y1=int(y_center-960), y2=int(y_center+960))  # 1920px tall
            
            # Set duration
            clip = clip.set_duration(segment["duration"])
            
            # Apply effects
            if "effects" in segment:
                clip = self.apply_effects(clip, segment["effects"])
            
            # Add text overlay
            if "text" in segment:
                text_clip = self.create_text_overlay(segment["text"], segment["duration"])
                clip = CompositeVideoClip([clip, text_clip])
            
            clips.append(clip)
        
        # Concatenate all clips
        final_clip = mp.concatenate_videoclips(clips, method="compose")
        
        # Process audio tracks
        audio_tracks = []
        
        # Add voiceovers for each segment
        current_time = 0
        for segment in segments:
            if "voiceover_path" in segment:
                try:
                    voiceover = AudioFileClip(segment["voiceover_path"])
                    # Set the start time for this voiceover
                    voiceover = voiceover.set_start(current_time)
                    audio_tracks.append(voiceover)
                except Exception as e:
                    self.logger.warning(f"Failed to load voiceover: {e}")
            current_time += segment["duration"]
        
        # Add background music if specified
        if music and "path" in music:
            try:
                background_music = AudioFileClip(music["path"])
                if music.get("loop_count"):
                    background_music = background_music.loop(music["loop_count"])
                if music.get("fade_out"):
                    background_music = background_music.audio_fadeout(2)
                # Lower the volume of background music
                background_music = background_music.volumex(0.3)
                audio_tracks.append(background_music)
            except Exception as e:
                self.logger.warning(f"Failed to load background music: {e}")
        
        # Combine all audio tracks if we have any
        if audio_tracks:
            try:
                final_audio = CompositeAudioClip(audio_tracks)
                final_clip = final_clip.set_audio(final_audio)
            except Exception as e:
                self.logger.error(f"Failed to combine audio tracks: {e}")
        
        # Write final video with optimized settings
        # Use more compatible encoding settings
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            preset='medium',  # Better compatibility
            threads=4,
            bitrate='4000k',
            ffmpeg_params=[
                "-profile:v", "baseline",
                "-level", "3.0",
                "-pix_fmt", "yuv420p",  # Required for QuickTime
                "-movflags", "+faststart"  # Enable streaming
            ]
        )
        
        # Cleanup temp files
        for clip in clips:
            clip.close()
        
        return output_path
