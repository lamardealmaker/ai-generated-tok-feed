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
    
    def apply_effects(self, clip: VideoFileClip, effects: Dict[str, Any], is_first: bool = False) -> VideoFileClip:
        """Apply video effects to a clip."""
        # Skip effects for first clip
        if is_first:
            return clip
            
        # Apply color adjustments
        if "color" in effects:
            color = effects["color"]
            if "contrast" in color:
                clip = clip.fx(mp.vfx.colorx, color["contrast"])
            if "brightness" in color:
                clip = clip.fx(mp.vfx.colorx, color["brightness"])
            if "saturation" in color:
                clip = clip.fx(mp.vfx.colorx, color["saturation"])
        
        # Apply filters
        if "filters" in effects:
            for filter_name in effects["filters"]:
                if filter_name == "vibrant":
                    clip = clip.fx(mp.vfx.colorx, 1.2)
                elif filter_name == "warm":
                    clip = clip.fx(mp.vfx.colorx, 1.1)
                elif filter_name == "cinematic":
                    clip = clip.fx(mp.vfx.lum_contrast)
        
        # Apply transitions if specified
        if "transition" in effects:
            transition = effects["transition"]
            if transition["name"] == "fade":
                clip = clip.crossfadein(transition["duration"])
            elif transition["name"] == "slide":
                # TODO: Implement slide transition
                pass
            
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
            
            # Apply effects (pass is_first flag)
            if "effects" in segment:
                is_first = (segment == segments[0])
                clip = self.apply_effects(clip, segment["effects"], is_first)
            
            # Add text overlay
            if "text" in segment:
                text_clip = self.create_text_overlay(segment["text"], segment["duration"])
                clip = CompositeVideoClip([clip, text_clip])
            
            clips.append(clip)
        
        # Concatenate all clips with no transitions
        final_clip = mp.concatenate_videoclips(clips, method="compose")
        
        # Get total duration from segments
        total_duration = sum(segment["duration"] for segment in segments)
        # Ensure final clip duration matches segments
        final_clip = final_clip.set_duration(total_duration)
        
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
                # Trim or loop music to match video duration
                total_duration = sum(segment["duration"] for segment in segments)
                if background_music.duration > total_duration:
                    background_music = background_music.subclip(0, total_duration)
                elif background_music.duration < total_duration:
                    # Calculate how many loops we need
                    loops_needed = int(total_duration / background_music.duration) + 1
                    loops = [background_music] * loops_needed
                    background_music = concatenate_audioclips(loops)
                    background_music = background_music.subclip(0, total_duration)
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
