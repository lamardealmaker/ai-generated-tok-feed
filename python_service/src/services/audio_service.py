from typing import Optional
import os
import tempfile
from moviepy.editor import AudioFileClip, vfx


class AudioService:
    def __init__(self):
        self.default_music_duration = 30  # default duration in seconds

    def prepare_background_music(
        self,
        music_path: str,
        duration: Optional[float] = None,
        volume_reduction: float = 1.0  # Full volume
    ) -> str:
        """
        Prepare background music for the video by:
        1. Adjusting its length to match video duration
        2. Adding fade in/out effects
        3. Adjusting volume
        
        Args:
            music_path: Path to the music file
            duration: Desired duration in seconds
            volume_reduction: Volume multiplier (between 0 and 1)
        
        Returns:
            Path to the processed audio file
        """
        # Load the audio file
        audio = AudioFileClip(music_path)
        
        # Set duration
        target_duration = duration or self.default_music_duration
        
        # If audio is shorter than target duration, loop it
        if audio.duration < target_duration:
            repeats = int(target_duration / audio.duration) + 1
            # Create a list of the same clip repeated
            clips = [audio] * repeats
            # Concatenate all clips
            from moviepy.editor import concatenate_audioclips
            audio = concatenate_audioclips(clips)
        
        # Trim to exact duration
        audio = audio.set_duration(target_duration)
        
        # Adjust volume
        audio = audio.volumex(volume_reduction)
        
        # Export to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        audio.write_audiofile(temp_file.name)
        audio.close()
        
        return temp_file.name

    def cleanup_temp_files(self, *file_paths: str):
        """Delete temporary audio files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up {file_path}: {str(e)}")

    def create_audio_clip(self, audio_path: str) -> AudioFileClip:
        """Create a MoviePy AudioFileClip from an audio file"""
        return AudioFileClip(audio_path)
