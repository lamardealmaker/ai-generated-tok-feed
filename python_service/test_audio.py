from moviepy.editor import ColorClip, AudioFileClip
import tempfile

def test_audio():
    # Create a simple color clip
    color_clip = ColorClip(size=(640, 480), color=(0, 0, 0), duration=10)
    
    # Load the audio
    audio = AudioFileClip("test_music.mp3").set_duration(10)
    
    # Combine video and audio
    final_clip = color_clip.set_audio(audio)
    
    # Write to file
    final_clip.write_videofile(
        "test_output.mp4",
        codec='libx264',
        audio_codec='aac',
        audio=True,
        fps=24,
        audio_bitrate='192k'
    )
    
    # Cleanup
    color_clip.close()
    audio.close()

if __name__ == "__main__":
    test_audio()
