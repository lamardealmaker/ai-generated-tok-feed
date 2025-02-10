import pytest
import os
from src.services.audio_service import AudioService
from pydub import AudioSegment


@pytest.fixture
def audio_service():
    return AudioService()


@pytest.fixture
def sample_audio():
    # Create a simple test audio file
    audio = AudioSegment.silent(duration=5000)  # 5 seconds of silence
    temp_path = "test_audio.mp3"
    audio.export(temp_path, format="mp3")
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_prepare_background_music(audio_service, sample_audio):
    # Test with default duration
    output_path = audio_service.prepare_background_music(sample_audio)
    assert os.path.exists(output_path)
    
    # Verify the audio file properties
    audio = AudioSegment.from_file(output_path)
    assert len(audio) == audio_service.default_music_duration * 1000  # Convert to ms
    
    # Cleanup
    audio_service.cleanup_temp_files(output_path)


def test_prepare_background_music_custom_duration(audio_service, sample_audio):
    custom_duration = 15  # seconds
    output_path = audio_service.prepare_background_music(sample_audio, duration=custom_duration)
    
    # Verify duration
    audio = AudioSegment.from_file(output_path)
    assert len(audio) == custom_duration * 1000
    
    # Cleanup
    audio_service.cleanup_temp_files(output_path)


def test_cleanup_temp_files(audio_service):
    # Create a temporary file
    temp_audio = AudioSegment.silent(duration=1000)
    temp_path = "test_cleanup.mp3"
    temp_audio.export(temp_path, format="mp3")
    
    # Test cleanup
    assert os.path.exists(temp_path)
    audio_service.cleanup_temp_files(temp_path)
    assert not os.path.exists(temp_path)
