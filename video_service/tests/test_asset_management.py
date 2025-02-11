import pytest
import os
from pathlib import Path
import shutil
from src.director.assets.manager import AssetManager
from src.director.assets.videodb import VideoDBToolWrapper

@pytest.fixture
def asset_manager(tmp_path):
    """Create AssetManager instance with temporary directory."""
    return AssetManager(str(tmp_path))

@pytest.fixture
def videodb_tool():
    """Create VideoDBToolWrapper instance."""
    config = {
        'resolution': (1080, 1920),
        'fps': 30,
        'quality': 'high'
    }
    return VideoDBToolWrapper(config)

def test_directory_structure(asset_manager):
    """Test that all required directories are created."""
    expected_dirs = [
        'images/original',
        'images/processed',
        'audio/music',
        'audio/voiceover',
        'video/clips',
        'video/final',
        'temp'
    ]
    
    for dir_path in expected_dirs:
        assert (Path(asset_manager.base_path) / dir_path).exists()

@pytest.mark.asyncio
async def test_temp_directory_management(asset_manager):
    """Test temporary directory creation and cleanup."""
    session_id = "test_session"
    
    # Create temp directory
    temp_dir = await asset_manager.create_temp_directory(session_id)
    assert temp_dir.exists()
    
    # Create a test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("test")
    
    # Clean up
    await asset_manager.cleanup_temp_files(session_id)
    assert not temp_dir.exists()

@pytest.mark.asyncio
async def test_voiceover_storage(asset_manager):
    """Test storing voiceover audio."""
    session_id = "test_session"
    audio_data = b"test_audio_data"
    metadata = {
        "duration": 10,
        "format": "mp3",
        "sample_rate": 44100
    }
    
    # Store voiceover
    filepath = await asset_manager.store_voiceover(session_id, audio_data, metadata)
    
    # Check file exists
    assert Path(filepath).exists()
    
    # Check metadata file exists
    meta_path = Path(filepath).with_suffix('.json')
    assert meta_path.exists()

@pytest.mark.asyncio
async def test_final_video_storage(asset_manager):
    """Test storing final video."""
    session_id = "test_session"
    
    # Create a dummy video file
    temp_video = Path(asset_manager.base_path) / "temp.mp4"
    temp_video.write_bytes(b"test_video_data")
    
    metadata = {
        "duration": 60,
        "resolution": "1080p",
        "fps": 30
    }
    
    # Store video
    filepath = await asset_manager.store_final_video(
        session_id,
        str(temp_video),
        metadata
    )
    
    # Check file exists
    assert Path(filepath).exists()
    
    # Check metadata file exists
    meta_path = Path(filepath).with_suffix('.json')
    assert meta_path.exists()
    
    # Cleanup
    temp_video.unlink()

@pytest.mark.asyncio
async def test_videodb_tool_initialization(videodb_tool):
    """Test VideoDBTool wrapper initialization."""
    assert videodb_tool.config['resolution'] == (1080, 1920)
    assert videodb_tool.config['fps'] == 30
    assert videodb_tool.config['quality'] == 'high'
