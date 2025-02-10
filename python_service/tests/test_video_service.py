import pytest
from src.services.video_service import VideoService
from fastapi import UploadFile
import io

@pytest.fixture
def video_service():
    return VideoService()

@pytest.mark.asyncio
async def test_generate_video(video_service):
    # Test data
    properties = {
        "title": "Test Property",
        "price": "$500,000",
        "location": "Test Location",
        "images": ["test_image_1.jpg", "test_image_2.jpg"]
    }
    
    # Create a mock file for background music
    mock_file = io.BytesIO(b"mock audio content")
    background_music = UploadFile(filename="test.mp3", file=mock_file)
    
    # Test video generation
    result = await video_service.generate_video(properties, background_music)
    
    assert "video_id" in result
    assert "status" in result
    assert result["status"] == "processing"

@pytest.mark.asyncio
async def test_get_video_status(video_service):
    # Test getting video status
    video_id = "test_video_id"
    result = await video_service.get_video_status(video_id)
    
    assert "video_id" in result
    assert "status" in result
