import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.services.video_service import VideoService
from src.services.image_processor import ImageProcessor
from fastapi import UploadFile
import io
import moviepy.editor as mp

@pytest.fixture
def mock_firebase():
    """Mock Firebase services"""
    with patch('firebase_admin.storage.bucket') as mock_bucket, \
         patch('firebase_admin.firestore.client') as mock_firestore:
        # Setup mock document
        mock_doc = Mock()
        mock_doc.exists = True
        mock_doc.to_dict.return_value = {
            'status': 'completed',
            'video_url': 'https://example.com/video.mp4',
            'created_at': '2024-02-10T10:00:00Z',
            'completed_at': '2024-02-10T10:01:00Z'
        }
        
        # Setup mock collection and document references
        mock_firestore.return_value.collection.return_value.document.return_value.get.return_value = mock_doc
        
        yield {
            'bucket': mock_bucket.return_value,
            'firestore': mock_firestore.return_value
        }

@pytest.fixture
def video_service(mock_firebase):
    return VideoService()

@pytest.fixture
def mock_image_processor():
    with patch('src.services.video_service.ImageProcessor') as mock:
        processor = Mock(spec=ImageProcessor)
        processor.create_slides = AsyncMock(return_value=[
            mp.ColorClip(size=(1080, 1920), color=(0, 0, 0), duration=3)
        ])
        mock.return_value = processor
        yield processor

@pytest.mark.asyncio
async def test_generate_video(video_service, mock_image_processor):
    # Test data
    properties = {
        "title": "Luxury Waterfront Villa",
        "price": "$2,500,000",
        "location": "Miami Beach, FL",
        "images": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"],
        "details": [
            "5 Bedrooms",
            "4.5 Bathrooms",
            "4,500 sqft",
            "Oceanfront"
        ]
    }
    
    # Create a mock file for background music
    mock_file = io.BytesIO(b"mock audio content")
    background_music = UploadFile(filename="test.mp3", file=mock_file)
    
    # Mock Firebase upload
    video_service._upload_to_firebase = AsyncMock(
        return_value="https://storage.googleapis.com/video.mp4"
    )
    
    # Test video generation
    result = await video_service.generate_video(properties, background_music)
    
    assert "video_id" in result
    assert "status" in result
    assert result["status"] == "completed"
    assert "video_url" in result
    
    # Verify image processor was called
    mock_image_processor.create_slides.assert_called_once_with(
        properties["images"]
    )

@pytest.mark.asyncio
async def test_generate_video_missing_fields(video_service):
    # Test cases for missing required fields
    test_cases = [
        {
            "properties": {"price": "$500,000", "location": "Test Location"},
            "missing": "title, images"
        },
        {
            "properties": {"title": "Test", "location": "Test Location"},
            "missing": "price, images"
        },
        {
            "properties": {"title": "Test", "price": "$500,000"},
            "missing": "location, images"
        }
    ]
    
    for case in test_cases:
        with pytest.raises(ValueError, match=f"Missing required properties: {case['missing']}"):
            await video_service.generate_video(case['properties'])

@pytest.mark.asyncio
async def test_get_video_status(video_service):
    # Test getting video status
    video_id = "test_video_id"
    result = await video_service.get_video_status(video_id)
    
    assert result["video_id"] == video_id
    assert result["status"] == "completed"
    assert "video_url" in result
    assert "created_at" in result
    assert "completed_at" in result

@pytest.mark.asyncio
async def test_get_video_status_not_found(video_service, mock_firebase):
    # Mock document not found
    mock_firebase['firestore'].collection.return_value.document.return_value.get.return_value.exists = False
    
    # Test getting non-existent video status
    with pytest.raises(Exception, match="Video with id test_not_found not found"):
        await video_service.get_video_status("test_not_found")
