import pytest
from src.services.text_overlay import TextOverlay
import moviepy.editor as mp

@pytest.fixture
def text_overlay():
    return TextOverlay()

def test_create_property_details_overlay(text_overlay):
    # Test data
    properties = {
        "title": "Luxury Waterfront Villa",
        "price": "$2,500,000",
        "location": "Miami Beach, FL",
        "details": [
            "5 Bedrooms",
            "4.5 Bathrooms",
            "4,500 sqft",
            "Oceanfront"
        ]
    }
    
    # Create overlay
    duration = 3.0
    size = (1080, 1920)
    overlay = text_overlay.create_property_details_overlay(properties, duration, size)
    
    # Verify overlay properties
    assert isinstance(overlay, mp.CompositeVideoClip)
    assert overlay.duration == duration
    assert overlay.size == size
    assert len(overlay.clips) > 1  # Should have background + text clips

def test_create_property_details_overlay_minimal(text_overlay):
    # Test with minimal properties
    properties = {
        "title": "Test Property",
        "price": "$500,000"
    }
    
    # Create overlay
    overlay = text_overlay.create_property_details_overlay(properties, 3.0)
    
    # Verify overlay
    assert isinstance(overlay, mp.CompositeVideoClip)
    assert len(overlay.clips) > 1  # Should have at least background + some text

def test_text_clip_creation(text_overlay):
    # Test internal text clip creation
    clip = text_overlay._create_text_clip(
        "Test Text",
        font_size=50,
        color="white",
        position=(100, 100),
        duration=3.0
    )
    
    # Verify clip properties
    assert isinstance(clip, mp.CompositeVideoClip)  # Should be composite due to shadow
    assert clip.duration == 3.0

def test_text_clip_no_shadow(text_overlay):
    # Test text clip without shadow
    clip = text_overlay._create_text_clip(
        "Test Text",
        font_size=50,
        color="white",
        position=(100, 100),
        duration=3.0,
        with_shadow=False
    )
    
    # Verify it's a simple TextClip when no shadow
    assert isinstance(clip, mp.TextClip)
