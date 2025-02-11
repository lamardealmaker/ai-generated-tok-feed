import pytest
import os
from src.director.tools.voiceover_tool import VoiceoverTool
from src.director.tools.music_library_tool import MusicLibraryTool

@pytest.fixture
def voiceover_tool():
    # Use mock mode for testing
    return VoiceoverTool({"mock_mode": True})

@pytest.fixture
def music_tool():
    return MusicLibraryTool()

@pytest.fixture
def sample_property_data():
    return {
        "price": 1500000,
        "features": ["luxury finishes", "modern design"],
        "amenities": ["pool", "spa"]
    }

def test_voiceover_capabilities(voiceover_tool):
    """Test voiceover tool capabilities."""
    capabilities = voiceover_tool.get_capabilities()
    
    assert "name" in capabilities
    assert "description" in capabilities
    assert "supported_voices" in capabilities
    assert "max_text_length" in capabilities
    assert "supported_formats" in capabilities
    
    assert "luxury" in capabilities["supported_voices"]
    assert "professional" in capabilities["supported_voices"]

def test_voiceover_generation(voiceover_tool):
    """Test voiceover generation."""
    result = voiceover_tool.generate_voiceover(
        text="Welcome to this luxury property",
        style="luxury",
        output_format="mp3"
    )
    
    assert "status" in result
    assert "duration" in result
    assert "file_path" in result
    assert "voice_id" in result
    assert "settings" in result
    
    assert result["status"] == "success"
    assert result["duration"] > 0
    assert result["voice_id"] == "premium_male_1"

def test_music_capabilities(music_tool):
    """Test music library tool capabilities."""
    capabilities = music_tool.get_capabilities()
    
    assert "name" in capabilities
    assert "description" in capabilities
    assert "supported_styles" in capabilities
    assert "features" in capabilities
    
    assert "luxury" in capabilities["supported_styles"]
    assert "modern" in capabilities["supported_styles"]
    assert "track_selection" in capabilities["features"]

def test_track_selection(music_tool):
    """Test music track selection."""
    result = music_tool.select_track(
        style="luxury",
        duration=30.0,
        mood="sophisticated"
    )
    
    assert "track" in result
    assert "start_time" in result
    assert "end_time" in result
    assert "fade_in" in result
    assert "fade_out" in result
    
    assert result["start_time"] == 0.0
    assert result["end_time"] == 30.0
    assert result["track"]["mood"] == "sophisticated"

def test_transition_generation(music_tool):
    """Test music transition generation."""
    result = music_tool.generate_transition(
        style="modern",
        duration=1.0
    )
    
    assert "name" in result
    assert "duration" in result
    assert "fade_curve" in result
    
    assert result["duration"] <= 1.0
    assert result["fade_curve"] == "smooth"

def test_mood_analysis(music_tool, sample_property_data):
    """Test property mood analysis."""
    mood = music_tool.analyze_mood(sample_property_data)
    
    assert mood == "sophisticated"
    
    # Test with different property data
    budget_property = {
        "price": 300000,
        "features": ["updated kitchen"]
    }
    assert music_tool.analyze_mood(budget_property) == "neutral"

def test_error_handling(music_tool, voiceover_tool):
    """Test error handling in tools."""
    # Test invalid style
    with pytest.raises(ValueError):
        music_tool.select_track(
            style="nonexistent",
            duration=30.0
        )
        
    # Test invalid text length
    with pytest.raises(ValueError):
        voiceover_tool.generate_voiceover(
            text="a" * 6000,  # Exceeds max length
            style="professional"
        )
