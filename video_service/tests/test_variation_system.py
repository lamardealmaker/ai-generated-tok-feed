import pytest
from src.director.agents.text_overlay_agent import TextOverlayAgent
from src.director.agents.voice_agent import VoiceAgent
from src.director.tools.music_tool import MusicTool
from src.director.tools.video_db_tool import VideoDBTool

@pytest.fixture
def text_overlay_agent():
    return TextOverlayAgent()

@pytest.fixture
def music_tool():
    return MusicTool()

@pytest.fixture
def video_db_tool():
    return VideoDBTool()

@pytest.fixture
def voice_agent():
    return VoiceAgent()

def test_text_overlay_variations(text_overlay_agent):
    """Test text overlay style variations."""
    # Test modern style
    modern_style = text_overlay_agent.get_style_for_template("modern_rapid_standard")
    assert "title" in modern_style
    assert modern_style["title"]["font"] == "Helvetica Neue"
    
    # Test luxury style
    luxury_style = text_overlay_agent.get_style_for_template("luxury_balanced_standard")
    assert "title" in luxury_style
    assert luxury_style["title"]["font"] == "Didot"
    
    # Test minimal style
    minimal_style = text_overlay_agent.get_style_for_template("minimal_relaxed_standard")
    assert "title" in minimal_style
    assert minimal_style["title"]["font"] == "SF Pro Display"

def test_text_overlay_config(text_overlay_agent):
    """Test text overlay configuration generation."""
    text_content = {
        "title": "Beautiful Modern Home",
        "price": "$750,000",
        "features": "4 beds, 3 baths"
    }
    
    timing = {
        "title": 0.0,
        "title_duration": 4.0,
        "price": 1.0,
        "price_duration": 3.0,
        "features": 2.0,
        "features_duration": 3.0
    }
    
    overlays = text_overlay_agent.generate_overlay_config(
        "modern_rapid_standard",
        text_content,
        timing
    )
    
    assert len(overlays) == 3
    assert overlays[0]["content"] == "Beautiful Modern Home"
    assert overlays[0]["timing"]["start"] == 0.0
    assert overlays[0]["timing"]["duration"] == 4.0

def test_music_selection(music_tool):
    """Test music selection system."""
    # Test rapid pacing
    rapid_track = music_tool.select_music("modern_rapid_standard", 30.0)
    assert rapid_track["bpm"] >= 120  # Should be upbeat
    
    # Test relaxed pacing
    relaxed_track = music_tool.select_music("minimal_relaxed_standard", 30.0)
    assert relaxed_track["bpm"] <= 85  # Should be slower
    
    # Test with mood override
    energetic_track = music_tool.select_music(
        "modern_balanced_standard",
        30.0,
        mood_override="energetic"
    )
    assert energetic_track["mood"] == "energetic"

def test_video_transitions(video_db_tool):
    """Test video transition system."""
    # Test modern style
    modern_transitions = video_db_tool.get_transitions("modern_rapid_standard", count=3)
    assert len(modern_transitions) == 3
    assert all("duration" in t for t in modern_transitions)
    
    # Test luxury style
    luxury_transitions = video_db_tool.get_transitions("luxury_balanced_standard", count=2)
    assert len(luxury_transitions) == 2
    assert all("duration" in t for t in luxury_transitions)
    
    # Test with variation
    transitions = video_db_tool.get_transitions(
        "modern_rapid_standard",
        count=5,
        variation=0.3
    )
    assert len(transitions) == 5
    # Check that durations vary
    durations = [t["duration"] for t in transitions]
    assert len(set(durations)) > 1  # Should have some variation

def test_video_effects(video_db_tool):
    """Test video effects system."""
    # Test modern style
    modern_effects = video_db_tool.get_effects("modern_rapid_standard")
    assert "color" in modern_effects
    assert modern_effects["color"]["contrast"] > 1.0
    
    # Test luxury style
    luxury_effects = video_db_tool.get_effects("luxury_balanced_standard")
    assert "filters" in luxury_effects
    assert "cinematic" in luxury_effects["filters"]
    
    # Test with different intensities
    strong_effects = video_db_tool.get_effects("modern_rapid_standard", intensity=1.0)
    mild_effects = video_db_tool.get_effects("modern_rapid_standard", intensity=0.5)
    assert strong_effects["color"]["contrast"] > mild_effects["color"]["contrast"]

@pytest.mark.asyncio
async def test_voice_generation(voice_agent):
    """Test voice generation system."""
    # Skip if no API key
    if not voice_agent.api_key:
        pytest.skip("No ElevenLabs API key available")
    
    script = "Welcome to this beautiful modern home."
    
    # Test professional style
    pro_path = await voice_agent.generate_voiceover(
        script,
        style="professional"
    )
    assert pro_path.endswith(".mp3")
    
    # Test friendly style
    friendly_path = await voice_agent.generate_voiceover(
        script,
        style="friendly"
    )
    assert friendly_path.endswith(".mp3")
    
    # Paths should be different for different styles
    assert pro_path != friendly_path
