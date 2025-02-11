import pytest
import os
from pathlib import Path
from src.director.agents.voice_agent import VoiceAgent
from src.director.agents.music_agent import MusicAgent
from src.director.agents.style_agent import StyleAgent

@pytest.fixture
def voice_agent():
    """Create VoiceAgent instance."""
    return VoiceAgent()

@pytest.fixture
def music_agent(tmp_path):
    """Create MusicAgent instance with temporary directory."""
    return MusicAgent(str(tmp_path))

@pytest.fixture
def style_agent():
    """Create StyleAgent instance."""
    return StyleAgent()

def test_voice_agent_initialization(voice_agent):
    """Test voice agent initialization."""
    assert voice_agent.config is not None
    assert 'voices' in voice_agent.config
    assert 'default_voice' in voice_agent.config
    
    # Test voice configurations
    for voice_type in ['professional', 'friendly', 'luxury']:
        voice_config = voice_agent.config['voices'][voice_type]
        assert 'voice_id' in voice_config
        assert 'settings' in voice_config

def test_voice_settings_retrieval(voice_agent):
    """Test voice settings retrieval."""
    settings = voice_agent.get_voice_settings('professional')
    assert settings is not None
    assert 'voice_id' in settings
    assert 'settings' in settings
    assert 'stability' in settings['settings']
    assert 'similarity_boost' in settings['settings']

def test_music_agent_initialization(music_agent):
    """Test music agent initialization."""
    assert music_agent.catalog is not None
    assert 'genres' in music_agent.catalog
    assert 'moods' in music_agent.catalog
    assert 'tempos' in music_agent.catalog
    
    # Test catalog structure
    for genre in ['upbeat_electronic', 'corporate', 'ambient']:
        genre_info = music_agent.catalog['genres'][genre]
        assert 'mood' in genre_info
        assert 'tempo' in genre_info
        assert 'tracks' in genre_info

@pytest.mark.asyncio
async def test_music_selection(music_agent):
    """Test music selection."""
    selection = await music_agent.select_music('energetic', 60)
    assert selection is not None
    assert 'track' in selection
    assert 'genre' in selection
    assert 'mood' in selection
    assert 'tempo' in selection

def test_music_track_filtering(music_agent):
    """Test music track filtering."""
    tracks = music_agent.get_available_tracks(
        genre='upbeat_electronic',
        mood='energetic',
        tempo='fast'
    )
    assert len(tracks) > 0
    for track in tracks:
        assert 'title' in track
        assert 'duration' in track
        assert 'file' in track
        assert 'tags' in track

def test_style_agent_initialization(style_agent):
    """Test style agent initialization."""
    assert style_agent.config is not None
    assert 'styles' in style_agent.config
    assert 'variations' in style_agent.config
    
    # Test style configurations
    for style in ['energetic', 'professional', 'luxury']:
        style_config = style_agent.config['styles'][style]
        assert 'video' in style_config
        assert 'text' in style_config
        assert 'audio' in style_config

def test_style_retrieval(style_agent):
    """Test style retrieval."""
    style = style_agent.get_style('energetic')
    assert style is not None
    assert 'video' in style
    assert 'text' in style
    assert 'audio' in style

def test_style_adaptation(style_agent):
    """Test style adaptation based on property details."""
    # Test luxury property
    style = style_agent.adapt_style_to_content(
        'house',
        2000000,  # $2M property
        'Luxury buyers'
    )
    assert style == 'luxury'
    
    # Test young market
    style = style_agent.adapt_style_to_content(
        'apartment',
        500000,
        'Young professionals'
    )
    assert style == 'energetic'
    
    # Test default case
    style = style_agent.adapt_style_to_content(
        'house',
        400000,
        'Families'
    )
    assert style == 'professional'

def test_style_elements(style_agent):
    """Test getting specific style elements."""
    video_elements = style_agent.get_style_elements('luxury', 'video')
    assert video_elements is not None
    assert 'transitions' in video_elements
    assert 'effects' in video_elements
    assert 'pacing' in video_elements
    assert 'color_grade' in video_elements
