import pytest
from pathlib import Path
import json
from src.director.ai.llm_manager import LLMManager
from src.director.ai.content_generator import ContentGenerator

@pytest.fixture
def llm_manager(tmp_path):
    """Create LLMManager instance with temporary config."""
    config_dir = tmp_path / "config" / "director"
    config_dir.mkdir(parents=True)
    return LLMManager(str(config_dir / "llm.yaml"))

@pytest.fixture
def content_generator(llm_manager):
    """Create ContentGenerator instance."""
    return ContentGenerator(llm_manager)

@pytest.fixture
def sample_listing():
    """Sample listing data for testing."""
    return {
        'title': 'Modern Downtown Loft',
        'price': 750000,
        'location': 'Downtown District',
        'features': [
            'Open floor plan',
            'Floor-to-ceiling windows',
            'Gourmet kitchen'
        ],
        'target_market': 'young professionals',
        'highlights': 'Prime location, luxury finishes'
    }

def test_llm_config_initialization(llm_manager):
    """Test LLM configuration initialization."""
    assert llm_manager.config is not None
    assert 'model' in llm_manager.config
    assert 'style_presets' in llm_manager.config
    assert 'prompt_templates' in llm_manager.config

def test_style_preset_retrieval(llm_manager):
    """Test style preset retrieval."""
    preset = llm_manager.get_style_preset('energetic')
    assert preset is not None
    assert 'tone' in preset
    assert 'pacing' in preset
    assert 'language' in preset

@pytest.mark.asyncio
async def test_video_style_selection(llm_manager, sample_listing):
    """Test video style selection."""
    style = await llm_manager.select_video_style(sample_listing)
    assert style in llm_manager.config['style_presets']

@pytest.mark.asyncio
async def test_script_generation(llm_manager, sample_listing):
    """Test script generation."""
    script_data = await llm_manager.generate_video_script(
        sample_listing,
        'energetic',
        60
    )
    
    assert 'script' in script_data
    assert 'segments' in script_data
    assert 'style_directions' in script_data
    
    # Verify segments structure
    assert len(script_data['segments']) > 0
    for segment in script_data['segments']:
        assert 'time' in segment
        assert 'content' in segment

@pytest.mark.asyncio
async def test_content_generation(content_generator, sample_listing):
    """Test complete content generation."""
    content = await content_generator.generate_video_content(
        sample_listing,
        60
    )
    
    # Verify all required content elements
    assert 'script' in content
    assert 'style' in content
    assert 'segments' in content
    assert 'directions' in content
    
    # Verify style settings
    assert 'preset' in content['style']
    assert 'settings' in content['style']
    
    # Verify directions
    assert 'transitions' in content['directions']
    assert 'effects' in content['directions']
    assert 'music' in content['directions']
    
    # Verify transitions
    transitions = content['directions']['transitions']
    assert len(transitions) == len(content['segments']) - 1
    
    # Verify effects
    effects = content['directions']['effects']
    assert len(effects) > 0
    for effect in effects:
        assert 'type' in effect
        assert 'intensity' in effect
        
    # Verify music
    music = content['directions']['music']
    assert 'genre' in music
    assert 'tempo' in music
    assert 'intensity' in music
