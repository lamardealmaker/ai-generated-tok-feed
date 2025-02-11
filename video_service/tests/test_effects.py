import pytest
import numpy as np
import cv2
from src.director.effects.transitions import TransitionLibrary
from src.director.effects.text_overlay import TextOverlay, TextTemplate
from src.director.effects.visual_effects import EffectLibrary, EffectChain

@pytest.fixture
def sample_frame():
    """Create a sample video frame."""
    return np.zeros((1080, 1920, 3), dtype=np.uint8)

@pytest.fixture
def transition_library():
    """Create TransitionLibrary instance."""
    return TransitionLibrary()

@pytest.fixture
def text_overlay():
    """Create TextOverlay instance."""
    return TextOverlay()

@pytest.fixture
def text_template():
    """Create TextTemplate instance."""
    return TextTemplate()

@pytest.fixture
def effect_library():
    """Create EffectLibrary instance."""
    return EffectLibrary()

def test_transition_library_initialization(transition_library):
    """Test transition library initialization."""
    assert len(transition_library.list_transitions()) > 0
    
    # Test transition retrieval
    transition = transition_library.get_transition('dissolve')
    assert transition is not None
    
    # Test transition info
    info = transition_library.get_transition_info('dissolve')
    assert 'name' in info
    assert 'description' in info
    assert 'parameters' in info

@pytest.mark.asyncio
async def test_transition_effects(transition_library, sample_frame):
    """Test applying transition effects."""
    frame1 = sample_frame.copy()
    frame2 = np.ones_like(frame1) * 255
    
    # Test dissolve transition
    dissolve = transition_library.get_transition('dissolve')
    result = await dissolve.apply(frame1, frame2, 0.5)
    assert result is not None
    assert result.shape == frame1.shape
    
    # Test slide transition
    slide = transition_library.get_transition('slide_left')
    result = await slide.apply(frame1, frame2, 0.5)
    assert result is not None
    assert result.shape == frame1.shape

def test_text_overlay_initialization(text_overlay):
    """Test text overlay initialization."""
    assert text_overlay.style is not None

@pytest.mark.asyncio
async def test_text_rendering(text_overlay, sample_frame):
    """Test text rendering."""
    text = "Test Text"
    position = (0.5, 0.5)
    
    result = await text_overlay.render_text(
        sample_frame,
        text,
        position,
        progress=1.0
    )
    
    assert result is not None
    assert result.shape == sample_frame.shape

def test_text_template_initialization(text_template):
    """Test text template initialization."""
    assert len(text_template.list_templates()) > 0
    
    # Test template retrieval
    template = text_template.get_template('price')
    assert template is not None
    assert 'style' in template
    assert 'position' in template

def test_effect_library_initialization(effect_library):
    """Test effect library initialization."""
    assert len(effect_library.list_effects()) > 0
    
    # Test effect retrieval
    effect = effect_library.get_effect('color_enhance')
    assert effect is not None
    
    # Test effect info
    info = effect_library.get_effect_info('color_enhance')
    assert 'name' in info
    assert 'description' in info
    assert 'parameters' in info

@pytest.mark.asyncio
async def test_visual_effects(effect_library, sample_frame):
    """Test applying visual effects."""
    # Test color enhance
    color_effect = effect_library.get_effect('color_enhance')
    result = await color_effect.apply(sample_frame)
    assert result is not None
    assert result.shape == sample_frame.shape
    
    # Test vignette
    vignette = effect_library.get_effect('vignette')
    result = await vignette.apply(sample_frame)
    assert result is not None
    assert result.shape == sample_frame.shape

@pytest.mark.asyncio
async def test_effect_chain(effect_library, sample_frame):
    """Test effect chain."""
    chain = EffectChain()
    
    # Add multiple effects
    chain.add_effect('color_enhance', 0.5)
    chain.add_effect('vignette', 0.7)
    
    # Apply chain
    result = await chain.apply(sample_frame)
    assert result is not None
    assert result.shape == sample_frame.shape
    
    # Test chain clear
    chain.clear()
    assert len(chain.effects) == 0
