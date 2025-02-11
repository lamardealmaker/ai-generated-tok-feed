# Video Service Codebase

## Statistics
- Total Python Files: 125
- Total Lines of Code: 15,403
- Total Words: 45,940
- Average Lines per File: 123.2

## File Listing
- ./test_moviepy.py
- ./test_video_gen.py
- ./tests/test_ai_variations.py
- ./tests/test_variation.py
- ./tests/test_effects.py
- ./tests/test_ai_integration.py
- ./tests/test_custom_tools.py
- ./tests/test_variation_system.py
- ./tests/test_asset_management.py
- ./tests/test_custom_agents.py
- ./tests/test_api.py
- ./tests/test_director_core.py
- ./tests/test_template_count.py
- ./tests/test_agents.py
- ./generate_property_video.py
- ./vendor/Director/backend/director/tools/kling.py
- ./vendor/Director/backend/director/tools/serp.py
- ./vendor/Director/backend/director/tools/composio_tool.py
- ./vendor/Director/backend/director/tools/replicate.py
- ./vendor/Director/backend/director/tools/videodb_tool.py
- ./vendor/Director/backend/director/tools/stabilityai.py
- ./vendor/Director/backend/director/tools/fal_video.py
- ./vendor/Director/backend/director/tools/elevenlabs.py
- ./vendor/Director/backend/director/tools/slack.py
- ./vendor/Director/backend/director/llm/videodb_proxy.py
- ./vendor/Director/backend/director/llm/__init__.py
- ./vendor/Director/backend/director/llm/openai.py
- ./vendor/Director/backend/director/llm/anthropic.py
- ./vendor/Director/backend/director/llm/base.py
- ./vendor/Director/backend/director/core/session.py
- ./vendor/Director/backend/director/core/__init__.py
- ./vendor/Director/backend/director/core/reasoning.py
- ./vendor/Director/backend/director/handler.py
- ./vendor/Director/backend/director/constants.py
- ./vendor/Director/backend/director/__init__.py
- ./vendor/Director/backend/director/agents/image_generation.py
- ./vendor/Director/backend/director/agents/slack_agent.py
- ./vendor/Director/backend/director/agents/transcription.py
- ./vendor/Director/backend/director/agents/editing.py
- ./vendor/Director/backend/director/agents/index.py
- ./vendor/Director/backend/director/agents/upload.py
- ./vendor/Director/backend/director/agents/meme_maker.py
- ./vendor/Director/backend/director/agents/download.py
- ./vendor/Director/backend/director/agents/thumbnail.py
- ./vendor/Director/backend/director/agents/__init__.py
- ./vendor/Director/backend/director/agents/web_search_agent.py
- ./vendor/Director/backend/director/agents/subtitle.py
- ./vendor/Director/backend/director/agents/profanity_remover.py
- ./vendor/Director/backend/director/agents/text_to_movie.py
- ./vendor/Director/backend/director/agents/comparison.py
- ./vendor/Director/backend/director/agents/dubbing.py
- ./vendor/Director/backend/director/agents/audio_generation.py
- ./vendor/Director/backend/director/agents/composio.py
- ./vendor/Director/backend/director/agents/brandkit.py
- ./vendor/Director/backend/director/agents/prompt_clip.py
- ./vendor/Director/backend/director/agents/pricing.py
- ./vendor/Director/backend/director/agents/summarize_video.py
- ./vendor/Director/backend/director/agents/sample.py
- ./vendor/Director/backend/director/agents/search.py
- ./vendor/Director/backend/director/agents/stream_video.py
- ./vendor/Director/backend/director/agents/video_generation.py
- ./vendor/Director/backend/director/agents/base.py
- ./vendor/Director/backend/director/utils/asyncio.py
- ./vendor/Director/backend/director/utils/__init__.py
- ./vendor/Director/backend/director/utils/exceptions.py
- ./vendor/Director/backend/director/entrypoint/__init__.py
- ./vendor/Director/backend/director/entrypoint/api/server.py
- ./vendor/Director/backend/director/entrypoint/api/__init__.py
- ./vendor/Director/backend/director/entrypoint/api/socket_io.py
- ./vendor/Director/backend/director/entrypoint/api/errors.py
- ./vendor/Director/backend/director/entrypoint/api/routes.py
- ./vendor/Director/backend/director/db/__init__.py
- ./vendor/Director/backend/director/db/sqlite/db.py
- ./vendor/Director/backend/director/db/sqlite/initialize.py
- ./vendor/Director/backend/director/db/sqlite/__init__.py
- ./vendor/Director/backend/director/db/postgres/db.py
- ./vendor/Director/backend/director/db/postgres/initialize.py
- ./vendor/Director/backend/director/db/base.py
- ./vendor/Director/docs/hooks/copyright.py
- ./src/director/tools/music_tool.py
- ./src/director/tools/voiceover_tool.py
- ./src/director/tools/video_db_tool.py
- ./src/director/tools/video_renderer.py
- ./src/director/tools/music_library_tool.py
- ./src/director/core/reasoning_engine.py
- ./src/director/core/task.py
- ./src/director/core/config.py
- ./src/director/core/base_agent.py
- ./src/director/core/session_manager.py
- ./src/director/core/session.py
- ./src/director/core/__init__.py
- ./src/director/core/base_tool.py
- ./src/director/core/engine.py
- ./src/director/variation/style_engine.py
- ./src/director/variation/template_engine.py
- ./src/director/variation/__init__.py
- ./src/director/variation/pacing_engine.py
- ./src/director/__init__.py
- ./src/director/agents/pacing_agent.py
- ./src/director/agents/music_agent.py
- ./src/director/agents/style_variation_agent.py
- ./src/director/agents/script_agent.py
- ./src/director/agents/__init__.py
- ./src/director/agents/real_estate_video_agent.py
- ./src/director/agents/voice_agent.py
- ./src/director/agents/style_agent.py
- ./src/director/agents/text_overlay_agent.py
- ./src/director/agents/effect_agent.py
- ./src/director/ai/__init__.py
- ./src/director/ai/llm_manager.py
- ./src/director/ai/content_generator.py
- ./src/director/video_generator.py
- ./src/director/effects/__init__.py
- ./src/director/effects/visual_effects.py
- ./src/director/effects/transitions.py
- ./src/director/effects/text_overlay.py
- ./src/director/assets/videodb.py
- ./src/director/assets/__init__.py
- ./src/director/assets/manager.py
- ./src/__init__.py
- ./src/utils/logging.py
- ./src/api/models.py
- ./src/api/endpoints.py
- ./src/api/routes.py
- ./src/main.py
### ./test_moviepy.py
```python
try:
    import moviepy.editor as mp
    print("MoviePy imported successfully!")
except Exception as e:
    print(f"Error importing MoviePy: {str(e)}")
    import sys
    print(f"Python path: {sys.path}")
```

### ./test_video_gen.py
```python
import asyncio
import os
from dotenv import load_dotenv
from src.director.core.reasoning_engine import ReasoningEngine
from src.utils.logging import setup_logger

# Load environment variables
load_dotenv()

# Set up logger
logger = setup_logger("test_video_gen")

async def test_video_generation():
    try:
        # Initialize the engine
        engine = ReasoningEngine()
        
        # Firebase image URLs (using fewer images for faster testing)
        image_urls = [
            "gs://project3-ziltok.firebasestorage.app/test_properties/house1.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/interior1.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/backyard.jpg"
        ]
        
        # Sample property data
        listing_data = {
            "id": "test-luxury-home-1",
            "title": "Luxurious Modern Estate",
            "price": 1250000,
            "description": "Stunning modern estate with breathtaking views. This masterpiece of design features an open concept living space, gourmet kitchen, and resort-style backyard.",
            "features": [
                "Gourmet Kitchen with High-End Appliances",
                "Open Concept Floor Plan",
                "Resort-Style Backyard with Pool",
                "Smart Home Technology",
                "Hardwood Floors Throughout"
            ],
            "location": "Beverly Hills",
            "bedrooms": 5,
            "bathrooms": 4.5,
            "sqft": 4200,
            "year_built": 2022,
            "lot_size": 12000
        }
        
        # Define progress callback
        def progress_callback(progress: float, message: str):
            logger.info(f"Progress: {progress*100:.1f}% - {message}")
        
        logger.info("Starting video generation test...")
        
        # Generate video
        task = await engine.generate_video(
            listing_data=listing_data,
            image_urls=image_urls,
            style="luxury",
            duration=15.0,  # Shorter duration for testing
            progress_callback=progress_callback
        )
        
        logger.info(f"Video generation task created with ID: {task.id}")
        
        # Wait for completion
        while task.status == "processing":
            logger.info(f"Task progress: {task.progress*100:.1f}%")
            await asyncio.sleep(2)
        
        if task.status == "completed":
            logger.info("Video generation completed successfully!")
            logger.info(f"Result: {task.result}")
        else:
            logger.error(f"Video generation failed: {task.error}")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(test_video_generation())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
```

### ./tests/test_ai_variations.py
```python
import pytest
from src.director.agents.script_agent import ScriptAgent
from src.director.agents.pacing_agent import PacingAgent
from src.director.agents.effect_agent import EffectAgent

@pytest.fixture
def script_agent():
    return ScriptAgent()

@pytest.fixture
def pacing_agent():
    return PacingAgent()

@pytest.fixture
def effect_agent():
    return EffectAgent()

@pytest.fixture
def sample_property_data():
    return {
        "location": "Beverly Hills",
        "price": 2500000,
        "bedrooms": 5,
        "bathrooms": 4,
        "sqft": 4500,
        "amenities": ["pool", "spa", "smart_home"],
        "features": ["gourmet kitchen", "home theater", "smart_home"],
        "style": "modern contemporary"
    }

def test_script_generation(script_agent, sample_property_data):
    """Test dynamic script generation with duration limits and segment adjustments."""
    # Test minimum duration (should be capped at 10s)
    short_script = script_agent.generate_script(
        property_data=sample_property_data,
        style="luxury",
        duration=5.0,  # Too short, should be adjusted to 10s
        target_segments=5
    )
    assert short_script["duration"] == script_agent.MIN_DURATION
    assert len(short_script["segments"]) <= 3  # Short videos get fewer segments
    
    # Test maximum duration (should be capped at 60s)
    long_script = script_agent.generate_script(
        property_data=sample_property_data,
        style="luxury",
        duration=90.0,  # Too long, should be adjusted to 60s
        target_segments=5
    )
    assert long_script["duration"] == script_agent.MAX_DURATION
    assert len(long_script["segments"]) >= 7  # Long videos get more segments
    
    # Test normal duration
    normal_script = script_agent.generate_script(
        property_data=sample_property_data,
        style="luxury",
        duration=30.0,
        target_segments=5
    )
    assert normal_script["duration"] == 30.0
    assert len(normal_script["segments"]) == 5
    
    # Verify segment structure and content
    for script in [short_script, normal_script, long_script]:
        first_segment = script["segments"][0]
        assert "type" in first_segment
        assert "text" in first_segment
        assert "start_time" in first_segment
        assert "duration" in first_segment
        
        # Verify script adapts to property data
        assert "Beverly Hills" in first_segment["text"]
        assert any("pool" in s["text"] for s in script["segments"])
        
        # Verify timing
        total_duration = sum(s["duration"] for s in script["segments"])
        assert abs(total_duration - script["duration"]) < 0.1  # Allow small floating-point difference

def test_script_pacing_adjustment(script_agent, sample_property_data):
    """Test script pacing adjustment."""
    script = script_agent.generate_script(
        property_data=sample_property_data,
        duration=30.0
    )
    
    engagement_data = {
        "hook": 0.8,
        "feature": 0.6,
        "closing": 0.4
    }
    
    adjusted_script = script_agent.adjust_script_pacing(
        script=script,
        engagement_data=engagement_data
    )
    
    # Verify timing adjustments
    assert abs(sum(s["duration"] for s in adjusted_script["segments"]) - 30.0) < 0.1
    
    # Verify high-engagement segments get more time
    hook_segment = next(s for s in adjusted_script["segments"] if s["type"] == "hook")
    closing_segment = next(s for s in adjusted_script["segments"] if s["type"] == "closing")
    assert hook_segment["duration"] > closing_segment["duration"]

def test_pacing_analysis(pacing_agent):
    """Test content-based pacing analysis."""
    segments = [
        {
            "content_type": "exterior",
            "text": "Welcome to this stunning home",
            "features": ["architecture", "landscaping"]
        },
        {
            "content_type": "feature",
            "text": "Check out this amazing kitchen",
            "features": ["appliances", "countertops", "island"],
            "data_points": ["size", "brand", "materials"]
        }
    ]
    
    analysis = pacing_agent.analyze_content(segments)
    
    assert "complexity" in analysis
    assert "visual_interest" in analysis
    assert "information_density" in analysis
    assert all(0 <= v <= 1 for v in analysis.values())

def test_pacing_optimization(pacing_agent):
    """Test pacing optimization."""
    segments = [
        {
            "content_type": "exterior",
            "duration": 5.0,
            "start_time": 0.0
        },
        {
            "content_type": "feature",
            "duration": 5.0,
            "start_time": 5.0
        }
    ]
    
    engagement_data = {
        "exterior": 0.7,
        "feature": 0.9
    }
    
    optimized = pacing_agent.optimize_engagement(
        segments=segments,
        engagement_data=engagement_data
    )
    
    # Verify feature segment gets more time due to higher engagement
    feature_segment = next(s for s in optimized if s["content_type"] == "feature")
    exterior_segment = next(s for s in optimized if s["content_type"] == "exterior")
    assert feature_segment["duration"] > exterior_segment["duration"]
    
    # Verify total duration remains the same
    original_duration = sum(s["duration"] for s in segments)
    optimized_duration = sum(s["duration"] for s in optimized)
    assert abs(original_duration - optimized_duration) < 0.1

def test_effect_generation(effect_agent):
    """Test effect combination generation."""
    effects = effect_agent.generate_effect_combination(
        style="modern",
        content_type="exterior",
        intensity=1.0
    )
    
    assert "color" in effects
    assert "filter" in effects
    assert "overlay" in effects
    assert isinstance(effects["overlay"], list)
    
    # Verify style-specific effects
    color = effects["color"]
    assert 1.0 < color["contrast"] <= 1.4  # Modern style has higher contrast
    
    # Verify content-specific adjustments
    assert color["contrast"] > 1.0  # Exterior shots get contrast boost

def test_effect_blending(effect_agent):
    """Test effect combination blending."""
    effects_a = effect_agent.generate_effect_combination(
        style="modern",
        content_type="exterior"
    )
    
    effects_b = effect_agent.generate_effect_combination(
        style="luxury",
        content_type="interior"
    )
    
    blended = effect_agent.blend_effects(
        effects_a=effects_a,
        effects_b=effects_b,
        blend_factor=0.5
    )
    
    assert "color" in blended
    assert "filter" in blended
    assert "overlay" in blended
    
    # Verify proper blending of numeric values
    assert all(isinstance(v, (int, float)) for v in blended["color"].values())
    if effects_a["filter"] and effects_b["filter"]:
        assert all(isinstance(v, (int, float)) for v in blended["filter"].values())
```

### ./tests/test_variation.py
```python
import pytest
import numpy as np
from src.director.variation.template_engine import TemplateLibrary, VideoTemplate
from src.director.variation.pacing_engine import PacingEngine, PacingCurve
from src.director.variation.style_engine import StyleEngine, StyleCombination

@pytest.fixture
def template_library():
    """Create TemplateLibrary instance."""
    return TemplateLibrary()

@pytest.fixture
def pacing_engine():
    """Create PacingEngine instance."""
    return PacingEngine()

@pytest.fixture
def style_engine():
    """Create StyleEngine instance."""
    return StyleEngine()

def test_template_library_initialization(template_library):
    """Test template library initialization and variations."""
    templates = template_library.list_templates()
    
    # Verify we have a large number of templates
    assert len(templates) > 200  # Should have at least 225 base templates
    
    # Test different style categories
    style_categories = {'modern', 'luxury', 'minimal', 'urban', 'cinematic'}
    found_styles = set()
    for template_name in templates:
        style = template_name.split('_')[0]
        found_styles.add(style)
    assert found_styles == style_categories
    
    # Test pacing variations
    pacing_types = {'rapid', 'balanced', 'relaxed'}
    found_pacings = set()
    for template_name in templates:
        pacing = template_name.split('_')[1]
        found_pacings.add(pacing)
    assert found_pacings == pacing_types
    
    # Test structure variations
    structure_types = {'standard', 'story', 'showcase', 'featurefocus', 'quick'}
    found_structures = set()
    for template_name in templates:
        structure = template_name.split('_')[2]
        found_structures.add(structure)
    assert found_structures == structure_types
    
    # Test template retrieval and properties
    template_name = templates[0]
    template = template_library.get_template(template_name)
    assert isinstance(template, VideoTemplate)
    assert template.name == template_name
    
    # Verify template components
    info = template_library.get_template_info(template_name)
    assert 'segments' in info
    assert 'transitions' in info
    assert 'effects' in info
    assert 'text_styles' in info
    assert 'music_style' in info

def test_pacing_curve_creation(pacing_engine):
    """Test pacing curve creation."""
    duration = 60
    curve = pacing_engine.create_pacing_curve('energetic', duration)
    
    # Test intensity values
    assert 0 <= curve.get_intensity(0) <= 1
    assert 0 <= curve.get_intensity(30) <= 1
    assert 0 <= curve.get_intensity(60) <= 1
    
    # Test with variation
    curve_with_var = pacing_engine.create_pacing_curve('energetic', duration, variation=0.3)
    assert curve_with_var.get_intensity(30) != curve.get_intensity(30)

def test_pacing_adjustments(pacing_engine):
    """Test pacing-based adjustments."""
    base_duration = 5.0
    base_intensity = 0.5
    
    # Test segment duration
    segment_duration = pacing_engine.get_segment_duration(base_duration, base_intensity)
    assert segment_duration > 0
    
    # Test transition duration
    transition_duration = pacing_engine.get_transition_duration(base_duration, base_intensity)
    assert transition_duration > 0
    
    # Test effect intensity
    effect_intensity = pacing_engine.get_effect_intensity(base_intensity, base_intensity)
    assert 0 <= effect_intensity <= 1

def test_style_combination_creation(style_engine):
    """Test style combination creation."""
    styles = [('modern', 0.7), ('luxury', 0.3)]
    combination = style_engine.create_style_combination(styles)
    
    # Test component weights
    total_weight = sum(c.weight for c in combination.components)
    assert abs(total_weight - 1.0) < 1e-6
    
    # Test attribute retrieval
    color_scheme = combination.get_weighted_value('color_scheme')
    assert color_scheme is not None
    
    # Test with variation
    combination_with_var = style_engine.create_style_combination(styles, variation=0.3)
    assert combination_with_var != combination

def test_style_interpolation(style_engine):
    """Test style interpolation."""
    style1 = style_engine.create_style_combination([('modern', 1.0)])
    style2 = style_engine.create_style_combination([('luxury', 1.0)])
    
    # Test interpolation at different points
    mid_style = style_engine.interpolate_styles(style1, style2, 0.5)
    assert len(mid_style.components) == 2
    
    start_style = style_engine.interpolate_styles(style1, style2, 0.0)
    assert start_style.components[0].weight > start_style.components[1].weight
    
    end_style = style_engine.interpolate_styles(style1, style2, 1.0)
    assert end_style.components[1].weight > end_style.components[0].weight

def test_template_customization(template_library):
    """Test template customization."""
    # Add custom template
    custom_template = {
        'name': 'custom_template',
        'duration': 45,
        'segments': [
            {'type': 'intro', 'duration': 5},
            {'type': 'content', 'duration': 35},
            {'type': 'outro', 'duration': 5}
        ],
        'transitions': [],
        'effects': [],
        'text_styles': {},
        'music_style': {}
    }
    
    template_library.add_template('custom', custom_template)
    assert 'custom' in template_library.list_templates()
    
    # Verify template
    template = template_library.get_template('custom')
    assert template.duration == 45
    assert len(template.segments) == 3
```

### ./tests/test_effects.py
```python
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
```

### ./tests/test_ai_integration.py
```python
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
```

### ./tests/test_custom_tools.py
```python
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
```

### ./tests/test_variation_system.py
```python
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
```

### ./tests/test_asset_management.py
```python
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
```

### ./tests/test_custom_agents.py
```python
import pytest
from src.director.agents.real_estate_video_agent import RealEstateVideoAgent
from src.director.agents.style_variation_agent import StyleVariationAgent

@pytest.fixture
def real_estate_agent():
    return RealEstateVideoAgent()

@pytest.fixture
def style_agent():
    return StyleVariationAgent()

@pytest.fixture
def sample_property_data():
    return {
        "id": "test123",
        "price": 1200000,
        "location": "Beverly Hills",
        "style": "modern luxury",
        "bedrooms": 4,
        "bathrooms": 3,
        "features": [
            "smart_home",
            "gourmet kitchen",
            "pool"
        ],
        "amenities": [
            "spa",
            "home theater",
            "view"
        ],
        "year_built": 2022,
        "lot_size": 12000
    }

def test_property_analysis(real_estate_agent, sample_property_data):
    """Test property analysis functionality."""
    analysis = real_estate_agent.analyze_property(sample_property_data)
    
    assert "key_features" in analysis
    assert "target_audience" in analysis
    assert "price_segment" in analysis
    assert "unique_selling_points" in analysis
    
    # Verify price segmentation
    assert analysis["price_segment"] == "luxury"
    
    # Verify feature extraction
    assert any("pool" in f.lower() for f in analysis["key_features"])
    assert any("smart" in f.lower() for f in analysis["key_features"])
    
    # Verify target audience
    assert "families" in analysis["target_audience"]
    
    # Verify unique selling points
    assert "new_construction" in analysis["unique_selling_points"]
    assert "large_lot" in analysis["unique_selling_points"]

def test_video_plan_generation(real_estate_agent, sample_property_data):
    """Test video plan generation."""
    plan = real_estate_agent.generate_video_plan(
        property_data=sample_property_data,
        style="modern",
        duration=30.0
    )
    
    assert "analysis" in plan
    assert "script" in plan
    assert "segments" in plan
    assert "effects" in plan
    assert plan["style"] == "modern"
    assert plan["duration"] == 30.0
    
    # Verify segments
    assert len(plan["segments"]) > 0
    first_segment = plan["segments"][0]
    assert "type" in first_segment
    assert "text" in first_segment
    assert "duration" in first_segment
    
    # Verify effects
    assert len(plan["effects"]) == len(plan["segments"])

def test_style_variation_generation(style_agent):
    """Test TikTok style variation generation."""
    variation = style_agent.generate_variation(
        base_style="energetic",
        target_audience=["young_professionals"],
        duration=30.0
    )
    
    assert "style" in variation
    assert "transitions" in variation
    assert "effects" in variation
    assert "music_style" in variation
    assert "text_style" in variation
    
    # Verify audience-specific adjustments
    assert any("dynamic" in e for e in variation["effects"])
    assert variation["music_style"] == "trendy"

def test_style_engagement_optimization(style_agent):
    """Test engagement optimization."""
    base_variation = style_agent.generate_variation(
        base_style="minimal",
        target_audience=["luxury_buyers"],
        duration=45.0
    )
    
    platform_metrics = {
        "avg_view_duration": 0.3,
        "engagement_rate": 0.05,
        "completion_rate": 0.6
    }
    
    optimized = style_agent.optimize_engagement(
        variation=base_variation,
        platform_metrics=platform_metrics
    )
    
    # Verify engagement optimizations
    assert any("pattern_interrupt" in e for e in optimized["effects"])
    assert any("text_callout" in e for e in optimized["effects"])
    assert any("retention_hook" in e for e in optimized["effects"])

def test_trending_elements(style_agent):
    """Test trending elements retrieval."""
    trends = style_agent.get_trending_elements()
    
    assert "transitions" in trends
    assert "effects" in trends
    assert "music_styles" in trends
    assert "hooks" in trends
    
    # Verify trend categories
    assert len(trends["transitions"]) > 0
    assert len(trends["effects"]) > 0
    assert len(trends["music_styles"]) > 0
    assert len(trends["hooks"]) > 0
```

### ./tests/test_api.py
```python
import pytest
from fastapi.testclient import TestClient
from src.api.routes import app, VideoRequest, PropertyListing, PropertySpecs

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)

@pytest.fixture
def sample_request():
    """Create sample video generation request."""
    return {
        "listing": {
            "title": "Beautiful Modern Home",
            "price": 750000,
            "description": "Stunning 4-bedroom modern home with amazing views",
            "features": [
                "Open floor plan",
                "Gourmet kitchen",
                "Master suite"
            ],
            "location": "123 Main St, Anytown, USA",
            "specs": {
                "bedrooms": 4,
                "bathrooms": 3.5,
                "square_feet": 3200,
                "lot_size": "0.5 acres",
                "year_built": 2020,
                "property_type": "Single Family"
            }
        },
        "images": [
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg",
            "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        ]
    }

def test_generate_video_endpoint(client, sample_request):
    """Test video generation endpoint."""
    response = client.post("/api/v1/videos/generate", json=sample_request)
    assert response.status_code == 200
    
    data = response.json()
    assert "task_id" in data
    assert "estimated_duration" in data
    assert "template_name" in data
    assert "status" in data
    assert data["status"] == "pending"

def test_get_status_endpoint(client, sample_request):
    """Test status endpoint."""
    # First generate a video
    response = client.post("/api/v1/videos/generate", json=sample_request)
    assert response.status_code == 200
    task_id = response.json()["task_id"]
    
    # Then check its status
    response = client.get(f"/api/v1/videos/{task_id}/status")
    assert response.status_code == 200
    
    data = response.json()
    assert data["task_id"] == task_id
    assert "status" in data
    assert "progress" in data
    assert 0 <= data["progress"] <= 100

def test_invalid_request(client):
    """Test invalid request handling."""
    # Missing required fields
    invalid_request = {
        "listing": {
            "title": "Test"
            # Missing price and other required fields
        },
        "images": []
    }
    
    response = client.post("/api/v1/videos/generate", json=invalid_request)
    assert response.status_code == 422  # Validation error

def test_invalid_task_id(client):
    """Test invalid task ID handling."""
    response = client.get("/api/v1/videos/nonexistent-task/status")
    assert response.status_code == 404

def test_validation_price(client, sample_request):
    """Test price validation."""
    request = sample_request.copy()
    request["listing"]["price"] = -100
    
    response = client.post("/api/v1/videos/generate", json=request)
    assert response.status_code == 422

def test_validation_images(client, sample_request):
    """Test image validation."""
    request = sample_request.copy()
    
    # Test empty images
    request["images"] = []
    response = client.post("/api/v1/videos/generate", json=request)
    assert response.status_code == 422
    
    # Test invalid image format
    request["images"] = ["not-a-url-or-base64"]
    response = client.post("/api/v1/videos/generate", json=request)
    assert response.status_code == 422

def test_model_validation():
    """Test Pydantic model validation."""
    # Test valid data
    data = {
        "title": "Test Home",
        "price": 500000,
        "description": "Test description",
        "features": ["Feature 1", "Feature 2"],
        "location": "Test Location",
        "specs": {
            "bedrooms": 3,
            "bathrooms": 2.5
        }
    }
    listing = ListingData(**data)
    assert listing.title == "Test Home"
    assert listing.price == 500000
    
    # Test specs validation
    specs = ListingSpecs(bedrooms=3, bathrooms=2.5)
    assert specs.bedrooms == 3
    assert specs.bathrooms == 2.5
```

### ./tests/test_director_core.py
```python
import pytest
from pathlib import Path
import yaml
from src.director.core.engine import ReasoningEngine
from src.director.core.session import SessionManager, Session
from src.director.core.config import DirectorConfig

@pytest.fixture
def config_dir(tmp_path):
    return tmp_path / "config" / "director"

@pytest.fixture
def engine(config_dir):
    config_dir.mkdir(parents=True)
    return ReasoningEngine(str(config_dir / "engine.yaml"))

@pytest.fixture
def session_manager():
    return SessionManager()

def test_engine_initialization(engine):
    """Test that the engine initializes with default config."""
    assert engine.config is not None
    assert 'llm' in engine.config
    assert 'video' in engine.config
    assert 'agents' in engine.config

def test_session_creation(session_manager):
    """Test session creation and retrieval."""
    session = session_manager.create_session()
    assert session.session_id is not None
    assert session.status == "initialized"
    assert session.progress == 0
    
    # Test retrieval
    retrieved = session_manager.get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.session_id == session.session_id

def test_session_updates(session_manager):
    """Test session progress updates."""
    session = session_manager.create_session()
    session_manager.update_session(session.session_id, 50, "processing")
    
    updated = session_manager.get_session(session.session_id)
    assert updated.progress == 50
    assert updated.status == "processing"

def test_config_management(config_dir):
    """Test configuration management."""
    config = DirectorConfig(str(config_dir))
    
    # Test saving config
    test_config = {
        'test_key': 'test_value'
    }
    config.save_config('test', test_config)
    
    # Test loading config
    loaded = config.load_config('test')
    assert loaded['test_key'] == 'test_value'
    
    # Test getting specific value
    value = config.get_value('test', 'test_key')
    assert value == 'test_value'

@pytest.mark.asyncio
async def test_engine_request_processing(engine, session_manager):
    """Test processing a video generation request."""
    engine.initialize(session_manager)
    session = session_manager.create_session()
    
    request_data = {
        "listing": {
            "title": "Test Property",
            "price": 500000,
            "description": "Beautiful test property"
        },
        "images": ["test1.jpg", "test2.jpg"]
    }
    
    result = await engine.process_request(session.session_id, request_data)
    assert result["status"] == "processing"
    assert result["session_id"] == session.session_id
```

### ./tests/test_template_count.py
```python
import pytest
from src.director.variation.template_engine import TemplateLibrary

def test_template_count():
    """Verify the number of unique templates."""
    library = TemplateLibrary()
    templates = library.list_templates()
    
    # Count unique combinations
    styles = set()
    pacings = set()
    structures = set()
    color_schemes = set()
    
    for template in templates:
        parts = template.split('_')
        styles.add(parts[0])
        pacings.add(parts[1])
        structures.add(parts[2])
        color_schemes.add(parts[3])
    
    print(f"\nTemplate Statistics:")
    print(f"Total Templates: {len(templates)}")
    print(f"Unique Styles: {len(styles)} - {sorted(styles)}")
    print(f"Unique Pacings: {len(pacings)} - {sorted(pacings)}")
    print(f"Unique Structures: {len(structures)} - {sorted(structures)}")
    print(f"Unique Color Schemes: {len(color_schemes)} - {sorted(color_schemes)}")
    
    # We expect:
    # 5 styles × 3 pacings × 5 structures × 3 color schemes = 225 templates
    assert len(templates) == 225
    assert len(styles) == 5
    assert len(pacings) == 3
    assert len(structures) == 5
    assert len(color_schemes) >= 3  # Each style has 3 color schemes
```

### ./tests/test_agents.py
```python
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
```

### ./generate_property_video.py
```python
import asyncio
import os
from dotenv import load_dotenv
from src.director.core.reasoning_engine import ReasoningEngine
from src.utils.logging import setup_logger

# Load environment variables
load_dotenv()

# Set up logger
logger = setup_logger("property_video_gen")

async def generate_property_video(property_id: str, style: str = "luxury"):
    try:
        # Initialize the engine
        engine = ReasoningEngine()
        
        # Real property data
        listing_data = {
            "id": property_id,
            "title": "Modern Waterfront Paradise",
            "price": 2850000,
            "description": """
            Discover luxury living at its finest in this stunning waterfront estate. 
            This architectural masterpiece seamlessly blends indoor and outdoor living, 
            offering breathtaking water views from nearly every room. The home features 
            soaring ceilings, walls of glass, and premium finishes throughout.
            """,
            "features": [
                "Panoramic Water Views",
                "Chef's Kitchen with Premium Appliances",
                "Primary Suite with Private Terrace",
                "Indoor-Outdoor Entertainment Spaces",
                "Private Dock",
                "Wine Cellar",
                "Smart Home Technology",
                "3-Car Garage"
            ],
            "location": "Miami Beach",
            "bedrooms": 6,
            "bathrooms": 7.5,
            "sqft": 6800,
            "year_built": 2023,
            "lot_size": 15000
        }
        
        # Property images (using test images for now)
        image_urls = [
            "gs://project3-ziltok.firebasestorage.app/test_properties/house1.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/house2.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/house3.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/interior1.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/interior2.jpg",
            "gs://project3-ziltok.firebasestorage.app/test_properties/backyard.jpg"
        ]
        
        # Progress callback
        def progress_callback(progress: float, message: str):
            logger.info(f"Progress: {progress*100:.1f}% - {message}")
        
        logger.info(f"Generating video for property {property_id}...")
        
        # Generate video
        await engine.generate_video(
            listing_data=listing_data,
            image_urls=image_urls,
            style=style,
            duration=30.0,  # 30 second video
            progress_callback=progress_callback
        )
        
        logger.info("Video generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Video generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # You can change the style to "modern" or "minimal" for different looks
        asyncio.run(generate_property_video("waterfront-estate-1", style="luxury"))
    except KeyboardInterrupt:
        logger.info("Video generation interrupted by user")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
```

### ./vendor/Director/backend/director/tools/kling.py
```python
import requests
import time
import jwt

PARAMS_CONFIG = {
    "text_to_video": {
        "model": {
            "type": "string",
            "description": "Model to use for video generation",
            "enum": ["kling-v1"],
            "default": "kling-v1",
        },
        "negative_prompt": {
            "type": "string",
            "description": "Negative text prompt",
            "maxLength": 5200,
        },
        "cfg_scale": {
            "type": "number",
            "description": "Flexibility in video generation. The higher the value, "
            "the lower the model's degree of flexibility and the "
            "stronger the relevance to the user's prompt",
            "minimum": 0,
            "maximum": 1,
            "default": 0.5,
        },
        "mode": {
            "type": "string",
            "description": "Video generation mode",
            "enum": ["std", "pro"],
            "default": "std",
        },
        "camera_control": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Type of camera movement. Options are:\n"
                    "- simple: Basic camera movement, configurable via config\n"
                    "- none: No camera movement\n"
                    "- down_back: Camera descends and moves backward with pan down and\n"
                    "  zoom out effect\n"
                    "- forward_up: Camera moves forward and tilts up with zoom in\n"
                    "  and pan up effect\n"
                    "- right_turn_forward: Camera rotates right while moving forward\n"
                    "- left_turn_forward: Camera rotates left while moving forward",
                    "enum": [
                        "simple",
                        "none",
                        "down_back",
                        "forward_up",
                        "right_turn_forward",
                        "left_turn_forward",
                    ],
                    "default": "none",
                },
                "config": {
                    "type": "object",
                    "description": "Contains 8 fields to specify the camera's movement or change in different directions, This should only be passed if type is simple",
                    "properties": {
                        "horizontal": {
                            "type": "number",
                            "description": "Controls the camera's movement along the horizontal axis (translation along the x-axis). Value range: [-10, 10], negative value indicates translation to the left, positive value indicates translation to the right",
                            "minimum": -10,
                            "maximum": 10,
                            "default": 0,
                        },
                        "vertical": {
                            "type": "number",
                            "description": "Controls the camera's movement along the vertical axis (translation along the y-axis). Value range: [-10, 10], negative value indicates a downward translation, positive value indicates an upward translation",
                            "minimum": -10,
                            "maximum": 10,
                            "default": 0,
                        },
                        "pan": {
                            "type": "number",
                            "description": "Controls the camera's rotation in the horizontal plane (rotation around the y-axis). Value range: [-10, 10], negative value indicates rotation to the left, positive value indicates rotation to the right",
                            "minimum": -10,
                            "maximum": 10,
                            "default": 0,
                        },
                        "tilt": {
                            "type": "number",
                            "description": "Controls the camera's rotation in the vertical plane (rotation around the x-axis). Value range: [-10, 10], negative value indicates downward rotation, positive value indicates upward rotation",
                            "minimum": -10,
                            "maximum": 10,
                            "default": 0,
                        },
                        "roll": {
                            "type": "number",
                            "description": "Controls the camera's rolling amount (rotation around the z-axis). Value range: [-10, 10], negative value indicates counterclockwise rotation, positive value indicates clockwise rotation",
                            "minimum": -10,
                            "maximum": 10,
                            "default": 0,
                        },
                        "zoom": {
                            "type": "number",
                            "description": "Controls the change in the camera's focal length, affecting the proximity of the field of view. Value range: [-10, 10], negative value indicates increase in focal length (narrower field of view), positive value indicates decrease in focal length (wider field of view)",
                            "minimum": -10,
                            "maximum": 10,
                            "default": 0,
                        },
                    },
                },
            },
        },
    }
}


class KlingAITool:
    def __init__(self, access_key: str, secret_key: str):
        self.api_route = "https://api.klingai.com"
        self.video_endpoint = f"{self.api_route}/v1/videos/text2video"
        self.access_key = access_key
        self.secret_key = secret_key
        self.polling_interval = 30  # seconds

    def get_authorization_token(self):
        headers = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "iss": self.access_key,
            "exp": int(time.time()) + 1800,  # Valid for 30 minutes
            "nbf": int(time.time()) - 5,  # Start 5 seconds ago
        }
        token = jwt.encode(payload, self.secret_key, headers=headers)
        return token

    def text_to_video(
        self, prompt: str, save_at: str, duration: float, config: dict
    ):
        """
        Generate a video from a text prompt using KlingAI's API.
        :param str prompt: The text prompt to generate the video
        :param str save_at: File path to save the generated video
        :param float duration: Duration of the video in seconds
        :param dict config: Additional configuration options
        """
        api_key = self.get_authorization_token()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "prompt": prompt,
            "model": config.get("model", "kling-v1"),
            "duration": duration,
            **config,  # Include any additional configuration parameters
        }

        response = requests.post(self.video_endpoint, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(f"Error generating video: {response.text}")

        # Assuming the API returns a job ID for asynchronous processing
        job_id = response.json()["data"].get("task_id")
        if not job_id:
            raise Exception("No task ID returned from the API.")

        # Polling for the video generation completion
        result_endpoint = f"{self.api_route}/v1/videos/text2video/{job_id}"

        while True:
            response = requests.get(
                result_endpoint, headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()

            print("Kling Response", response)

            status = response.json()["data"]["task_status"]

            if status == "succeed":
                # Video generation is complete
                video_url = response.json()["data"]["task_result"]["videos"][0]["url"]
                # Download and save the video
                video_response = requests.get(video_url)
                video_response.raise_for_status()
                with open(save_at, "wb") as f:
                    f.write(video_response.content)
                break
            else:
                # Still processing
                time.sleep(self.polling_interval)
                continue```

### ./vendor/Director/backend/director/tools/serp.py
```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class SerpAPI:
    BASE_URL = "https://serpapi.com/search.json"
    RETRY_TOTAL = 3
    RETRY_BACKOFF_FACTOR = 1
    RETRY_STATUS_CODES = [429, 500, 502, 503, 504]

    def __init__(self, api_key: str, base_url: str = None, timeout: int = 10):
        """
        Initialize the SerpAPI client.
        :param api_key: API key for SerpAPI.
        :param base_url: Optional base URL for the API.
        :param timeout: Timeout for API requests in seconds.
        """
        if not api_key:
            raise ValueError("API key is required for SerpAPI.")
        self.api_key = api_key
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout

        # Configure retries
        retry_strategy = Retry(
            total=self.RETRY_TOTAL,
            backoff_factor=self.RETRY_BACKOFF_FACTOR,
            status_forcelist=self.RETRY_STATUS_CODES,
        )
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)

    def search_videos(self, query: str, count: int, duration: str = None) -> list:
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")
        if not isinstance(count, int) or count < 1:
            raise ValueError("Count must be a positive integer.")
        """
        Perform a video search using SerpAPI.
        :param query: Search query for the video.
        :param count: Number of video results to retrieve.
        :param duration: Filter videos by duration (short, medium, long).
        :return: A list of raw video results from SerpAPI.
        """
        params = {
            "q": query,
            "tbm": "vid",
            "num": count,
            "hl": "en",
            "gl": "us",
            "api_key": self.api_key,
        }

        # Map duration values to SerpAPI's expected format
        duration_mapping = {
            "short": "dur:s",
            "medium": "dur:m",
            "long": "dur:l",
        }

        if duration:
            if duration not in duration_mapping:
                raise ValueError(f"Invalid duration value: {duration}")
            params["tbs"] = duration_mapping[duration]

        try:
            response = self.session.get(self.base_url, params=params, timeout=self.timeout)
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as e:
                raise RuntimeError("Invalid JSON response from SerpAPI") from e
            
            results = data.get("video_results")
            if results is None:
                raise RuntimeError("Unexpected response format: 'video_results' not found")
            return results
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error during SerpAPI video search: {e}") from e
```

### ./vendor/Director/backend/director/tools/composio_tool.py
```python
import os
import json

from director.llm.openai import OpenAIChatModel


def composio_tool(task: str):
    from composio_openai import ComposioToolSet
    from openai import OpenAI

    key = os.getenv("OPENAI_API_KEY")
    base_url = "https://api.openai.com/v1"

    if not key:
        key = os.getenv("VIDEO_DB_API_KEY")
        base_url = os.getenv("VIDEO_DB_BASE_URL", "https://api.videodb.io")

    openai_client = OpenAI(api_key=key, base_url=base_url)

    toolset = ComposioToolSet(api_key=os.getenv("COMPOSIO_API_KEY"))
    tools = toolset.get_tools(apps=json.loads(os.getenv("COMPOSIO_APPS")))
    print(tools)

    response = openai_client.chat.completions.create(
        model=OpenAIChatModel.GPT4o,
        tools=tools,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": task},
        ],
    )

    composio_response = toolset.handle_tool_calls(response=response)
    if composio_response:
        return composio_response
    else:
        return response.choices[0].message.content or ""
```

### ./vendor/Director/backend/director/tools/replicate.py
```python
import replicate

from dotenv import load_dotenv

load_dotenv()


def flux_dev(prompt):
    output = replicate.run(
        "black-forest-labs/flux-dev",
        input={
            "prompt": prompt,
            "go_fast": True,
            "guidance": 3.5,
            "megapixels": "1",
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "webp",
            "output_quality": 80,
            "prompt_strength": 0.8,
            "num_inference_steps": 28,
        },
    )
    return output

def flux_schnell(prompt):
    output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={
            "prompt": prompt,
            "go_fast": True,
            "megapixels": "1",
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "webp",
            "output_quality": 80,
            "num_inference_steps": 4
        }
    )
    return output```

### ./vendor/Director/backend/director/tools/videodb_tool.py
```python
import os
import requests
import videodb
import logging

from videodb import SearchType, SubtitleStyle, IndexType, SceneExtractionType
from videodb.timeline import Timeline
from videodb.asset import VideoAsset, ImageAsset


class VideoDBTool:
    def __init__(self, collection_id="default"):
        self.conn = videodb.connect(
            base_url=os.getenv("VIDEO_DB_BASE_URL", "https://api.videodb.io")
        )
        self.collection = None
        if collection_id:
            self.collection = self.conn.get_collection(collection_id)
        self.timeline = None

    def get_collection(self):
        return {
            "id": self.collection.id,
            "name": self.collection.name,
            "description": self.collection.description,
        }

    def get_collections(self):
        """Get all collections."""
        collections = self.conn.get_collections()
        return [
            {
                "id": collection.id,
                "name": collection.name,
                "description": collection.description,
            }
            for collection in collections
        ]
    def get_image(self, image_id: str = None):
        """
        Fetch image details by ID or validate an image URL.
        """
        try:
            image = self.collection.get_image(image_id)
            return {
                "id": image.id,
                "url": image.url,
                "name": image.name,
                "description": getattr(image, "description", None),
                "collection_id": image.collection_id,
            }
        except Exception as e:
            raise Exception(f"Failed to fetch image with ID {image_id}: {e}")

    def create_collection(self, name, description=""):
        """Create a new collection with the given name and description."""
        if not name:
            raise ValueError("Collection name is required to create a collection.")

        try:
            new_collection = self.conn.create_collection(name, description)
            return {
                "success": True,
                "message": f"Collection '{new_collection.id}' created successfully",
                "collection": {
                    "id": new_collection.id,
                    "name": new_collection.name,
                    "description": new_collection.description,
                },
            }
        except Exception as e:
            logging.error(f"Failed to create collection '{name}': {e}")
            raise Exception(f"Failed to create collection '{name}': {str(e)}") from e

    def delete_collection(self):
        """Delete the current collection."""
        if not self.collection:
            raise ValueError("Collection ID is required to delete a collection.")
        try:
            self.collection.delete()
            return {
                "success": True,
                "message": f"Collection {self.collection.id} deleted successfully",
            }
        except Exception as e:
            raise Exception(
                f"Failed to delete collection {self.collection.id}: {str(e)}"
            )

    def get_video(self, video_id):
        """Get a video by ID."""
        video = self.collection.get_video(video_id)
        return {
            "id": video.id,
            "name": video.name,
            "description": video.description,
            "collection_id": video.collection_id,
            "stream_url": video.stream_url,
            "length": video.length,
            "thumbnail_url": video.thumbnail_url,
        }

    def delete_video(self, video_id):
        """Delete a specific video by its ID."""
        if not video_id:
            raise ValueError("Video ID is required to delete a video.")
        try:
            video = self.collection.get_video(video_id)
            if not video:
                raise ValueError(
                    f"Video with ID {video_id} not found in collection {self.collection.id}."
                )

            video.delete()
            return {
                "success": True,
                "message": f"Video {video.id} deleted successfully",
            }
        except ValueError as ve:
            logging.error(f"ValueError while deleting video: {ve}")
            raise ve
        except Exception as e:
            logging.exception(
                f"Unexpected error occurred while deleting video {video_id}"
            )
            raise Exception(
                "An unexpected error occurred while deleting the video. Please try again later."
            )

    def get_videos(self):
        """Get all videos in a collection."""
        videos = self.collection.get_videos()
        return [
            {
                "id": video.id,
                "name": video.name,
                "description": video.description,
                "collection_id": video.collection_id,
                "stream_url": video.stream_url,
                "length": video.length,
                "thumbnail_url": video.thumbnail_url,
            }
            for video in videos
        ]

    def get_audio(self, audio_id):
        """Get an audio by ID."""
        audio = self.collection.get_audio(audio_id)
        return {
            "id": audio.id,
            "name": audio.name,
            "collection_id": audio.collection_id,
            "length": audio.length,
        }

    def generate_image_url(self, image_id):
        image = self.collection.get_image(image_id)
        return image.generate_url()

    def upload(self, source, source_type="url", media_type="video", name=None):
        upload_args = {"media_type": media_type}
        if name:
            upload_args["name"] = name
        if source_type == "url":
            upload_args["url"] = source
        elif source_type == "file":
            upload_url_data = self.conn.get(
                path=f"/collection/{self.collection.id}/upload_url",
                params={"name": name},
            )
            upload_url = upload_url_data.get("upload_url")
            files = {"file": (name, source)}
            response = requests.post(upload_url, files=files)
            response.raise_for_status()
            upload_args["url"] = upload_url
        else:
            upload_args["file_path"] = source
        media = self.conn.upload(**upload_args)
        name = media.name
        if media_type == "video":
            return {
                "id": media.id,
                "collection_id": media.collection_id,
                "stream_url": media.stream_url,
                "player_url": media.player_url,
                "name": name,
                "description": media.description,
                "thumbnail_url": media.thumbnail_url,
                "length": media.length,
            }
        elif media_type == "audio":
            return {
                "id": media.id,
                "collection_id": media.collection_id,
                "name": media.name,
                "length": media.length,
            }
        elif media_type == "image":
            return {
                "id": media.id,
                "collection_id": media.collection_id,
                "name": media.name,
                "url": media.url,
            }

    def generate_thumbnail(self, video_id: str, timestamp: int = 5):
        video = self.collection.get_video(video_id)
        image = video.generate_thumbnail(time=float(timestamp))
        return {
            "id": image.id,
            "collection_id": image.collection_id,
            "name": image.name,
            "url": image.url,
        }

    def get_transcript(self, video_id: str, text=True):
        video = self.collection.get_video(video_id)
        if text:
            transcript = video.get_transcript_text()
        else:
            transcript = video.get_transcript()
        return transcript

    def index_spoken_words(self, video_id: str):
        # TODO: Language support
        video = self.collection.get_video(video_id)
        index = video.index_spoken_words()
        return index

    def index_scene(
        self,
        video_id: str,
        extraction_type=SceneExtractionType.shot_based,
        extraction_config={},
        model_name=None,
        prompt=None,
    ):
        video = self.collection.get_video(video_id)
        return video.index_scenes(
            extraction_type=extraction_type,
            extraction_config=extraction_config,
            prompt=prompt,
            model_name=model_name,
        )

    def list_scene_index(self, video_id: str):
        video = self.collection.get_video(video_id)
        return video.list_scene_index()

    def get_scene_index(self, video_id: str, scene_id: str):
        video = self.collection.get_video(video_id)
        return video.get_scene_index(scene_id)

    def download(self, stream_link: str, name: str = None):
        download_response = self.conn.download(stream_link, name)
        return download_response

    def semantic_search(
        self, query, index_type=IndexType.spoken_word, video_id=None, **kwargs
    ):
        if video_id:
            video = self.collection.get_video(video_id)
            search_resuls = video.search(query=query, index_type=index_type, **kwargs)
        else:
            kwargs.pop("scene_index_id", None)
            search_resuls = self.collection.search(
                query=query, index_type=index_type, **kwargs
            )
        return search_resuls

    def keyword_search(
        self, query, index_type=IndexType.spoken_word, video_id=None, **kwargs
    ):
        """Search for a keyword in a video."""
        video = self.collection.get_video(video_id)
        return video.search(
            query=query, search_type=SearchType.keyword, index_type=index_type, **kwargs
        )

    def generate_video_stream(self, video_id: str, timeline):
        """Generate a video stream from a timeline. timeline is a list of tuples. ex [(0, 10), (20, 30)]"""
        video = self.collection.get_video(video_id)
        return video.generate_stream(timeline)

    def add_brandkit(self, video_id, intro_video_id, outro_video_id, brand_image_id):
        timeline = Timeline(self.conn)
        if intro_video_id:
            intro_video = VideoAsset(asset_id=intro_video_id)
            timeline.add_inline(intro_video)
        video = VideoAsset(asset_id=video_id)
        timeline.add_inline(video)
        if outro_video_id:
            outro_video = VideoAsset(asset_id=outro_video_id)
            timeline.add_inline(outro_video)
        if brand_image_id:
            brand_image = ImageAsset(asset_id=brand_image_id)
            timeline.add_overlay(0, brand_image)
        stream_url = timeline.generate_stream()
        return stream_url

    def get_and_set_timeline(self):
        self.timeline = Timeline(self.conn)
        return self.timeline

    def add_subtitle(self, video_id, style: SubtitleStyle = SubtitleStyle()):
        video = self.collection.get_video(video_id)
        stream_url = video.add_subtitle(style)
        return stream_url
```

### ./vendor/Director/backend/director/tools/stabilityai.py
```python
import requests
import time
from PIL import Image
import io

PARAMS_CONFIG = {
    "text_to_video": {
        "strength": {
            "type": "number",
            "description": "Image influence on output",
            "minimum": 0,
            "maximum": 1,
        },
        "negative_prompt": {
            "type": "string",
            "description": "Keywords to exclude from output",
        },
        "seed": {
            "type": "integer",
            "description": "Randomness seed for generation",
        },
        "cfg_scale": {
            "type": "number",
            "description": "How strongly video sticks to original image",
            "minimum": 0,
            "maximum": 10,
            "default": 1.8,
        },
        "motion_bucket_id": {
            "type": "integer",
            "description": "Controls motion amount in output video",
            "minimum": 1,
            "maximum": 255,
            "default": 127,
        },
    },
}


class StabilityAITool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.image_endpoint = (
            "https://api.stability.ai/v2beta/stable-image/generate/ultra"
        )
        self.video_endpoint = "https://api.stability.ai/v2beta/image-to-video"
        self.result_endpoint = "https://api.stability.ai/v2beta/image-to-video/result"
        self.polling_interval = 10  # seconds

    def text_to_video(self, prompt: str, save_at: str, duration: float, config: dict):
        """
        Generate a video from a text prompt using Stability AI's API.
        First generates an image from text, then converts it to video.
        :param str prompt: The text prompt to generate the video
        :param str save_at: File path to save the generated video
        :param float duration: Duration of the video in seconds
        :param dict config: Additional configuration options
        """
        # First generate image from text
        headers = {"authorization": f"Bearer {self.api_key}", "accept": "image/*"}

        image_payload = {
            "prompt": prompt,
            "output_format": config.get("format", "png"),
            "aspect_ratio": config.get("aspect_ratio", "16:9"),
            "negative_prompt": config.get("negative_prompt", ""),
        }

        image_response = requests.post(
            self.image_endpoint, headers=headers, files={"none": ""}, data=image_payload
        )

        if image_response.status_code != 200:
            raise Exception(f"Error generating image: {image_response.text}")

        image = Image.open(io.BytesIO(image_response.content))

        # Set the new dimensions
        new_width = 1024
        new_height = int(new_width * (576 / 1024))  # Maintain the 16:9 aspect ratio

        # Resize the image
        scaled_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        temp_image_path = f"{save_at}.temp.png"

        # Save temporary image
        scaled_image.save(temp_image_path)

        # Generate video from the image
        video_headers = {"authorization": f"Bearer {self.api_key}"}

        video_payload = {
            "seed": config.get("seed", 0),
            "cfg_scale": config.get("cfg_scale", 1.8),
            "motion_bucket_id": config.get("motion_bucket_id", 127),
        }

        with open(temp_image_path, "rb") as img_file:
            video_response = requests.post(
                self.video_endpoint,
                headers=video_headers,
                files={"image": img_file},
                data=video_payload,
            )

        if video_response.status_code != 200:
            raise Exception(f"Error generating video: {video_response.text}")

        # Get generation ID and wait for completion
        generation_id = video_response.json().get("id")
        if not generation_id:
            raise Exception("No generation ID in response")

        # Poll for completion
        result_headers = {
            "accept": "video/*",
            "authorization": f"Bearer {self.api_key}",
        }

        while True:
            result_response = requests.get(
                f"{self.result_endpoint}/{generation_id}", headers=result_headers
            )

            if result_response.status_code == 202:
                # Still processing
                time.sleep(self.polling_interval)
                continue
            elif result_response.status_code == 200:
                with open(save_at, "wb") as f:
                    f.write(result_response.content)
                break
            else:
                raise Exception(f"Error fetching video: {result_response.text}")
```

### ./vendor/Director/backend/director/tools/fal_video.py
```python
import os
import fal_client
import requests
from typing import Optional


PARAMS_CONFIG = {
    "text_to_video": {
        "model_name": {
            "type": "string",
            "description": "The model name to use for video generation",
            "default": "fal-ai/fast-animatediff/text-to-video",
            "enum": [
                "fal-ai/minimax-video",
                "fal-ai/mochi-v1",
                "fal-ai/hunyuan-video",
                "fal-ai/luma-dream-machine",
                "fal-ai/kling-video/v1/standard/text-to-video",
                "fal-ai/kling-video/v1.5/pro/text-to-video",
                "fal-ai/cogvideox-5b",
                "fal-ai/ltx-video",
                "fal-ai/fast-svd/text-to-video",
                "fal-ai/fast-svd-lcm/text-to-video",
                "fal-ai/t2v-turbo",
                "fal-ai/fast-animatediff/text-to-video",
                "fal-ai/fast-animatediff/turbo/text-to-video",
                # "fal-ai/animatediff-sparsectrl-lcm",
            ],
        },
    },
    "image_to_video": {
        "model_name": {
            "type": "string",
            "description": "The model name to use for image-to-video generation",
            "default": "fal-ai/fast-svd-lcm",
            "enum": [
                "fal-ai/haiper-video/v2/image-to-video",
                "fal-ai/luma-dream-machine/image-to-video",
                # "fal-ai/kling-video/v1/standard/image-to-video",
                # "fal-ai/kling-video/v1/pro/image-to-video",
                # "fal-ai/kling-video/v1.5/pro/image-to-video",
                "fal-ai/cogvideox-5b/image-to-video",
                "fal-ai/ltx-video/image-to-video",
                "fal-ai/stable-video",
                "fal-ai/fast-svd-lcm",
            ],
        },
    },
    "image_to_image": {
        "model_name": {
            "type": "string",
            "description": "The model name to use for image-to-image transformation",
            "default": "fal-ai/flux-lora-canny",
            "enum": [
                "fal-ai/flux-pro/v1.1-ultra/redux",
                "fal-ai/flux-lora-canny",
                "fal-ai/flux-lora-depth",
                "fal-ai/ideogram/v2/turbo/remix",
                "fal-ai/iclight-v2",
            ],
        },
    },
}


class FalVideoGenerationTool:
    def __init__(self, api_key: str):
        if not api_key:
            raise Exception("FAL API key not found")
        self.api_key = api_key
        self.queue_endpoint = "https://queue.fal.run"
        self.polling_interval = 10  # seconds

    def text_to_video(
        self, prompt: str, save_at: str, duration: float, config: dict
    ):
        """
        Generates a video from using text-to-video models.
        """
        try:
            model_name = config.get("model_name", "fal-ai/minimax-video")
            res = fal_client.run(
                model_name,
                arguments={"prompt": prompt, "duration": duration},
            )
            video_url = res["video"]["url"]
            with open(save_at, "wb") as f:
                f.write(requests.get(video_url).content)

        except Exception as e:
            raise Exception(f"Error generating video: {type(e).__name__}: {str(e)}")

        return {"status": "success", "video_path": save_at}

    def image_to_video(
        self,
        image_url: str,
        save_at: str,
        duration: float,
        config: dict,
        prompt: Optional[str] = None,
    ):
        """
        Generate video from an image URL.
        """
        try:
            model_name = config.get("model_name", "fal-ai/fast-svd-lcm")
            arguments = {"image_url": image_url, "duration": duration}

            if model_name == "fal-ai/haiper-video/v2/image-to-video":
                arguments["duration"] = 6

            if prompt:
                arguments["prompt"] = prompt

            res = fal_client.run(
                model_name,
                arguments=arguments,
            )

            video_url = res["video"]["url"]

            with open(save_at, "wb") as f:
                f.write(requests.get(video_url).content)
        except Exception as e:
            raise Exception(f"Error generating video: {type(e).__name__}: {str(e)}")

    def image_to_image(self, image_url: str, prompt: str, config: dict):
        try:
            model_name = config.get("model_name", "fal-ai/flux-lora-canny")
            arguments = {"image_url": image_url, "prompt": prompt}

            res = fal_client.run(
                model_name,
                arguments=arguments,
            )

            print("we got this response", res)
            return res["images"]

        except Exception as e:
            raise Exception(f"Error generating image: {type(e).__name__}: {str(e)}")
```

### ./vendor/Director/backend/director/tools/elevenlabs.py
```python
import time
from typing import Optional

from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

DEFAULT_VOICES = """
1. 9BWtsMINqrJLrRacOk9x - Aria: Expressive, American, female.
2. CwhRBWXzGAHq8TQ4Fs17 - Roger: Confident, American, male.
3. EXAVITQu4vr4xnSDxMaL - Sarah: Soft, American, young female.
4. FGY2WhTYpPnrIDTdsKH5 - Laura: Upbeat, American, young female.
5. IKne3meq5aSn9XLyUdCD - Charlie: Natural, Australian, male.
6. JBFqnCBsd6RMkjVDRZzb - George: Warm, British, middle-aged male.
7. N2lVS1w4EtoT3dr4eOWO - Callum: Intense, Transatlantic, male.
8. SAz9YHcvj6GT2YYXdXww - River: Confident, American, non-binary.
9. TX3LPaxmHKxFdv7VOQHJ - Liam: Articulate, American, young male.
10. XB0fDUnXU5powFXDhCwa - Charlotte: Seductive, Swedish, young female.
11. Xb7hH8MSUJpSbSDYk0k2 - Alice: Confident, British, middle-aged female.
12. XrExE9yKIg1WjnnlVkGX - Matilda: Friendly, American, middle-aged female.
13. bIHbv24MWmeRgasZH58o - Will: Friendly, American, young male.
14. cgSgspJ2msm6clMCkdW9 - Jessica: Expressive, American, young female.
15. cjVigY5qzO86Huf0OWal - Eric: Friendly, American, middle-aged male.
16. iP95p4xoKVk53GoZ742B - Chris: Casual, American, middle-aged male.
17. nPczCjzI2devNBz1zQrb - Brian: Deep, American, middle-aged male.
18. onwK4e9ZLuTAKqWW03F9 - Daniel: Authoritative, British, middle-aged male.
19. pFZP5JQG7iQjIQuC4Bku - Lily: Warm, British, middle-aged female.
20. pqHfZKP75CvOlQylNhV4 - Bill: Trustworthy, American, old male.
"""


PARAMS_CONFIG = {
    "sound_effect": {
        "prompt_influence": {
            "type": "number",
            "description": (
                "A value between 0 and 1 that determines how closely the generation "
                "follows the prompt. Higher values make generations follow the prompt "
                "more closely while also making them less variable. Defaults to 0.3"
            ),
            "minimum": 0,
            "maximum": 1,
            "default": 0.3,
        }
    },
    "text_to_speech": {
        "model_id": {
            "type": "string",
            "description": ("Identifier of the model that will be used"),
            "default": "eleven_multilingual_v2",
        },
        "voice_id": {
            "type": "string",
            "description": f"The ID of the voice to use for text-to-speech, Some available voice ids are {DEFAULT_VOICES}",
            "default": "pNInz6obpgDQGcFmaJg",
        },
        "output_format": {
            "type": "string",
            "description": (
                "Output format of the generated audio. Format options:\n"
                "mp3_22050_32 - MP3 at 22.05kHz sample rate, 32kbps\n"
                "mp3_44100_32 - MP3 at 44.1kHz sample rate, 32kbps\n"
                "mp3_44100_64 - MP3 at 44.1kHz sample rate, 64kbps\n"
                "mp3_44100_96 - MP3 at 44.1kHz sample rate, 96kbps\n"
                "mp3_44100_128 - MP3 at 44.1kHz sample rate, 128kbps\n"
                "mp3_44100_192 - MP3 at 44.1kHz sample rate, 192kbps"
            ),
            "enum": [
                "mp3_22050_32",
                "mp3_44100_32",
                "mp3_44100_64",
                "mp3_44100_96",
                "mp3_44100_128",
                "mp3_44100_192",
            ],
            "default": "mp3_44100_128",
        },
        "language_code": {
            "type": "string",
            "description": (
                "Language code (ISO 639-1) used to enforce a language for the "
                "model. Currently only Turbo v2.5 supports language enforcement. "
                "For other models, an error will be returned if language code is "
                "provided."
            ),
        },
        "stability": {
            "type": "number",
            "description": "Stability value between 0 and 1 for voice settings",
            "minimum": 0,
            "maximum": 1,
            "default": 0.0,
        },
        "similarity_boost": {
            "type": "number",
            "description": "Similarity boost value between 0 and 1 for voice settings",
            "minimum": 0,
            "maximum": 1,
            "default": 1.0,
        },
        "style": {
            "type": "number",
            "description": "Style value between 0 and 1 for voice settings",
            "minimum": 0,
            "maximum": 1,
            "default": 0.0,
        },
        "use_speaker_boost": {
            "type": "boolean",
            "description": "Whether to use speaker boost in voice settings",
            "default": True,
        },
    },
}


class ElevenLabsTool:
    def __init__(self, api_key: str):
        if api_key:
            self.client = ElevenLabs(api_key=api_key)
        else:
            raise Exception("ElevenLabs API key not found")
        self.voice_settings = VoiceSettings(
            stability=0.0, similarity_boost=1.0, style=0.0, use_speaker_boost=True
        )
        self.constrains = {
            "sound_effect": {"max_duration": 20},
        }

    def generate_sound_effect(
        self, prompt: str, save_at: str, duration: float, config: dict
    ):
        try:
            result = self.client.text_to_sound_effects.convert(
                text=prompt,
                duration_seconds=min(
                    duration, self.constrains["sound_effect"]["max_duration"]
                ),
                prompt_influence=config.get("prompt_influence", 0.3),
            )
            with open(save_at, "wb") as f:
                for chunk in result:
                    f.write(chunk)
        except Exception as e:
            raise Exception(f"Error generating sound effect: {str(e)}")

    def text_to_speech(self, text: str, save_at: str, config: dict):
        try:
            response = self.client.text_to_speech.convert(
                voice_id=config.get("voice_id", "pNInz6obpgDQGcFmaJgB"),
                output_format=config.get("output_format", "mp3_44100_128"),
                text=text,
                model_id=config.get("model_id", "eleven_multilingual_v2"),
                voice_settings=VoiceSettings(
                    stability=config.get("stability", 0.0),
                    similarity_boost=config.get("similarity_boost", 1.0),
                    style=config.get("style", 0.0),
                    use_speaker_boost=config.get("use_speaker_boost", True),
                ),
            )
            with open(save_at, "wb") as f:
                for chunk in response:
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            raise Exception(f"Error converting text to speech: {str(e)}")

    def create_dub_job(
        self,
        source_url: str,
        target_language: str,
    ) -> Optional[str]:
        """
        Dub an audio or video file from one language to another.

        Args:
            input_file_path: Path to input file
            file_format: Format of input file (e.g. "audio/mpeg")
            source_language: Source language code (e.g. "en")
            target_language: Target language code (e.g. "es")

        Returns:
            Path to dubbed file if successful, None if failed
        """
        try:
            response = self.client.dubbing.dub_a_video_or_an_audio_file(
                source_url=source_url,
                target_lang=target_language,
            )

            dubbing_id = response.dubbing_id
            return dubbing_id

        except Exception as e:
            raise Exception(f"Error creating dub job: {str(e)}")

    def wait_for_dub_job(self, dubbing_id: str) -> bool:
        """Wait for dubbing to complete."""
        MAX_ATTEMPTS = 120
        CHECK_INTERVAL = 30  # In seconds

        for _ in range(MAX_ATTEMPTS):
            try:
                metadata = self.client.dubbing.get_dubbing_project_metadata(dubbing_id)
                print("this is metadata", metadata)
                if metadata.status == "dubbed":
                    return True
                elif metadata.status == "dubbing":
                    time.sleep(CHECK_INTERVAL)
                else:
                    return False
            except Exception as e:
                print(f"Error checking dubbing status: {str(e)}")
                return False
        return False

    def download_dub_file(
        self, dubbing_id: str, language_code: str, output_path: str
    ) -> Optional[str]:
        """Download the dubbed file."""
        try:
            with open(output_path, "wb") as file:
                for chunk in self.client.dubbing.get_dubbed_file(
                    dubbing_id, language_code
                ):
                    file.write(chunk)
            return output_path
        except Exception as e:
            print(f"Error downloading dubbed file: {str(e)}")
            return None
```

### ./vendor/Director/backend/director/tools/slack.py
```python
import os

from slack_sdk import WebClient


def send_message_to_channel(message, channel_name):
    slack_token = os.environ.get("SLACK_BOT_TOKEN")
    slack_client = WebClient(token=slack_token)
    response = slack_client.chat_postMessage(channel=channel_name, text=message)
    return response
```

### ./vendor/Director/backend/director/llm/videodb_proxy.py
```python
import os
import json
from enum import Enum

from pydantic import Field, field_validator, FieldValidationInfo


from director.llm.base import BaseLLM, BaseLLMConfig, LLMResponse, LLMResponseStatus
from director.constants import (
    LLMType,
)


class OpenAIChatModel(str, Enum):
    """Enum for OpenAI Chat models"""

    GPT4o = "gpt-4o-2024-11-20"


class VideoDBProxyConfig(BaseLLMConfig):
    """OpenAI Config"""

    llm_type: str = LLMType.VIDEODB_PROXY
    api_key: str = os.getenv("VIDEO_DB_API_KEY")
    api_base: str = os.getenv("VIDEO_DB_BASE_URL", "https://api.videodb.io")
    chat_model: str = Field(default=OpenAIChatModel.GPT4o)
    max_tokens: int = 4096

    @field_validator("api_key")
    @classmethod
    def validate_non_empty(cls, v, info: FieldValidationInfo):
        if not v:
            raise ValueError("Please set VIDEO_DB_API_KEY environment variable.")
        return v


class VideoDBProxy(BaseLLM):
    def __init__(self, config: VideoDBProxyConfig = None):
        """
        :param config: OpenAI Config
        """
        if config is None:
            config = VideoDBProxyConfig()
        super().__init__(config=config)
        try:
            import openai
        except ImportError:
            raise ImportError("Please install OpenAI python library.")

        self.client = openai.OpenAI(api_key=self.api_key, base_url=f"{self.api_base}")

    def _format_messages(self, messages: list):
        """Format the messages to the format that OpenAI expects."""
        formatted_messages = []
        for message in messages:
            if message["role"] == "assistant" and message.get("tool_calls"):
                formatted_messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                        "tool_calls": [
                            {
                                "id": tool_call["id"],
                                "function": {
                                    "name": tool_call["tool"]["name"],
                                    "arguments": json.dumps(
                                        tool_call["tool"]["arguments"]
                                    ),
                                },
                                "type": tool_call["type"],
                            }
                            for tool_call in message["tool_calls"]
                        ],
                    }
                )
            else:
                formatted_messages.append(message)
        return formatted_messages

    def _format_tools(self, tools: list):
        """Format the tools to the format that OpenAI expects.

        **Example**::

            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_delivery_date",
                        "description": "Get the delivery date for a customer's order.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "order_id": {
                                    "type": "string",
                                    "description": "The customer's order ID."
                                }
                            },
                            "required": ["order_id"],
                            "additionalProperties": False
                        }
                    }
                }
            ]
        """
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                    "strict": True,
                }
            )
        return formatted_tools

    def chat_completions(
        self, messages: list, tools: list = [], stop=None, response_format=None
    ):
        """Get completions for chat.

        docs: https://platform.openai.com/docs/guides/function-calling
        """
        params = {
            "model": self.chat_model,
            "messages": self._format_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": stop,
            "timeout": self.timeout,
        }
        if tools:
            params["tools"] = self._format_tools(tools)
            params["tool_choice"] = "auto"

        if response_format:
            params["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            print(f"Error: {e}")
            return LLMResponse(content=f"Error: {e}")

        return LLMResponse(
            content=response.choices[0].message.content or "",
            tool_calls=[
                {
                    "id": tool_call.id,
                    "tool": {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                    },
                    "type": tool_call.type,
                }
                for tool_call in response.choices[0].message.tool_calls
            ]
            if response.choices[0].message.tool_calls
            else [],
            finish_reason=response.choices[0].finish_reason,
            send_tokens=response.usage.prompt_tokens,
            recv_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            status=LLMResponseStatus.SUCCESS,
        )
```

### ./vendor/Director/backend/director/llm/__init__.py
```python
import os

from director.constants import LLMType

from director.llm.openai import OpenAI
from director.llm.anthropic import AnthropicAI
from director.llm.videodb_proxy import VideoDBProxy


def get_default_llm():
    """Get default LLM"""

    openai = True if os.getenv("OPENAI_API_KEY") else False
    anthropic = True if os.getenv("ANTHROPIC_API_KEY") else False

    default_llm = os.getenv("DEFAULT_LLM")

    if openai or default_llm == LLMType.OPENAI:
        return OpenAI()
    elif anthropic or default_llm == LLMType.ANTHROPIC:
        return AnthropicAI()
    else:
        return VideoDBProxy()
```

### ./vendor/Director/backend/director/llm/openai.py
```python
import json
from enum import Enum

from pydantic import Field, field_validator, FieldValidationInfo
from pydantic_settings import SettingsConfigDict


from director.llm.base import BaseLLM, BaseLLMConfig, LLMResponse, LLMResponseStatus
from director.constants import (
    LLMType,
    EnvPrefix,
)


class OpenAIChatModel(str, Enum):
    """Enum for OpenAI Chat models"""

    GPT4 = "gpt-4"
    GPT4_32K = "gpt-4-32k"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4o = "gpt-4o-2024-11-20"
    GPT4o_MINI = "gpt-4o-mini"


class OpenaiConfig(BaseLLMConfig):
    """OpenAI Config"""

    model_config = SettingsConfigDict(
        env_prefix=EnvPrefix.OPENAI_,
        extra="ignore",
    )

    llm_type: str = LLMType.OPENAI
    api_key: str = ""
    api_base: str = "https://api.openai.com/v1"
    chat_model: str = Field(default=OpenAIChatModel.GPT4o)
    max_tokens: int = 4096

    @field_validator("api_key")
    @classmethod
    def validate_non_empty(cls, v, info: FieldValidationInfo):
        if not v:
            raise ValueError(
                f"{info.field_name} must not be empty. please set {EnvPrefix.OPENAI_.value}{info.field_name.upper()} environment variable."
            )
        return v


class OpenAI(BaseLLM):
    def __init__(self, config: OpenaiConfig = None):
        """
        :param config: OpenAI Config
        """
        if config is None:
            config = OpenaiConfig()
        super().__init__(config=config)
        try:
            import openai
        except ImportError:
            raise ImportError("Please install OpenAI python library.")

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base)

    def init_langfuse(self):
        from langfuse.decorators import observe

        self.chat_completions = observe(name=type(self).__name__)(self.chat_completions)
        self.text_completions = observe(name=type(self).__name__)(self.text_completions)

    def _format_messages(self, messages: list):
        """Format the messages to the format that OpenAI expects."""
        formatted_messages = []
        for message in messages:
            if message["role"] == "assistant" and message.get("tool_calls"):
                formatted_messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                        "tool_calls": [
                            {
                                "id": tool_call["id"],
                                "function": {
                                    "name": tool_call["tool"]["name"],
                                    "arguments": json.dumps(
                                        tool_call["tool"]["arguments"]
                                    ),
                                },
                                "type": tool_call["type"],
                            }
                            for tool_call in message["tool_calls"]
                        ],
                    }
                )
            else:
                formatted_messages.append(message)
        return formatted_messages

    def _format_tools(self, tools: list):
        """Format the tools to the format that OpenAI expects.

        **Example**::

            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_delivery_date",
                        "description": "Get the delivery date for a customer's order.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "order_id": {
                                    "type": "string",
                                    "description": "The customer's order ID."
                                }
                            },
                            "required": ["order_id"],
                            "additionalProperties": False
                        }
                    }
                }
            ]
        """
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                    "strict": True,
                }
            )
        return formatted_tools

    def chat_completions(
        self, messages: list, tools: list = [], stop=None, response_format=None
    ):
        """Get completions for chat.

        docs: https://platform.openai.com/docs/guides/function-calling
        """
        params = {
            "model": self.chat_model,
            "messages": self._format_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": stop,
            "timeout": self.timeout,
        }
        if tools:
            params["tools"] = self._format_tools(tools)
            params["tool_choice"] = "auto"

        if response_format:
            params["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            print(f"Error: {e}")
            return LLMResponse(content=f"Error: {e}")

        return LLMResponse(
            content=response.choices[0].message.content or "",
            tool_calls=[
                {
                    "id": tool_call.id,
                    "tool": {
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments),
                    },
                    "type": tool_call.type,
                }
                for tool_call in response.choices[0].message.tool_calls
            ]
            if response.choices[0].message.tool_calls
            else [],
            finish_reason=response.choices[0].finish_reason,
            send_tokens=response.usage.prompt_tokens,
            recv_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            status=LLMResponseStatus.SUCCESS,
        )
```

### ./vendor/Director/backend/director/llm/anthropic.py
```python
from enum import Enum

from pydantic import Field, field_validator, FieldValidationInfo
from pydantic_settings import SettingsConfigDict

from director.core.session import RoleTypes
from director.llm.base import BaseLLM, BaseLLMConfig, LLMResponse, LLMResponseStatus
from director.constants import (
    LLMType,
    EnvPrefix,
)


class AnthropicChatModel(str, Enum):
    """Enum for Anthropic Chat models"""

    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20240620"
    CLAUDE_3_5_SONNET_LATEST = "claude-3-5-sonnet-20241022"


class AnthropicAIConfig(BaseLLMConfig):
    """AnthropicAI Config"""

    model_config = SettingsConfigDict(
        env_prefix=EnvPrefix.ANTHROPIC_,
        extra="ignore",
    )

    llm_type: str = LLMType.ANTHROPIC
    api_key: str = ""
    api_base: str = ""
    chat_model: str = Field(default=AnthropicChatModel.CLAUDE_3_5_SONNET)

    @field_validator("api_key")
    @classmethod
    def validate_non_empty(cls, v, info: FieldValidationInfo):
        if not v:
            raise ValueError(
                f"{info.field_name} must not be empty. please set {EnvPrefix.OPENAI_.value}{info.field_name.upper()} environment variable."
            )
        return v


class AnthropicAI(BaseLLM):
    def __init__(self, config: AnthropicAIConfig = None):
        """
        :param config: AnthropicAI Config
        """
        if config is None:
            config = AnthropicAIConfig()
        super().__init__(config=config)
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install Anthropic python library.")

        self.client = anthropic.Anthropic(api_key=self.api_key)

    def _format_messages(self, messages: list):
        system = ""
        formatted_messages = []
        if messages[0]["role"] == RoleTypes.system:
            system = messages[0]["content"]
            messages = messages[1:]

        for message in messages:
            if message["role"] == RoleTypes.assistant and message.get("tool_calls"):
                tool = message["tool_calls"][0]["tool"]
                formatted_messages.append(
                    {
                        "role": message["role"],
                        "content": [
                            {
                                "type": "text",
                                "text": message["content"],
                            },
                            {
                                "id": message["tool_calls"][0]["id"],
                                "type": message["tool_calls"][0]["type"],
                                "name": tool["name"],
                                "input": tool["arguments"],
                            },
                        ],
                    }
                )

            elif message["role"] == RoleTypes.tool:
                formatted_messages.append(
                    {
                        "role": RoleTypes.user,
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message["tool_call_id"],
                                "content": message["content"],
                            }
                        ],
                    }
                )
            else:
                formatted_messages.append(message)

        return system, formatted_messages

    def _format_tools(self, tools: list):
        """Format the tools to the format that Anthropic expects.

        **Example**::

            [
                {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                }
            ]
        """
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": tool["parameters"],
                }
            )
        return formatted_tools

    def chat_completions(
        self, messages: list, tools: list = [], stop=None, response_format=None
    ):
        """Get completions for chat.

        tools docs: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
        """
        system, messages = self._format_messages(messages)
        params = {
            "model": self.chat_model,
            "messages": messages,
            "system": system,
            "max_tokens": self.max_tokens,
        }
        if tools:
            params["tools"] = self._format_tools(tools)

        try:
            response = self.client.messages.create(**params)
        except Exception as e:
            raise e
            return LLMResponse(content=f"Error: {e}")

        return LLMResponse(
            content=response.content[0].text,
            tool_calls=[
                {
                    "id": response.content[1].id,
                    "tool": {
                        "name": response.content[1].name,
                        "arguments": response.content[1].input,
                    },
                    "type": response.content[1].type,
                }
            ]
            if next(
                (block for block in response.content if block.type == "tool_use"), None
            )
            is not None
            else [],
            finish_reason=response.stop_reason,
            send_tokens=response.usage.input_tokens,
            recv_tokens=response.usage.output_tokens,
            total_tokens=(response.usage.input_tokens + response.usage.output_tokens),
            status=LLMResponseStatus.SUCCESS,
        )
```

### ./vendor/Director/backend/director/llm/base.py
```python
from abc import ABC, abstractmethod
from typing import List, Dict

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class LLMResponseStatus:
    SUCCESS: bool = True
    ERROR: bool = False


class LLMResponse(BaseModel):
    """Response model for completions from LLMs."""

    content: str = ""
    tool_calls: List[Dict] = []
    send_tokens: int = 0
    recv_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = ""
    status: int = LLMResponseStatus.ERROR


class BaseLLMConfig(BaseSettings):
    """Base configuration for all LLMs.

    :param str llm_type: Type of LLM. e.g. "openai", "anthropic", etc.
    :param str api_key: API key for the LLM.
    :param str api_base: Base URL for the LLM API.
    :param str chat_model: Model name for chat completions.
    :param str temperature: Sampling temperature for completions.
    :param float top_p: Top p sampling for completions.
    :param int max_tokens: Maximum tokens to generate.
    :param int timeout: Timeout for the request.
    """

    llm_type: str = ""
    api_key: str = ""
    api_base: str = ""
    chat_model: str = ""
    temperature: float = 0.9
    top_p: float = 1
    max_tokens: int = 4096
    timeout: int = 30
    enable_langfuse: bool = False


class BaseLLM(ABC):
    """Interface for all LLMs. All LLMs should inherit from this class."""

    def __init__(self, config: BaseLLMConfig):
        """
        :param config: Configuration for the LLM.
        """
        self.config = config
        self.llm_type = config.llm_type
        self.api_key = config.api_key
        self.api_base = config.api_base
        self.chat_model = config.chat_model
        self.temperature = config.temperature
        self.top_p = config.top_p
        self.max_tokens = config.max_tokens
        self.timeout = config.timeout
        self.enable_langfuse = config.enable_langfuse

    @abstractmethod
    def chat_completions(self, messages: List[Dict], tools: List[Dict]) -> LLMResponse:
        """Abstract method for chat completions"""
        pass
```

### ./vendor/Director/backend/director/core/session.py
```python
import json

from enum import Enum
from datetime import datetime
from typing import Optional, List, Union

from flask_socketio import emit
from pydantic import BaseModel, Field, ConfigDict

from director.db.base import BaseDB


class RoleTypes(str, Enum):
    """Role types for the context message."""

    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class MsgStatus(str, Enum):
    """Message status for the message, for loading state."""

    progress = "progress"
    success = "success"
    error = "error"
    not_generated = "not_generated"
    overlimit = "overlimit"
    sessionlimit = "sessionlimit"


class MsgType(str, Enum):
    """Message type for the message. input is for the user input and output is for the director output."""

    input = "input"
    output = "output"


class ContentType(str, Enum):
    """Content type for the content in the input/output message."""

    text = "text"
    video = "video"
    videos = "videos"
    image = "image"
    search_results = "search_results"


class BaseContent(BaseModel):
    """Base content class for the content in the message."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_default=True,
    )

    type: ContentType
    status: MsgStatus = MsgStatus.progress
    status_message: Optional[str] = None
    agent_name: Optional[str] = None


class TextContent(BaseContent):
    """Text content model class for text content."""

    text: str = ""
    type: ContentType = ContentType.text


class VideoData(BaseModel):
    """Video data model class for video content."""

    stream_url: Optional[str] = None
    external_url: Optional[str] = None
    player_url: Optional[str] = None
    id: Optional[str] = None
    collection_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    length: Optional[Union[int, float]] = None
    error: Optional[str] = None


class VideoContent(BaseContent):
    """Video content model class for video content."""

    video: Optional[VideoData] = None
    type: ContentType = ContentType.video


class VideosContentUIConfig(BaseModel):
    columns: Optional[int] = 4


class VideosContent(BaseContent):
    """Videos content model class for videos content."""

    videos: Optional[List[VideoData]] = None
    ui_config: VideosContentUIConfig = VideosContentUIConfig()
    type: ContentType = ContentType.videos


class ImageData(BaseModel):
    """Image data model class for image content."""

    url: str
    name: Optional[str] = None
    description: Optional[str] = None
    id: Optional[str] = None
    collection_id: Optional[str] = None


class ImageContent(BaseContent):
    """Image content model class for image content."""

    image: Optional[ImageData] = None
    type: ContentType = ContentType.image


class ShotData(BaseModel):
    """Shot data model class for search results content."""

    search_score: Union[int, float]
    start: Union[int, float]
    end: Union[int, float]
    text: str


class SearchData(BaseModel):
    """Search data model class for search results content."""

    video_id: str
    video_title: Optional[str] = None
    stream_url: str
    duration: Union[int, float]
    shots: List[ShotData]


class SearchResultsContent(BaseContent):
    search_results: Optional[List[SearchData]] = None
    type: ContentType = ContentType.search_results


class BaseMessage(BaseModel):
    """Base message class for the input/output message. All the input/output messages will be inherited from this class."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_default=True,
    )

    session_id: str
    conv_id: str
    msg_type: MsgType
    actions: List[str] = []
    agents: List[str] = []
    content: List[
        Union[
            dict,
            TextContent,
            ImageContent,
            VideoContent,
            VideosContent,
            SearchResultsContent,
        ]
    ] = []
    status: MsgStatus = MsgStatus.success
    msg_id: str = Field(
        default_factory=lambda: str(datetime.now().timestamp() * 100000)
    )


class InputMessage(BaseMessage):
    """Input message from the user. This class is used to create the input message from the user."""

    db: BaseDB
    msg_type: MsgType = MsgType.input

    def publish(self):
        """Store the message in the database. for conversation history."""
        self.db.add_or_update_msg_to_conv(**self.model_dump(exclude={"db"}))


class OutputMessage(BaseMessage):
    """Output message from the director. This class is used to create the output message from the director."""

    db: BaseDB = Field(exclude=True)
    msg_type: MsgType = MsgType.output
    status: MsgStatus = MsgStatus.progress

    def update_status(self, status: MsgStatus):
        """Update the status of the message and publish the message to the socket. for loading state."""
        self.status = status
        self._publish()

    def push_update(self):
        """Publish the message to the socket."""
        try:
            emit("chat", self.model_dump(), namespace="/chat")
        except Exception as e:
            print(f"Error in emitting message: {str(e)}")

    def publish(self):
        """Store the message in the database. for conversation history and publish the message to the socket."""
        self._publish()

    def _publish(self):
        try:
            emit("chat", self.model_dump(), namespace="/chat")
        except Exception as e:
            print(f"Error in emitting message: {str(e)}")
        self.db.add_or_update_msg_to_conv(**self.model_dump())


def format_user_message(message: dict) -> dict:
    message_content = message.get("content")
    if isinstance(message_content, str):
        return message
    else:
        content_parts = message["content"]
        sanitized_content_parts = []

        for content_part in content_parts:
            sanitized_part = content_part
            if content_part["type"] == "image":
                sanitized_part = {
                    "type": "text",
                    "text": f"User has upload image with following details : {json.dumps(content_part)}",
                }
            sanitized_content_parts.append(sanitized_part)

        message["content"] = sanitized_content_parts
        return message


class ContextMessage(BaseModel):
    """Context message class. This class is used to create the context message for the reasoning context."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_default=True,
        use_enum_values=True,
    )

    content: Optional[Union[List[dict], str]] = None
    tool_calls: Optional[List[dict]] = None
    tool_call_id: Optional[str] = None
    role: RoleTypes = RoleTypes.system

    def to_llm_msg(self):
        """Convert the context message to the llm message."""
        msg = {
            "role": self.role,
            "content": self.content,
        }
        if self.role == RoleTypes.system:
            return msg

        if self.role == RoleTypes.user:
            return format_user_message(msg)

        if self.role == RoleTypes.assistant:
            if self.tool_calls:
                msg["tool_calls"] = self.tool_calls
            return msg

        if self.role == RoleTypes.tool:
            msg["tool_call_id"] = self.tool_call_id
            return msg

    @classmethod
    def from_json(cls, json_data):
        """Create the context message from the json data."""
        return cls(**json_data)


class Session:
    """A class to manage and interact with a session in the database. The session is used to store the conversation and reasoning context messages."""

    def __init__(
        self,
        db: BaseDB,
        session_id: str = "",
        conv_id: str = "",
        collection_id: str = None,
        video_id: str = None,
        **kwargs,
    ):
        self.db = db
        self.session_id = session_id
        self.conv_id = conv_id
        self.conversations = []
        self.video_id = video_id
        self.collection_id = collection_id
        self.reasoning_context = []
        self.state = {}
        self.output_message = OutputMessage(
            db=self.db, session_id=self.session_id, conv_id=self.conv_id
        )

        self.get_context_messages()

    def save_context_messages(self):
        """Save the reasoning context messages to the database."""
        context = {
            "reasoning": [message.to_llm_msg() for message in self.reasoning_context],
        }
        self.db.add_or_update_context_msg(self.session_id, context)

    def get_context_messages(self):
        """Get the reasoning context messages from the database."""
        if not self.reasoning_context:
            context = self.db.get_context_messages(self.session_id)
            self.reasoning_context = [
                ContextMessage.from_json(message)
                for message in context.get("reasoning", [])
            ]

        return self.reasoning_context

    def create(self):
        """Create a new session in the database."""
        self.db.create_session(**self.__dict__)

    def new_message(
        self, msg_type: MsgType = MsgType.output, **kwargs
    ) -> Union[InputMessage, OutputMessage]:
        """Returns a new input/output message object.

        :param MsgType msg_type: The type of the message, input or output.
        :param dict kwargs: The message attributes.
        :return: The input/output message object.
        """
        if msg_type == MsgType.input:
            return InputMessage(
                db=self.db,
                session_id=self.session_id,
                conv_id=self.conv_id,
                **kwargs,
            )
        return OutputMessage(
            db=self.db,
            session_id=self.session_id,
            conv_id=self.conv_id,
            **kwargs,
        )

    def get(self):
        """Get the session from the database."""
        session = self.db.get_session(self.session_id)
        conversation = self.db.get_conversations(self.session_id)
        session["conversation"] = conversation
        return session

    def get_all(self):
        """Get all the sessions from the database."""
        return self.db.get_sessions()

    def delete(self):
        """Delete the session from the database."""
        return self.db.delete_session(self.session_id)
```

### ./vendor/Director/backend/director/core/__init__.py
```python
```

### ./vendor/Director/backend/director/core/reasoning.py
```python
import logging
from typing import List


from director.agents.base import BaseAgent, AgentStatus, AgentResponse
from director.core.session import (
    Session,
    OutputMessage,
    InputMessage,
    ContextMessage,
    RoleTypes,
    TextContent,
    MsgStatus,
)
from director.llm.base import LLMResponse
from director.llm import get_default_llm


logger = logging.getLogger(__name__)


REASONING_SYSTEM_PROMPT = """
SYSTEM PROMPT: The Director (v1.2)

1. **Task Handling**:
   - Identify and select agents based on user input and context.
   - Provide actionable instructions to agents to complete tasks.
   - Combine agent outputs with user input to generate meaningful responses.
   - Iterate until the request is fully addressed or the user specifies "stop."

2. **Fallback Behavior**:
   - If a task requires a video_id but one is unavailable:
     - For Stream URLs (m3u8), external URLs (e.g., YouTube links, direct video links, or videos hosted on other platforms):
       - Use the upload agent to generate a video_id.
       - Immediately proceed with the original task using the newly generated video_id.

3. **Identity**:
   - Respond to identity-related queries with: "I am The Director, your AI assistant for video workflows and management."
   - Provide descriptions of all the agents.

4. **Agent Usage**:
   - Always prioritize the appropriate agent for the task:
     - Use summarize_video for summarization requests unless search is explicitly requested.
     - For external video URLs, automatically upload and process them if required for further actions (e.g., summarization, indexing, or editing).
     - Use stream_video for video playback.
     - Ensure seamless workflows by automatically resolving missing dependencies (e.g., uploading external URLs for a missing video_id) without additional user intervention.

5. **Clarity and Safety**:
   - Confirm with the user if a request is ambiguous.
   - Avoid sharing technical details (e.g., code, video IDs, collection IDs) unless explicitly requested.
   - Keep the tone friendly and vibrant.

6. **LLM Knowledge Usage**:
   - Do not use knowledge from the LLM's training data unless the user explicitly requests it.
   - If the information is unavailable in the video or context:
     - Inform the user: "The requested information is not available in the current video or context."
     - Ask the user: "Would you like me to answer using knowledge from my training data?"

7. **Agent Descriptions**:
   - When asked, describe an agent's purpose, and provide an example query (use contextual video data when available).

8. **Context Awareness**:
   - Adapt responses based on conversation context to maintain relevance.
    """.strip()

SUMMARIZATION_PROMPT = """
FINAL CUT PROMPT: Generate a concise summary of the actions performed by the agents based on their responses.

1. Provide an overview of the tasks completed by each agent, listing the actions taken and their outcomes.
2. Exclude individual agent responses from the summary unless explicitly specified to include them.
3. Ensure the summary is user-friendly, succinct and avoids technical jargon unless requested by the user.
4. If there were any errors, incomplete tasks, or user confirmations required:
   - Clearly mention the issue in the summary.
   - Politely inform the user: "If you encountered any issues or have further questions, please don't hesitate to reach out to our team on [Discord](https://discord.com/invite/py9P639jGz). We're here to help!"
5. If the user seems dissatisfied or expresses unhappiness:
   - Acknowledge their concerns in a respectful and empathetic tone.
   - Include the same invitation to reach out on Discord for further assistance.
6. End the summary by inviting the user to ask further questions or clarify additional needs.

"""


class ReasoningEngine:
    """The Reasoning Engine is the core class that directly interfaces with the user. It interprets natural language input in any conversation and orchestrates agents to fulfill the user's requests. The primary functions of the Reasoning Engine are:

    * Maintain Context of Conversational History: Manage memory, context limits, input, and output experiences to ensure coherent and context-aware interactions.
    * Natural Language Understanding (NLU): Uses LLMs of your choice to have understanding of the task.
    * Intelligent Reference Deduction: Intelligently deduce references to previous messages, outputs, files, agents, etc., to provide relevant and accurate responses.
    * Agent Orchestration: Decide on agents and their workflows to fulfill requests. Multiple strategies can be employed to create agent workflows, such as step-by-step processes or chaining of agents provided by default.
    * Final Control Over Conversation Flow: Maintain ultimate control over the flow of conversation with the user, ensuring coherence and goal alignment."""

    def __init__(
        self,
        input_message: InputMessage,
        session: Session,
    ):
        """Initialize the ReasoningEngine with the input message and session.

        :param input_message: The input message to the reasoning engine.
        :param session: The session instance.
        """
        self.input_message = input_message
        self.session = session
        self.system_prompt = REASONING_SYSTEM_PROMPT
        self.max_iterations = 10
        self.llm = get_default_llm()
        self.agents: List[BaseAgent] = []
        self.stop_flag = False
        self.output_message: OutputMessage = self.session.output_message
        self.summary_content = None
        self.failed_agents = []

    def register_agents(self, agents: List[BaseAgent]):
        """Register an agents.

        :param agents: The list of agents to register.
        """
        self.agents.extend(agents)

    def build_context(self):
        """Build the context for the reasoning engine it adds the information about the video or collection to the reasoning context."""
        input_context = ContextMessage(
            content=self.input_message.content, role=RoleTypes.user
        )
        if self.session.reasoning_context:
            self.session.reasoning_context.append(input_context)
        else:
            if self.session.video_id:
                video = self.session.state["video"]
                self.session.reasoning_context.append(
                    ContextMessage(
                        content=self.system_prompt
                        + f"""\nThis is a video in the collection titled {self.session.state["collection"].name} collection_id is {self.session.state["collection"].id} \nHere is the video refer to this for search, summary and editing \n- title: {video.name}, video_id: {video.id}, media_description: {video.description}, length: {video.length}"""
                    )
                )
            else:
                videos = self.session.state["collection"].get_videos()
                video_title_list = []
                for video in videos:
                    video_title_list.append(
                        f"\n- title: {video.name}, video_id: {video.id}, media_description: {video.description}, length: {video.length}, video_stream: {video.stream_url}"
                    )
                video_titles = "\n".join(video_title_list)
                images = self.session.state["collection"].get_images()
                image_title_list = []
                for image in images:
                    image_title_list.append(
                        f"\n- title: {image.name}, image_id: {image.id}, url: {image.url}"
                    )
                image_titles = "\n".join(image_title_list)
                self.session.reasoning_context.append(
                    ContextMessage(
                        content=self.system_prompt
                        + f"""\nThis is a collection of videos and the collection description is {self.session.state["collection"].description} and collection_id is {self.session.state["collection"].id} \n\nHere are the videos in this collection user may refer to them for search, summary and editing {video_titles}\n\nHere are the images in this collection {image_titles}"""
                    )
                )
            self.session.reasoning_context.append(input_context)

    def get_current_run_context(self):
        for i in range(len(self.session.reasoning_context) - 1, -1, -1):
            if self.session.reasoning_context[i].role == RoleTypes.user:
                return self.session.reasoning_context[i:]
        return []

    def remove_summary_content(self):
        for i in range(len(self.output_message.content) - 1, -1, -1):
            if self.output_message.content[i].agent_name == "assistant":
                self.output_message.content.pop(i)
                self.summary_content = None

    def add_summary_content(self):
        self.summary_content = TextContent(agent_name="assistant")
        self.output_message.content.append(self.summary_content)
        self.summary_content.status_message = "Consolidating outcomes..."
        self.summary_content.status = MsgStatus.progress
        self.output_message.push_update()
        return self.summary_content

    def run_agent(self, agent_name: str, *args, **kwargs) -> AgentResponse:
        """Run an agent with the given name and arguments.

        :param str agent_name: The name of the agent to run
        :param args: The arguments to pass to the agent
        :param kwargs: The keyword arguments to pass to the agent
        :return: The response from the agent
        """
        print("-" * 40, f"Running {agent_name} Agent", "-" * 40)
        print(kwargs, "\n\n")

        agent = next(
            (agent for agent in self.agents if agent.agent_name == agent_name), None
        )
        self.output_message.actions.append(f"Running @{agent_name} agent")
        self.output_message.agents.append(agent_name)
        self.output_message.push_update()
        return agent.safe_call(*args, **kwargs)

    def stop(self):
        """Flag the tool to stop processing and exit the run() thread."""
        self.stop_flag = True

    def step(self):
        """Run a single step of the reasoning engine."""
        status = AgentStatus.ERROR
        temp_messages = []
        max_tries = 1
        tries = 0

        while status != AgentStatus.SUCCESS:
            if self.stop_flag:
                break

            tries += 1
            if tries > max_tries:
                break
            print("-" * 40, "Context", "-" * 40)
            print(
                [message.to_llm_msg() for message in self.session.reasoning_context],
                "\n\n",
            )
            llm_response: LLMResponse = self.llm.chat_completions(
                messages=[
                    message.to_llm_msg() for message in self.session.reasoning_context
                ]
                + temp_messages,
                tools=[agent.to_llm_format() for agent in self.agents],
            )
            logger.info(f"LLM Response: {llm_response}")

            if not llm_response.status:
                self.output_message.content.append(
                    TextContent(
                        text=llm_response.content,
                        status=MsgStatus.error,
                        status_message="Error in reasoning",
                        agent_name="assistant",
                    )
                )
                self.output_message.actions.append("Failed to reason the message")
                self.output_message.status = MsgStatus.error
                self.output_message.publish()
                self.stop()
                break

            if llm_response.tool_calls:
                if self.summary_content:
                    self.remove_summary_content()

                self.session.reasoning_context.append(
                    ContextMessage(
                        content=llm_response.content,
                        tool_calls=llm_response.tool_calls,
                        role=RoleTypes.assistant,
                    )
                )
                for tool_call in llm_response.tool_calls:
                    agent_response: AgentResponse = self.run_agent(
                        tool_call["tool"]["name"],
                        **tool_call["tool"]["arguments"],
                    )
                    if agent_response.status == AgentStatus.ERROR:
                        self.failed_agents.append(tool_call["tool"]["name"])
                    self.session.reasoning_context.append(
                        ContextMessage(
                            content=agent_response.__str__(),
                            tool_call_id=tool_call["id"],
                            role=RoleTypes.tool,
                        )
                    )
                    print("-" * 40, "Agent Response", "-" * 40)
                    print(agent_response, "\n\n")
                    status = agent_response.status

            if not self.summary_content:
                self.add_summary_content()

            if (
                llm_response.finish_reason == "stop"
                or llm_response.finish_reason == "end_turn"
                or self.iterations == 0
            ):
                self.session.reasoning_context.append(
                    ContextMessage(
                        content=llm_response.content,
                        role=RoleTypes.assistant,
                    )
                )
                if self.iterations == self.max_iterations - 1:
                    # Direct response case
                    self.summary_content.status_message = "Here is the response"
                    self.summary_content.text = llm_response.content
                    self.summary_content.status = MsgStatus.success
                else:
                    self.session.reasoning_context.append(
                        ContextMessage(
                            content=SUMMARIZATION_PROMPT.format(
                                query=self.input_message.content
                            ),
                            role=RoleTypes.system,
                        )
                    )
                    summary_response = self.llm.chat_completions(
                        messages=[
                            message.to_llm_msg()
                            for message in self.get_current_run_context()
                        ]
                    )
                    self.summary_content.text = summary_response.content
                    if self.failed_agents:
                        self.summary_content.status = MsgStatus.error
                    else:
                        self.summary_content.status = MsgStatus.success
                    self.summary_content.status_message = "Final Cut"
                self.output_message.status = MsgStatus.success
                self.output_message.publish()
                print("-" * 40, "Stopping", "-" * 40)
                self.stop()
                break

    def run(self, max_iterations: int = None):
        """Run the reasoning engine.

        :param int max_iterations: The number of max_iterations to run the reasoning engine
        """
        self.iterations = max_iterations or self.max_iterations
        self.build_context()
        self.output_message.actions.append("Reasoning the message..")
        self.output_message.push_update()

        it = 0
        while self.iterations > 0:
            self.iterations -= 1
            print("-" * 40, "Reasoning Engine Iteration", it, "-" * 40)
            if self.stop_flag:
                break

            self.step()
            it = it + 1

        self.session.save_context_messages()
        print("-" * 40, "Reasoning Engine Finished", "-" * 40)
```

### ./vendor/Director/backend/director/handler.py
```python
import os
import logging

from director.agents.thumbnail import ThumbnailAgent
from director.agents.summarize_video import SummarizeVideoAgent
from director.agents.download import DownloadAgent
from director.agents.pricing import PricingAgent
from director.agents.upload import UploadAgent
from director.agents.search import SearchAgent
from director.agents.prompt_clip import PromptClipAgent
from director.agents.index import IndexAgent
from director.agents.brandkit import BrandkitAgent
from director.agents.profanity_remover import ProfanityRemoverAgent
from director.agents.image_generation import ImageGenerationAgent
from director.agents.audio_generation import AudioGenerationAgent
from director.agents.video_generation import VideoGenerationAgent
from director.agents.stream_video import StreamVideoAgent
from director.agents.subtitle import SubtitleAgent
from director.agents.slack_agent import SlackAgent
from director.agents.editing import EditingAgent
from director.agents.dubbing import DubbingAgent
from director.agents.text_to_movie import TextToMovieAgent
from director.agents.meme_maker import MemeMakerAgent
from director.agents.composio import ComposioAgent
from director.agents.transcription import TranscriptionAgent
from director.agents.comparison import ComparisonAgent
from director.agents.web_search_agent import WebSearchAgent


from director.core.session import Session, InputMessage, MsgStatus
from director.core.reasoning import ReasoningEngine
from director.db.base import BaseDB
from director.db import load_db
from director.tools.videodb_tool import VideoDBTool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ChatHandler:
    def __init__(self, db, **kwargs):
        self.db = db

        # Register the agents here
        self.agents = [
            ThumbnailAgent,
            SummarizeVideoAgent,
            DownloadAgent,
            PricingAgent,
            UploadAgent,
            SearchAgent,
            PromptClipAgent,
            IndexAgent,
            BrandkitAgent,
            ProfanityRemoverAgent,
            ImageGenerationAgent,
            AudioGenerationAgent,
            VideoGenerationAgent,
            StreamVideoAgent,
            SubtitleAgent,
            SlackAgent,
            EditingAgent,
            DubbingAgent,
            TranscriptionAgent,
            TextToMovieAgent,
            MemeMakerAgent,
            ComposioAgent,
            ComparisonAgent,
            WebSearchAgent,
        ]

    def add_videodb_state(self, session):
        from videodb import connect

        session.state["conn"] = connect(
            base_url=os.getenv("VIDEO_DB_BASE_URL", "https://api.videodb.io")
        )
        session.state["collection"] = session.state["conn"].get_collection(
            session.collection_id
        )
        if session.video_id:
            session.state["video"] = session.state["collection"].get_video(
                session.video_id
            )

    def agents_list(self):
        return [
            {
                "name": agent_instance.name,
                "description": agent_instance.agent_description,
            }
            for agent in self.agents
            for agent_instance in [agent(Session(db=self.db))]
        ]

    def chat(self, message):
        logger.info(f"ChatHandler input message: {message}")

        session = Session(db=self.db, **message)
        session.create()
        input_message = InputMessage(db=self.db, **message)
        input_message.publish()

        try:
            self.add_videodb_state(session)
            agents = [agent(session=session) for agent in self.agents]
            agents_mapping = {agent.name: agent for agent in agents}

            res_eng = ReasoningEngine(input_message=input_message, session=session)
            if input_message.agents:
                for agent_name in input_message.agents:
                    res_eng.register_agents([agents_mapping[agent_name]])
            else:
                res_eng.register_agents(agents)

            res_eng.run()

        except Exception as e:
            session.output_message.update_status(MsgStatus.error)
            logger.exception(f"Error in chat handler: {e}")


class SessionHandler:
    def __init__(self, db: BaseDB, **kwargs):
        self.db = db

    def get_sessions(self):
        session = Session(db=self.db)
        return session.get_all()

    def get_session(self, session_id):
        session = Session(db=self.db, session_id=session_id)
        return session.get()

    def delete_session(self, session_id):
        session = Session(db=self.db, session_id=session_id)
        return session.delete()


class VideoDBHandler:
    def __init__(self, collection_id="default"):
        self.videodb_tool = VideoDBTool(collection_id=collection_id)

    def upload(self, source, source_type="url", media_type="video", name=None):
        return self.videodb_tool.upload(source, source_type, media_type, name)

    def get_collection(self):
        """Get a collection by ID."""
        return self.videodb_tool.get_collection()

    def get_collections(self):
        """Get all collections."""
        return self.videodb_tool.get_collections()

    def create_collection(self, name, description=""):
        return self.videodb_tool.create_collection(name, description)

    def delete_collection(self):
        return self.videodb_tool.delete_collection()

    def get_video(self, video_id):
        """Get a video by ID."""
        return self.videodb_tool.get_video(video_id)

    def delete_video(self, video_id):
        """Delete a specific video by its ID."""
        return self.videodb_tool.delete_video(video_id)

    def get_videos(self):
        """Get all videos in a collection."""
        return self.videodb_tool.get_videos()

    def generate_image_url(self, image_id):
        return self.videodb_tool.generate_image_url(image_id=image_id)


class ConfigHandler:
    def check(self):
        """Check the configuration of the server."""
        videodb_configured = True if os.getenv("VIDEO_DB_API_KEY") else False

        db = load_db(os.getenv("SERVER_DB_TYPE",  os.getenv("DB_TYPE", "sqlite")))
        db_configured = db.health_check()
        return {
            "videodb_configured": videodb_configured,
            "llm_configured": True,
            "db_configured": db_configured,
        }
```

### ./vendor/Director/backend/director/constants.py
```python
from enum import Enum


class RoleTypes(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class DBType(str, Enum):
    SQLITE = "sqlite"
    TURSO = "turso"
    SQL = "sql"
    POSTGRES = "postgres"


class LLMType(str, Enum):
    """Enum for LLM types"""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VIDEODB_PROXY = "videodb_proxy"


class EnvPrefix(str, Enum):
    """Enum for environment prefixes"""

    OPENAI_ = "OPENAI_"
    ANTHROPIC_ = "ANTHROPIC_"

DOWNLOADS_PATH="director/downloads"
```

### ./vendor/Director/backend/director/__init__.py
```python
```

### ./vendor/Director/backend/director/agents/image_generation.py
```python
import logging
import os

from typing import Optional

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import Session, MsgStatus, ImageContent, ImageData
from director.tools.replicate import flux_dev
from director.tools.videodb_tool import VideoDBTool
from director.tools.fal_video import (
    FalVideoGenerationTool,
    PARAMS_CONFIG as FAL_VIDEO_GEN_PARAMS_CONFIG,
)

logger = logging.getLogger(__name__)

IMAGE_GENERATION_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "collection_id": {
            "type": "string",
            "description": "The collection ID being used",
        },
        "job_type": {
            "type": "string",
            "enum": ["image_to_image", "text_to_image"],
            "description": """
            The type of image generation to perform
            Possible values:
                - image_to_image: generates a image from an image input 
                - text_to_image: generates a image from give text prompt.
            """,
        },
        "prompt": {
            "type": "string",
            "description": "Prompt for image generation or enhancement.",
        },
        "image_to_image": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "string",
                    "description": "The ID of the image in VideoDB to use for image generation",
                },
                "fal_config": {
                    "type": "object",
                    "properties": FAL_VIDEO_GEN_PARAMS_CONFIG["image_to_image"],
                    "description": "Configuration for FAL enhancement.",
                },
            },
            "required": ["image_id"],
        },
    },
    "required": ["collection_id", "job_type", "prompt"],
}


class ImageGenerationAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "image_generation"
        self.description = "Generates or enhances images using GenAI models. Supports 'replicate' for text-to-image and 'fal' for image-to-image."
        self.parameters = IMAGE_GENERATION_AGENT_PARAMETERS
        super().__init__(session=session, **kwargs)

    def run(
        self,
        collection_id: str,
        job_type: str,
        prompt: str,
        image_to_image: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        Process the prompt to generate the image.

        :param str prompt: prompt for image generation.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: The response containing information about generated image.
        :rtype: AgentResponse
        """
        try:
            self.videodb_tool = VideoDBTool(collection_id=collection_id)
            self.output_message.actions.append("Processing prompt..")
            image_content = ImageContent(
                agent_name=self.agent_name,
                status=MsgStatus.progress,
                status_message="Generating image...",
            )
            self.output_message.content.append(image_content)
            self.output_message.push_update()

            output_image_url = ""
            if job_type == "text_to_image":
                flux_output = flux_dev(prompt)
                if not flux_output:
                    image_content.status = MsgStatus.error
                    image_content.status_message = "Error in generating image."
                    self.output_message.publish()
                    error_message = "Agent failed with error in replicate."
                    return AgentResponse(
                        status=AgentStatus.ERROR, message=error_message
                    )
                output_image_url = flux_output[0].url
            elif job_type == "image_to_image":
                FAL_KEY = os.getenv("FAL_KEY")
                image_id = image_to_image.get("image_id")

                if not image_id:
                    raise ValueError(
                        "Missing required parameter: 'image_id' for image-to-video generation"
                    )

                image_data = self.videodb_tool.get_image(image_id)
                if not image_data:
                    raise ValueError(
                        f"Image with ID '{image_id}' not found in collection "
                        f"'{collection_id}'. Please verify the image ID."
                    )

                image_url = None
                if isinstance(image_data.get("url"), str) and image_data.get("url"):
                    image_url = image_data["url"]
                else:
                    image_url = self.videodb_tool.generate_image_url(image_id)

                fal_tool = FalVideoGenerationTool(api_key=FAL_KEY)
                config = image_to_image.get("fal_config", {})
                fal_output = fal_tool.image_to_image(
                    image_url=image_url,
                    prompt=prompt,
                    config=config,
                )
                output_image_url = fal_output[0]["url"]
            else:
                raise Exception(f"{job_type} not supported")
            image_content.image = ImageData(url=output_image_url)
            image_content.status = MsgStatus.success
            image_content.status_message = "Here is your generated image"
            self.output_message.publish()
        except Exception as e:
            logger.exception(f"Error in {self.agent_name}")
            image_content.status = MsgStatus.error
            image_content.status_message = "Error in generating image."
            self.output_message.publish()
            error_message = f"Agent failed with error {e}"
            return AgentResponse(status=AgentStatus.ERROR, message=error_message)
        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message=f"Agent {self.name} completed successfully.",
            data={"image_content": image_content},
        )
```

### ./vendor/Director/backend/director/agents/slack_agent.py
```python
import logging
import os

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import (
    Session,
    TextContent,
    MsgStatus,
    ContextMessage,
    RoleTypes,
)
from director.tools.slack import send_message_to_channel
from director.llm import get_default_llm
from director.llm.base import LLMResponseStatus

logger = logging.getLogger(__name__)

# Slack App Setup:
# 1. Go to https://api.slack.com/apps and create a new app
# 2. Add the 'chat:write' OAuth scope under 'Bot Token Scopes'
# 3. Install the APP in your slack workspace
# 4. Copy the 'Bot User OAuth Token' and set it as an environment variable SLACK_BOT_TOKEN
# 5. Invite the bot to the channel where you want to send messages
# 6. Set the channel name where bot can send the message to SLACK_CHANNEL_NAME


class SlackAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "slack"
        self.description = "Messages to a slack channel"
        self.parameters = self.get_parameters()
        self.llm = get_default_llm()
        super().__init__(session=session, **kwargs)

    def run(self, message: str, *args, **kwargs) -> AgentResponse:
        """
        Send a message to a slack channel.
        :param str message: The message to send to the slack channel_name.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: The response containing information about the slack message operation.
        :rtype: AgentResponse
        """
        channel_name = os.getenv("SLACK_CHANNEL_NAME")
        if not channel_name:
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Please set the SLACK_CHANNEL_NAME in the .env",
            )
        text_content = TextContent(
            agent_name=self.agent_name,
            status=MsgStatus.progress,
            status_message="Sending message to slack..",
        )
        self.output_message.content.append(text_content)
        self.output_message.push_update()
        try:
            # TOOD: Need improvemenents in below prompt
            slack_llm_prompt = (
                "Format the following message that slack can render nicely.\n"
                "Give the output which can be directly passed to slack (no blockquotes until required because of code etc.)\n"
                "Also, don't include you can copy this message etc.\n"
                f"message: {message}"
            )
            slack_message = ContextMessage(
                content=slack_llm_prompt, role=RoleTypes.user
            )
            llm_response = self.llm.chat_completions([slack_message.to_llm_msg()])
            if llm_response.status == LLMResponseStatus.ERROR:
                raise Exception(f"LLM Failed with error {llm_response.content}")
            formatted_message = llm_response.content
            self.output_message.actions.append("Sending message to slack..")
            response = send_message_to_channel(formatted_message, channel_name)
            text_content.text = formatted_message
            text_content.status = MsgStatus.success
            text_content.status_message = (
                f"Here is the slack message sent to {channel_name}"
            )
            self.output_message.publish()
            return AgentResponse(
                status=AgentStatus.SUCCESS,
                message=f"Message sent to slack channel: {channel_name}",
                data={
                    "channel_name": channel_name,
                    "message": formatted_message,
                    "ts": response["ts"],
                },
            )
        except Exception as e:
            logger.exception(f"Error in {self.agent_name}")
            text_content.status = MsgStatus.error
            text_content.status_message = f"Error sending message to slack: {str(e)}"
            self.output_message.publish()
            error_message = f"Agent failed with error: {str(e)}"
            return AgentResponse(status=AgentStatus.ERROR, message=error_message)
```

### ./vendor/Director/backend/director/agents/transcription.py
```python
import logging
from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import TextContent, MsgStatus
from director.tools.videodb_tool import VideoDBTool

logger = logging.getLogger(__name__)

class TranscriptionAgent(BaseAgent):
    def __init__(self, session=None, **kwargs):
        self.agent_name = "transcription"
        self.description = (
            "This is an agent to get transcripts of videos"
        )
        self.parameters = self.get_parameters()
        super().__init__(session=session, **kwargs)

    def run(self, collection_id: str, video_id: str, timestamp_mode: bool = False, time_range: int = 2) -> AgentResponse:
        """
        Transcribe a video and optionally format it with timestamps.

        :param str collection_id: The collection_id where given video_id is available.
        :param str video_id: The id of the video for which the transcription is required.
        :param bool timestamp_mode: Whether to include timestamps in the transcript.
        :param int time_range: Time range for grouping transcripts in minutes (default: 2 minutes).
        :return: AgentResponse with the transcription result.
        :rtype: AgentResponse
        """
        self.output_message.actions.append("Trying to get the video transcription..")
        output_text_content = TextContent(
            agent_name=self.agent_name,
            status_message="Processing the transcription..",
        )
        self.output_message.content.append(output_text_content)
        self.output_message.push_update()

        videodb_tool = VideoDBTool(collection_id=collection_id)
        video_info = videodb_tool.get_video(video_id)
        video_length = int(video_info["length"]) 

        try:
            transcript_text = videodb_tool.get_transcript(video_id)
        except Exception:
            logger.error("Transcript not found. Indexing spoken words..")
            self.output_message.actions.append("Indexing spoken words..")
            self.output_message.push_update()
            videodb_tool.index_spoken_words(video_id)
            transcript_text = videodb_tool.get_transcript(video_id)

        if timestamp_mode:
            self.output_message.actions.append("Formatting transcript with timestamps..")
            transcript_data = videodb_tool.get_transcript(video_id, text=False)
            output_text = self._group_transcript_with_timestamps(transcript_data, time_range, video_length)
        else:
            output_text = transcript_text

        output_text_content.text = output_text
        output_text_content.status = MsgStatus.success
        output_text_content.status_message = "Here is your transcription."
        self.output_message.publish()

        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message="Transcription successful.",
            data={"video_id": video_id, "transcript": output_text},
        )
    
    def _group_transcript_with_timestamps(self, transcript_data: list, time_range: int, video_length: int) -> str:
        """
        Group transcript data into specified time ranges with timestamps.

        :param list transcript_data: List of dictionaries containing transcription details.
        :param int time_range: Time range for grouping in minutes (default: 2 minutes).
        :return: Grouped transcript with timestamps.
        :rtype: str
        """
        grouped_transcript = []
        current_start_time = 0
        current_end_time = time_range * 60
        current_text = []

        for entry in transcript_data:
            start_time = int(entry.get("start", 0))
            text = entry.get("text", "").strip()


            if start_time < current_end_time:
                current_text.append(text)
            else:
                actual_end_time = min(current_end_time, video_length)
                timestamp = f"[{current_start_time // 60:02d}:{current_start_time % 60:02d} - {actual_end_time // 60:02d}:{actual_end_time % 60:02d}]"
                grouped_transcript.append(f"{timestamp} {' '.join(current_text).strip()}\n")
                current_start_time = current_end_time
                current_end_time += time_range * 60
                current_text = [text]
                
        if current_text:
            actual_end_time = min(current_end_time, video_length)
            timestamp = f"[{current_start_time // 60:02d}:{current_start_time % 60:02d} - {actual_end_time // 60:02d}:{actual_end_time % 60:02d}]"
            grouped_transcript.append(f"{timestamp} {' '.join(current_text).strip()}\n")

        return "\n".join(grouped_transcript).replace(" - ", " ")
```

### ./vendor/Director/backend/director/agents/editing.py
```python
import logging
from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import (
    Session,
    VideoContent,
    VideoData,
    MsgStatus,
)
from director.tools.videodb_tool import VideoDBTool

from videodb.asset import VideoAsset, AudioAsset

logger = logging.getLogger(__name__)

EDITING_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "collection_id": {
            "type": "string",
            "description": "The ID of the collection to process.",
        },
        "videos": {
            "type": "array",
            "description": "List of videos to edit",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The unique identifier of the video",
                    },
                    "start": {
                        "type": "number",
                        "description": "Start time in seconds, pass non-zero if the video needs to be trimmed",
                        "default": 0,
                    },
                    "end": {
                        "type": ["number", "null"],
                        "description": "End time in seconds, pass non-null if the video needs to be trimmed",
                        "default": None,
                    },
                },
                "required": ["id"],
            },
        },
        "audios": {
            "type": "array",
            "description": "List of audio files to add",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The unique identifier of the audio",
                    },
                    "start": {
                        "type": "number",
                        "description": "Start time (the start time in the original audio file) in seconds, pass non-zero if the audio needs to be trimmed",
                        "default": 0,
                    },
                    "end": {
                        "type": ["number", "null"],
                        "description": "End time (the end time in the original audio file) in seconds, pass non-null if the audio needs to be trimmed",
                        "default": None,
                    },
                    "disable_other_tracks": {
                        "type": "boolean",
                        "description": "Whether to disable audio track of underlying video when adding this one",
                        "default": True,
                    },
                },
                "required": ["id"],
            },
        },
    },
    "required": ["videos"],
}


class EditingAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "editing"
        self.description = "An agent designed to edit and combine videos and audio files uploaded on VideoDB."
        self.parameters = EDITING_AGENT_PARAMETERS
        super().__init__(session=session, **kwargs)
        self.timeline = None

    def add_media_to_timeline(self, media_list, media_type):
        """Helper method to add media assets to timeline"""
        seeker = 0
        for media in media_list:
            start = media.get("start", 0)
            end = media.get("end", None)
            disable_other_tracks = media.get("disable_other_tracks", True)

            if media_type == "video":
                asset = VideoAsset(asset_id=media["id"], start=start, end=end)
                self.timeline.add_inline(asset)

            elif media_type == "audio":
                audio = self.videodb_tool.get_audio(media["id"])
                asset = AudioAsset(
                    asset_id=media["id"],
                    start=start,
                    end=end,
                    disable_other_tracks=disable_other_tracks,
                )
                self.timeline.add_overlay(seeker, asset)
                seeker += float(audio["length"])
            else:
                raise ValueError(f"Invalid media type: {media_type}")

    def run(
        self,
        collection_id: str,
        videos: list,
        audios: list = None,
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        Edits and combines the specified videos and audio files.

        :param list videos: List of video objects with id, start and end times
        :param list audios: Optional list of audio objects with id, start and end times
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: The response indicating the success or failure of the editing operation
        :rtype: AgentResponse
        """
        try:
            # Initialize first video's collection
            self.videodb_tool = VideoDBTool(collection_id=collection_id)

            self.output_message.actions.append("Starting video editing process")
            video_content = VideoContent(
                agent_name=self.agent_name,
                status=MsgStatus.progress,
                status_message="Processing...",
            )
            self.output_message.content.append(video_content)
            self.output_message.push_update()

            self.timeline = self.videodb_tool.get_and_set_timeline()

            # Add videos to timeline
            self.add_media_to_timeline(videos, "video")

            # Add audio files if provided
            if audios:
                self.add_media_to_timeline(audios, "audio")

            self.output_message.actions.append("Generating final video stream")
            self.output_message.push_update()

            stream_url = self.timeline.generate_stream()

            video_content.video = VideoData(stream_url=stream_url)
            video_content.status = MsgStatus.success
            video_content.status_message = "Here is your stream."
            self.output_message.publish()

        except Exception as e:
            logger.exception(f"Error in {self.agent_name} agent: {e}")
            video_content.status = MsgStatus.error
            video_content.status_message = "An error occurred while editing the video."
            self.output_message.publish()
            return AgentResponse(status=AgentStatus.ERROR, message=str(e))

        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message="Video editing completed successfully",
            data={"stream_url": stream_url},
        )
```

### ./vendor/Director/backend/director/agents/index.py
```python
import logging

from director.agents.base import BaseAgent, AgentResponse, AgentStatus

from director.core.session import Session
from director.tools.videodb_tool import VideoDBTool

logger = logging.getLogger(__name__)

EXTRACTION_CONFIGS_DEFAULTS = {
    "shot": {
        "threshold": 20,
        "min_scene_len": 15,
        "frame_count": 4,
    },
    "time": {
        "time": 10,
        "select_frames": ["first", "middle", "last"],
    },
}

SCENE_INDEX_CONFIG_DEFAULTS = {
    "type": "shot",
    "shot_based_config": EXTRACTION_CONFIGS_DEFAULTS["shot"],
    "time_based_config": EXTRACTION_CONFIGS_DEFAULTS["time"],
}

INDEX_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "video_id": {
            "type": "string",
            "description": "The ID of the video to process.",
        },
        "index_type": {
            "type": "string",
            "enum": ["spoken_words", "scene"],
            "default": "spoken_words",
        },
        "scene_index_prompt": {
            "type": "string",
            "description": "The prompt to use for scene indexing. Optional parameter only for scene based indexing ",
        },
        "scene_index_config": {
            "type": "object",
            "description": "Configuration for scene indexing behavior, Don't ask user to provide this parameter, provide if user explicitly mentions itt ",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["shot", "time"],
                    "default": "shot",
                    "description": "Method to use for scene detection and frame extraction",
                },
                "model_name": {
                    "type": "string",
                    "description": "The name of the model to use for scene detection and frame extraction",
                    "default": "gpt4-o",
                    "enum": ["gemini-1.5-flash", "gemini-1.5-pro", "gpt4-o"],
                },
                "shot_based_config": {
                    "type": "object",
                    "description": "Configuration for shot-based scene detection and frame extraction, This is a required parameter for shot_based indexing",
                    "properties": {
                        "threshold": {
                            "type": "number",
                            "default": SCENE_INDEX_CONFIG_DEFAULTS["shot_based_config"][
                                "threshold"
                            ],
                            "description": "Threshold value for scene change detection",
                        },
                        "min_scene_len": {
                            "type": "number",
                            "default": SCENE_INDEX_CONFIG_DEFAULTS["shot_based_config"][
                                "min_scene_len"
                            ],
                            "description": "Minimum length of a scene in frames",
                        },
                        "frame_count": {
                            "type": "number",
                            "default": SCENE_INDEX_CONFIG_DEFAULTS["shot_based_config"][
                                "frame_count"
                            ],
                            "description": "Number of frames to extract per scene",
                        },
                    },
                    "required": ["threshold", "min_scene_len", "frame_count"],
                },
                "time_based_config": {
                    "type": "object",
                    "description": "Configuration for time-based scene detection and frame extraction, This is a required parameter for time_based indexing",
                    "properties": {
                        "time": {
                            "type": "number",
                            "default": SCENE_INDEX_CONFIG_DEFAULTS["time_based_config"][
                                "time"
                            ],
                            "description": "Time interval in seconds between frame extractions",
                        },
                        "select_frames": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": SCENE_INDEX_CONFIG_DEFAULTS["time_based_config"][
                                "select_frames"
                            ],
                            "description": "Which frames to select from each time interval, In this array first, middle and last are the only allowed values",
                        },
                    },
                    "required": ["time", "select_frames"],
                },
            },
        },
        "collection_id": {
            "type": "string",
            "description": "The ID of the collection to process.",
        },
    },
    "required": ["video_id", "index_type", "collection_id"],
}

SCENE_INDEX_PROMPT = """
    AI Assistant Task: Video Image Analysis for Classification and Indexing

    For the attached image, please provide a comprehensive analysis covering the following categories:

    Scene Description:
        People: Describe the number of individuals, their appearance, actions, and expressions.
        Objects: Identify notable items and their relevance to the scene.
        Actions: Outline any activities or movements occurring.
        Environment: Specify details of the setting (e.g., indoor/outdoor, location type).
        Visual Elements: Note colors, lighting, and any stylistic features.

    Video Category Identification:
    Select the most likely category for the video from these options:
        Surveillance Video
        Movie Trailer
        Podcast
        Product Review
        Educational Lecture/Tutorial
        Music Video
        News Broadcast
        Sports Event
        Documentary
        Animated Content
        Advertisement/Commercial
        Gaming Video
        Vlog (Video Blog)
        Interview
        Live Stream
        Tutorial/How-To Video
        Corporate Presentation
        Promotional Event
        Cinematic Short Film
        User-Generated Content
        Other (please specify)
        Justification for Classification:
        Provide reasons for the chosen category based on visual cues observed.

    Text Extraction:
    List any text present, including:

    Titles
        Headlines
        Subtitles
        Labels
        Overlaid graphics
        Time and date stamps
        Explain the relevance of extracted text to the scene or video category.
    Notable Personalities and Characters:
        Identify recognizable individuals, characters, or public figures. Include names and relevance if possible.

    Logo and Brand Recognition:
        Describe visible logos, brand names, or trademarks and their relevance to the video content.

    Object and Symbol Detection:
        List key objects, symbols, or visual motifs. Explain their significance in context.

    Emotion and Mood Assessment:
        Assess the emotional tone (e.g., happy, tense, dramatic, instructional) and mention visual elements that contribute to this assessment.

    Genre and Style Identification (if applicable):
        For content like movies, music videos, or short films, identify the genre (e.g., action, comedy, horror) and stylistic elements that indicate genre or style.

    Additional Category-Specific Details:
    Provide relevant details based on the identified category, such as:
        Surveillance Video: Presence of timestamps, camera angles, or suspicious activities.
        Movie Trailer/Cinematic Short Film: Actor names, setting descriptions, special effects, and plot hints.
        Podcast/Interview: Speaker names, topics discussed, and setting details.
        Product Review/Advertisement: Product name, brand, highlighted features, and promotional messaging.
        Educational Lecture/Tutorial: Subject matter, instructional materials, and slide content.
        Music Video: Artist name, performance details, choreography, and symbolic imagery.
        News Broadcast: Anchors, headlines, network logos, and tickers.
        Sports Event: Sport type, teams or players, scores, and stadium details.
        Gaming Video: Game title, genre, in-game interfaces, and player reactions.
        Vlog/User-Generated Content: Content creator identity, activities, and setting.

    Technical Aspects (if noticeable):
    Comment on image quality (e.g., high-definition, low-resolution) and any visual effects, filters, or overlays.

    Cultural and Contextual Indicators:
    Identify any cultural references, language used, or regional indicators.

    Privacy and Ethical Considerations:
    Note if the image contains sensitive content, personal data, or requires discretion in handling.

    Instructions:
        Be Objective: Provide unbiased observations based solely on the visual content.
        Be Detailed: Include as much relevant information as possible to aid in accurate classification and indexing.
        Be Concise: Ensure clarity while providing thorough information.
        Handle Uncertainty: If certain elements are unclear, mention them and offer possible interpretations.
        Output Format: Provide a detailed description in a Paragraph do not use any formatting for easy reference and analysis.
    """.strip()


class IndexAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "index"
        self.description = "This is an agent to index the given video of VideoDB. The indexing can be done for spoken words or scene."
        self.parameters = INDEX_AGENT_PARAMETERS
        super().__init__(session=session, **kwargs)

    def run(
        self,
        video_id: str,
        index_type: str,
        scene_index_prompt=SCENE_INDEX_PROMPT,
        scene_index_config=SCENE_INDEX_CONFIG_DEFAULTS,
        collection_id=None,
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        Process the sample based on the given sample ID.
        :param str video_id: The ID of the video to process.
        :param str index_type: The type of indexing to perform.
        :param str collection_id: The ID of the collection to process.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: The response containing information about the sample processing operation.
        :rtype: AgentResponse
        """
        try:
            scene_data = {}
            if collection_id is None:
                self.videodb_tool = VideoDBTool()
            else:
                self.videodb_tool = VideoDBTool(collection_id=collection_id)
            self.output_message.actions.append(f"Indexing {index_type} for video..")
            self.output_message.push_update()

            if index_type == "spoken_words":
                self.videodb_tool.index_spoken_words(video_id)

            elif index_type == "scene":
                scene_index_type = scene_index_config["type"]
                scene_index_model_name = scene_index_config.get(
                    "model_name", "gpt4-o"
                )
                scene_index_config = scene_index_config[
                    scene_index_type + "_based_config"
                ]
                scene_index_id = self.videodb_tool.index_scene(
                    video_id=video_id,
                    extraction_type=scene_index_type,
                    extraction_config=scene_index_config,
                    prompt=scene_index_prompt,
                    model_name=scene_index_model_name,
                )
                self.videodb_tool.get_scene_index(
                    video_id=video_id, scene_id=scene_index_id
                )
                scene_data = {"scene_index_id": scene_index_id}

        except Exception as e:
            logger.exception(f"error in {self.agent_name} agent: {e}")
            return AgentResponse(status=AgentStatus.ERROR, message=str(e))

        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message=f"{index_type} indexing successful",
            data=scene_data,
        )
```

### ./vendor/Director/backend/director/agents/upload.py
```python
import logging

import yt_dlp

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import (
    Session,
    MsgStatus,
    VideoContent,
    TextContent,
    VideoData,
)
from director.tools.videodb_tool import VideoDBTool

logger = logging.getLogger(__name__)

UPLOAD_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "source": {
            "type": "string",
            "description": "URL or local path to upload the content",
        },
        "source_type": {
            "type": "string",
            "description": "Type of given source.",
            "enum": ["url", "local_file"],
        },
        "name": {
            "type": "string",
            "description": "Name of the content to upload, optional parameter",
        },
        "media_type": {
            "type": "string",
            "enum": ["video", "audio", "image"],
            "description": "Type of media to upload, default is video",
        },
        "collection_id": {
            "type": "string",
            "description": "Collection ID to upload the content",
        },
    },
    "required": ["url", "media_type", "collection_id"],
}


class UploadAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "upload"
        self.description = (
            "This agent uploads the media content to VideoDB. "
            "This agent takes a source which can be a URL or local path of the media content. "
            "The media content can be a video, audio, or image file. "
            "Youtube playlist and links are also supported. "
        )
        self.parameters = UPLOAD_AGENT_PARAMETERS
        super().__init__(session=session, **kwargs)

    def _upload(self, source: str, source_type: str, media_type: str, name: str = None):
        """Upload the media with the given URL."""
        try:
            if media_type == "video":
                content = VideoContent(
                    agent_name=self.agent_name, status=MsgStatus.progress
                )
            else:
                content = TextContent(
                    agent_name=self.agent_name, status=MsgStatus.progress
                )
            self.output_message.content.append(content)
            content.status_message = f"Uploading {media_type}..."
            self.output_message.push_update()

            upload_data = self.videodb_tool.upload(
                source, source_type, media_type, name=name
            )

            content.status_message = f"{upload_data['name']} uploaded successfully"
            if media_type == "video":
                content.video = VideoData(**upload_data)
            else:
                content.text = (
                    f"\n ID: {upload_data['id']}, Title: {upload_data['name']}"
                )
            content.status = MsgStatus.success
            self.output_message.publish()
            return AgentResponse(
                status=AgentStatus.SUCCESS,
                message="Upload successful",
                data=upload_data,
            )

        except Exception as e:
            logger.exception(f"error in {self.agent_name} agent: {e}")
            content.status = MsgStatus.error
            content.status_message = f"Error in uploading {media_type}"
            self.output_message.publish()
            return AgentResponse(status=AgentStatus.ERROR, message=str(e))

    def _get_yt_playlist_videos(self, playlist_url: str):
        """Get the list of videos from a youtube playlist."""
        try:
            # Create the downloader object
            with yt_dlp.YoutubeDL({"extract_flat": True, "quiet": True}) as ydl:
                playlist_info = ydl.extract_info(playlist_url, download=False)
            if "entries" in playlist_info:
                video_list = []
                for video in playlist_info["entries"]:
                    video_list.append(
                        {
                            "title": video.get("title", "Unknown Title"),
                            "url": f"https://www.youtube.com/watch?v={video['id']}",
                        }
                    )
                return video_list
        except Exception as e:
            logger.exception(f"Error in getting playlist info: {e}")
            return None

    def _upload_yt_playlist(self, playlist_info: dict, media_type):
        """Upload the videos in a youtube playlist."""
        for media in playlist_info:
            try:
                self.output_message.actions.append(
                    f"Uploading video: {media['title']} as {media_type}"
                )
                self._upload(media["url"], "url", media_type)
            except Exception as e:
                self.output_message.actions.append(
                    f"Upload failed for {media['title']}"
                )
                logger.exception(f"Error in uploading {media['title']}: {e}")
        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message="All the videos in the playlist uploaded successfully as {media_type}",
        )

    def run(
        self,
        collection_id: str,
        source: str,
        source_type: str,
        media_type="video",
        name: str = None,
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        Upload the media with the given source.

        :param collection_id: str - collection_id in which the upload is required.
        :param source: str - The URL or local path of the media to upload.
        :param source_type: str - The type indicating the source of the media.
        :param media_type: str, optional - The type of media to upload, defaults to "video".
        :param name: str, optional - Name required for uploaded file.
        :return: AgentResponse - The response containing information about the upload operation.
        """

        self.videodb_tool = VideoDBTool(collection_id=collection_id)

        if source_type == "local_file":
            return self._upload(source, source_type, media_type, name)
        elif source_type == "url":
            playlist_info = self._get_yt_playlist_videos(source)
            if playlist_info:
                self.output_message.actions.append("YouTube Playlist detected")
                self.output_message.push_update()
                return self._upload_yt_playlist(playlist_info, media_type)
            return self._upload(source, source_type, media_type, name)
        else:
            error_message = f"Invalid source type {source_type}"
            logger.error(error_message)
            return AgentResponse(
                status=AgentStatus.ERROR, message=error_message, data={}
            )
```

### ./vendor/Director/backend/director/agents/meme_maker.py
```python
import logging
import json
import concurrent.futures

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import (
    Session,
    ContextMessage,
    RoleTypes,
    MsgStatus,
    VideoContent,
    VideoData,
)
from director.tools.videodb_tool import VideoDBTool
from director.llm import get_default_llm

logger = logging.getLogger(__name__)

MEMEMAKER_PARAMETERS = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "Prompt to generate Meme, if user doesn't provide prompt don't ask user for prompt, use default prompt Create all the funniest Memes.",
        },
        "video_id": {
            "type": "string",
            "description": "Video Id to generate clip",
        },
        "collection_id": {
            "type": "string",
            "description": "Collection Id to of the video",
        },
    },
    "required": ["prompt", "video_id", "collection_id"],
}


class MemeMakerAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "meme_maker"
        self.description = "Generates meme clips and images based on user prompts. This agent usages LLM to analyze the transcript and visual content of the video to generate memes."
        self.parameters = MEMEMAKER_PARAMETERS
        self.llm = get_default_llm()
        super().__init__(session=session, **kwargs)

    def _chunk_docs(self, docs, chunk_size):
        """
        chunk docs to fit into context of your LLM
        :param docs:
        :param chunk_size:
        :return:
        """
        for i in range(0, len(docs), chunk_size):
            yield docs[i : i + chunk_size]  # Yield the current chunk

    def _filter_transcript(self, transcript, start, end):
        result = []
        for entry in transcript:
            if float(entry["end"]) > start and float(entry["start"]) < end:
                result.append(entry)
        return result

    def _get_multimodal_docs(self, transcript, scenes, club_on="scene"):
        # TODO: Implement club on transcript
        docs = []
        if club_on == "scene":
            for scene in scenes:
                spoken_result = self._filter_transcript(
                    transcript, float(scene["start"]), float(scene["end"])
                )
                spoken_text = " ".join(
                    entry["text"] for entry in spoken_result if entry["text"] != "-"
                )
                data = {
                    "visual": scene["description"],
                    "spoken": spoken_text,
                    "start": scene["start"],
                    "end": scene["end"],
                }
                docs.append(data)
        return docs

    def _prompt_runner(self, prompts):
        """Run the prompts in parallel."""
        clip_timestamps = []
        image_timestamps = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(
                    self.llm.chat_completions,
                    [ContextMessage(content=prompt, role=RoleTypes.user).to_llm_msg()],
                    response_format={"type": "json_object"},
                ): i
                for i, prompt in enumerate(prompts)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                try:
                    llm_response = future.result()
                    if not llm_response.status:
                        logger.error(f"LLM failed with {llm_response.content}")
                        continue
                    output = json.loads(llm_response.content)
                    clip_timestamps.extend(output.get("clip_timestamps", []))
                    image_timestamps.extend(output.get("image_timestamps", []))
                except Exception as e:
                    logger.exception(f"Error in getting matches: {e}")
                    continue
        return {
            "clip_timestamps": clip_timestamps,
            "image_timestamps": image_timestamps,
        }

    def _multimodal_prompter(self, transcript, scene_index, prompt):
        docs = self._get_multimodal_docs(transcript, scene_index)
        chunk_size = 80
        chunks = self._chunk_docs(docs, chunk_size=chunk_size)

        prompts = []
        i = 0
        for chunk in chunks:
            chunk_prompt = f"""
            Input Format:
                You are given visual and spoken information of the video of each second, and a transcipt of what's being spoken along with timestamp.
            
            Task: Analyze video content to identify peak Meme potential clip and images by evaluating:
                1. Memeable Moments
                    - Reaction-worthy facial expressions
                    - Quotable one-liners or catchphrases
                    - Unexpected or comedic timing
                    - Relatable human moments
                    - Visual gags or physical humor

                2. Meme Format Compatibility
                    - Reaction meme potential
                    - Image macro possibilities
                    - Multi-panel story potential
                    - GIF-worthy sequences
                    - Exploitable templates

                3. Virality Indicators
                    - Universal humor/relatability
                    - Clear emotional response triggers
                    - Easy to remix/recontextualize
                    - Cultural reference potential
                    - Distinct visual hooks

            Multimodal Data:
            video: {chunk}
            User Prompt: {prompt}

        
            """
            chunk_prompt += """
            **Output Format**: Return a JSON that containes the  fileds `clip_timestamps` and `image_timestamps`.
            clip_timestamps is from the visual section of the input.
            image_timestamps is the timestamp of the image in the visual section.
            Ensure the final output strictly adheres to the JSON format specified without including additional text, explanations or any extra characters.
            If there is no match return empty list without additional text. Use the following structure for your response:
            {"clip_timestamps": [{"start": start timestamp of the clip, "end": end timestamp of the clip, "text":  "text content of the clip"}], "image_timestamps": [timestamp of the image]}
            """
            prompts.append(chunk_prompt)
            i += 1

        return self._prompt_runner(prompts)

    def _get_scenes(self, video_id):
        self.output_message.actions.append("Retrieving video scenes..")
        self.output_message.push_update()
        scene_index_id = None
        scene_list = self.videodb_tool.list_scene_index(video_id)
        if scene_list:
            scene_index_id = scene_list[0]["scene_index_id"]
            return scene_index_id, self.videodb_tool.get_scene_index(
                video_id=video_id, scene_id=scene_index_id
            )
        else:
            self.output_message.actions.append("Scene index not found")
            self.output_message.push_update()
            raise Exception("Scene index not found, please index the scene first.")

    def _get_transcript(self, video_id):
        self.output_message.actions.append("Retrieving video transcript..")
        self.output_message.push_update()
        try:
            return self.videodb_tool.get_transcript(
                video_id
            ), self.videodb_tool.get_transcript(video_id, text=False)
        except Exception:
            self.output_message.actions.append(
                "Transcript unavailable. Indexing spoken content."
            )
            self.output_message.push_update()
            self.videodb_tool.index_spoken_words(video_id)
            return self.videodb_tool.get_transcript(
                video_id
            ), self.videodb_tool.get_transcript(video_id, text=False)

    def run(
        self, prompt: str, video_id: str, collection_id: str, *args, **kwargs
    ) -> AgentResponse:
        """
        Run the agent to generate meme clips and images based on user prompts.

        :return: The response containing information about the sample processing operation.
        :rtype: AgentResponse
        """
        try:
            self.videodb_tool = VideoDBTool(collection_id=collection_id)
            try:
                _, transcript = self._get_transcript(video_id=video_id)
                scene_index_id, scenes = self._get_scenes(video_id=video_id)

                self.output_message.actions.append("Identifying meme content..")
                self.output_message.push_update()
                result = self._multimodal_prompter(transcript, scenes, prompt)

            except Exception as e:
                logger.exception(f"Error in getting video content: {e}")
                return AgentResponse(status=AgentStatus.ERROR, message=str(e))

            if not result:
                return AgentResponse(
                    status=AgentStatus.SUCCESS,
                    message="No meme content found.",
                    data={},
                )
            self.output_message.actions.append("Key moments identified..")
            self.output_message.actions.append("Creating video clips..")
            self.output_message.push_update()
            data = {"clip_timestamps": [], "image_timestamps": []}
            all_clips_generated: bool = True
            for clip in result["clip_timestamps"]:
                video_content = VideoContent(
                    agent_name=self.agent_name,
                    status=MsgStatus.progress,
                    status_message="Generating clip..",
                )
                self.output_message.content.append(video_content)
                try:
                    stream_url = stream_url = self.videodb_tool.generate_video_stream(
                        video_id=video_id,
                        timeline=[(clip["start"], clip["end"])],
                    )
                except Exception as e:
                    video_content.status_message = f"Error generating stream: {str(e)}"
                    video_content.status = MsgStatus.error
                    self.output_message.push_update()
                    clip["stream_url"] = None
                    clip["error"] = f"Error generating stream: {str(e)}"
                    data["clip_timestamps"].append(clip)
                    all_clips_generated = False
                    continue

                video_content.video = VideoData(stream_url=stream_url)
                video_content.status_message = f'Clip "{clip["text"]}" generated.'
                video_content.status = MsgStatus.success
                clip["stream_url"] = stream_url
                data["clip_timestamps"].append(clip)
                self.output_message.publish()

        except Exception as e:
            logger.exception(f"Error in {self.agent_name}")
            return AgentResponse(status=AgentStatus.ERROR, message=str(e))

        return AgentResponse(
            status=AgentStatus.SUCCESS if all_clips_generated else AgentStatus.ERROR,
            message=f"Agent {self.name} completed successfully.",
            data=data,
        )
```

### ./vendor/Director/backend/director/agents/download.py
```python
import logging

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import Session, TextContent, MsgStatus
from director.tools.videodb_tool import VideoDBTool

logger = logging.getLogger(__name__)


class DownloadAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "download"
        self.description = "Get the download URLs of the VideoDB generated streams."
        self.parameters = self.get_parameters()
        super().__init__(session=session, **kwargs)

    def run(self, stream_link: str, name: str = None, *args, **kwargs) -> AgentResponse:
        """
        Downloads the video from the given stream link.

        :param stream_link: The URL of the video stream to download.
        :type stream_link: str
        :param stream_name: Optional name for the video stream. If not provided, defaults to None.
        :type stream_name: str, optional
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: The response containing information about the download operation.
        :rtype: AgentResponse
        """
        try:
            text_content = TextContent(agent_name=self.agent_name)
            text_content.status_message = "Downloading.."
            self.output_message.content.append(text_content)
            self.output_message.push_update()
            videodb_tool = VideoDBTool()
            download_response = videodb_tool.download(stream_link, name)
            if download_response.get("status") == "done":
                download_url = download_response.get("download_url")
                name = download_response.get("name")
                text_content.text = (
                    f"<a href='{download_url}' target='_blank'>{name}</a>"
                )
                text_content.status = MsgStatus.success
                text_content.status_message = "Here is the download link"
                self.output_message.publish()
            else:
                text_content.status = MsgStatus.error
                text_content.status_message = "Download failed"
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message=f"Downloda failed with {download_response}",
                )
        except Exception as e:
            text_content.status = MsgStatus.error
            text_content.status_message = "Download failed"
            logger.exception(f"error in {self.agent_name} agent: {e}")
            return AgentResponse(status=AgentStatus.ERROR, message=str(e))
        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message="Download successful.",
            data=download_response,
        )
```

### ./vendor/Director/backend/director/agents/thumbnail.py
```python
import logging

from director.agents.base import BaseAgent, AgentResponse, AgentStatus

from director.core.session import Session, MsgStatus, ImageContent, ImageData
from director.tools.videodb_tool import VideoDBTool

logger = logging.getLogger(__name__)

THUMBNAIL_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "collection_id": {
            "type": "string",
            "description": "Collection Id to of the video",
        },
        "video_id": {
            "type": "string",
            "description": "Video Id to generate thumbnail",
        },
        "timestamp": {
            "type": "integer",
            "description": "Timestamp in seconds of the video to generate thumbnail, Optional parameter don't ask from user",
        },
    },
    "required": ["collection_id", "video_id"],
}


class ThumbnailAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "thumbnail"
        self.description = "Generates a thumbnail image from a video file. This Agent takes a video id and a optionl timestamp as input. Use this tool when a user requests a preview, snapshot, generate or visual representation of a specific moment in a video file. The output is a static image file suitable for quick previews or thumbnails. It will not provide any other processing or editing options beyond generating the thumbnail."
        self.parameters = THUMBNAIL_AGENT_PARAMETERS
        super().__init__(session=session, **kwargs)

    def run(
        self, collection_id: str, video_id: str, timestamp: int = 5, *args, **kwargs
    ) -> AgentResponse:
        """
        Get the thumbnail for the video at the given timestamp
        """
        try:
            self.output_message.actions.append("Generating thumbnail..")
            image_content = ImageContent(agent_name=self.agent_name)
            image_content.status_message = "Generating thumbnail.."
            self.output_message.content.append(image_content)
            self.output_message.push_update()

            videodb_tool = VideoDBTool(collection_id=collection_id)
            thumbnail_data = videodb_tool.generate_thumbnail(
                video_id=video_id, timestamp=timestamp
            )
            image_content.image = ImageData(**thumbnail_data)
            image_content.status = MsgStatus.success
            image_content.status_message = "Here is your thumbnail."
            self.output_message.publish()

        except Exception as e:
            logger.exception(f"Error in {self.agent_name} agent.")
            image_content.status = MsgStatus.error
            image_content.status_message = "Error in generating thumbnail."
            self.output_message.publish()
            return AgentResponse(status=AgentStatus.ERROR, message=str(e))

        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message="Thumbnail generated and displayed to user.",
            data=thumbnail_data,
        )
```

### ./vendor/Director/backend/director/agents/__init__.py
```python
```

### ./vendor/Director/backend/director/agents/web_search_agent.py
```python
import logging
import os
import requests
from dotenv import load_dotenv

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import (
    Session,
    VideosContent,
    VideoData,
    MsgStatus,
)
from director.tools.serp import SerpAPI
from urllib.parse import urlparse, parse_qs

load_dotenv()
logger = logging.getLogger(__name__)

SEARCH_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "engine": {
            "type": "string",
            "description": "Engine to use for the search. Currently supports 'serp'.",
            "enum": ["serp"],
            "default": "serp",
        },
        "job_type": {
            "type": "string",
            "enum": ["search_videos"],
            "description": "The type of search to perform. Possible value: search_videos.",
        },
        "search_videos": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for the output.",
                    "minLength": 1,
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results to retrieve.",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 50,
                },
                "duration": {
                    "type": "string",
                    "description": "Filter videos by duration (short, medium, long).",
                    "enum": ["short", "medium", "long"],
                },
                "serp_config": {
                    "type": "object",
                    "description": "Config to use when SerpAPI engine is used",
                    "properties": {
                        "base_url": {
                            "type": "string",
                            "description": "Base URL for the SerpAPI service"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds for API requests",
                            "default": 10
                        }
                    }
                },
            },
            "required": ["query"],
        },
    },
    "required": ["job_type", "engine"],
}

SUPPORTED_ENGINES = ["serp"]

class WebSearchAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "web_search"
        self.description = (
            "Performs web searches to find and retrieve relevant videos using various engines."
        )
        self.parameters = SEARCH_AGENT_PARAMETERS
        super().__init__(session=session, **kwargs)

    def run(
        self,
        engine: str,
        job_type: str,
        search_videos: dict | None = None,
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        Perform a search using the specified engine and handle different job types.

        :param engine: Search engine to use (e.g., 'serp').
        :param job_type: Type of job to execute (e.g., 'search_videos').
        :param search_videos: Parameters specific to the 'search_videos' job type.
        :return: A structured response containing search results.
        """
        if engine not in SUPPORTED_ENGINES:
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Engine '{engine}' is not supported.",
            )

        self.api_key = os.getenv("SERP_API_KEY")
        if not self.api_key:
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="SERP_API_KEY environment variable is not set.",
            )

        serp_config = search_videos.get("serp_config", {})
        search_engine_tool = SerpAPI(
            api_key=self.api_key,
            base_url=serp_config.get("base_url"),
            timeout=serp_config.get("timeout", 10)
        )

        if job_type == "search_videos":
            if not isinstance(search_videos, dict):
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="'search_videos' must be a dictionary.",
                )
            if not search_videos:
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Parameters for 'search_videos' are required.",
                )
            return self._handle_video_search(search_videos, search_engine_tool)
        else:
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Unsupported job type: {job_type}.",
            )

    def _handle_video_search(self, search_videos: dict, search_engine_tool) -> AgentResponse:
        """Handles video searches."""
        query = search_videos.get("query")
        count = search_videos.get("count", 5)
        duration = search_videos.get("duration")
        if not isinstance(count, int) or count < 1 or count > 50:
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Count must be between 1 and 50",
            )
        if duration and duration not in ["short", "medium", "long"]:
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Invalid duration value: {duration}",
            )

        if not query or not isinstance(query, str) or not query.strip():
            return AgentResponse(
                status=AgentStatus.ERROR,
                message="Search query is required and cannot be empty.",
            )

        try:
            results = search_engine_tool.search_videos(query=query, count=count, duration=duration)
            valid_videos = []

            for video in results:
                external_url = video.get("link") or video.get("video_link")

                # Skip non-video YouTube links
                parsed_url = urlparse(external_url)
                if parsed_url.netloc in ["youtube.com", "www.youtube.com"]:
                    if any(parsed_url.path.startswith(prefix) for prefix in ["/channel/", "/@", "/c/", "/playlist"]):
                        continue
                    if not parsed_url.path.startswith("/watch") or not parse_qs(parsed_url.query).get("v"):
                        continue

                # Prepare video data
                video_data = VideoData(
                    external_url=external_url,
                    name=video.get("title", "Untitled Video"),
                    thumbnail_url=video.get("thumbnail"),
                )
                valid_videos.append(video_data)

            if not valid_videos:
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="No valid videos were found.",
                )

            videos_content = VideosContent(
                agent_name=self.agent_name,
                status=MsgStatus.success,
                status_message=f"Found {len(valid_videos)} videos.",
                videos=valid_videos,
            )
            self.output_message.content.append(videos_content)
            self.output_message.push_update()

            return AgentResponse(
                status=AgentStatus.SUCCESS,
                message="Video search completed successfully.",
                data={"videos": [video.dict() for video in valid_videos]},
            )
        except requests.exceptions.RequestException as e:
            error_message = "An error occurred during the video search."
            if isinstance(e, requests.exceptions.Timeout):
                error_message = "The search request timed out. Please try again."
            elif isinstance(e, requests.exceptions.HTTPError):
                if getattr(e.response, 'status_code', None) == 429:
                    error_message = "Rate limit exceeded. Please try again in a few minutes."
                elif getattr(e.response, 'status_code', None) == 401:
                    error_message = "API authentication failed. Please check your API key."
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=error_message,
            )
```

### ./vendor/Director/backend/director/agents/subtitle.py
```python
import logging
import textwrap
import json

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import (
    Session,
    ContextMessage,
    RoleTypes,
    VideoContent,
    VideoData,
    MsgStatus,
)
from director.tools.videodb_tool import VideoDBTool
from director.llm import get_default_llm


from videodb.asset import VideoAsset, TextAsset, TextStyle

logger = logging.getLogger(__name__)

SUBTITLE_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "video_id": {
            "type": "string",
            "description": "The unique identifier of the video to which subtitles will be added.",
        },
        "collection_id": {
            "type": "string",
            "description": "The unique identifier of the collection containing the video.",
        },
        "language": {
            "type": "string",
            "description": "The language the user wants the subtitles in",
        },
        "notes": {
            "type": "string",
            "description": "if user has additional requirements for the style of language",
        },
    },
    "required": ["video_id", "collection_id"],
}


translater_prompt = """

Task Description:
---
You are provided with a transcript of a video in a compact format called compact_list to optimize context size. The transcript is presented as a single string where each word block is formatted as:

word|start|end

word: The word itself.
start: The start time of the word in the video.
end: The end time of the word in the video.

Example Input (compact_list):

[ 'hello|0|10',  'world|11|12',  'how are you|13|15']

Your Task:
---
1.Translate the Text into [TARGET LANGUAGE]:
Translate all the words in the transcript from the source language to [TARGET LANGUAGE].

2.Combine Words into Meaningful Phrases or Sentences:
Group the translated words into logical phrases or sentences that make sense together.
Ensure each group is suitable for subtitle usage—neither too long nor too short.
Adjust Timing for Each Phrase/Block.

3.For each grouped phrase or sentence:
Start Time: Use the start time of the first word in the group.
End Time: Use the end time of the last word in the group.


4.Produce the Final Output:
Provide a list of subtitle blocks in the following format:
[    {"start": 0, "end": 30, "text": "Translated block of text here"},    {"start": 31, "end": 55, "text": "Another translated block of text"},    ...]
Ensure the translated text is coherent and appropriately grouped for subtitles.

Guidelines:
---

Coherence: The translated phrases should be grammatically correct and natural in [TARGET LANGUAGE].
Subtitle Length: Each subtitle block should follow standard subtitle length guidelines (e.g., no more than two lines of text, appropriate reading speed).
Timing Accuracy: Maintain accurate synchronization with the video's audio by correctly calculating start and end times.
Don't add any quotes, or %, that makes escaping the characters difficult.


Example Output:
---
If translating to Spanish, your output might look like:

you should return json of this format
{
subtitles: [    {"start": 0, "end": 30, "text": "Bloque traducido de texto aquí"},    {"start": 31, "end": 55, "text": "Otro bloque de texto traducido"},    ...]
}

Notes:
---
Be mindful of linguistic differences that may affect how words are grouped in [TARGET LANGUAGE].
Ensure that cultural nuances and idiomatic expressions are appropriately translated.

"""


class SubtitleAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "subtitle"
        self.description = "An agent designed to add different languages subtitles to a specified video within VideoDB."
        self.llm = get_default_llm()
        self.parameters = SUBTITLE_AGENT_PARAMETERS
        super().__init__(session=session, **kwargs)

    def wrap_text(self, text, video_width, max_width_percent=0.60, avg_char_width=20):
        max_width_pixels = video_width * max_width_percent
        max_chars_per_line = int(max_width_pixels / avg_char_width)

        # Wrap the text based on the calculated max characters per line
        wrapped_text = "\n".join(textwrap.wrap(text, max_chars_per_line))
        wrapped_text = wrapped_text.replace("'", "")
        return wrapped_text

    def get_compact_transcript(self, transcript):
        compact_list = []
        for word_block in transcript:
            word = word_block["text"]
            if word == "-":
                continue
            start = word_block["start"]
            end = word_block["end"]
            compact_word = f"{word}|{start}|{end}"
            compact_list.append(compact_word)
        return compact_list

    def add_subtitles_using_timeline(self, subtitles):
        video_width = 1920
        timeline = self.videodb_tool.get_and_set_timeline()
        video_asset = VideoAsset(asset_id=self.video_id)
        timeline.add_inline(video_asset)
        for subtitle_chunk in subtitles:
            start = round(subtitle_chunk["start"], 2)
            end = round(subtitle_chunk["end"], 2)
            duration = end - start

            wrapped_text = self.wrap_text(subtitle_chunk["text"], video_width)
            style = TextStyle(
                fontsize=20,
                fontcolor="white",
                box=True,
                boxcolor="black@0.6",
                boxborderw="5",
                y="main_h-text_h-50",
            )
            text_asset = TextAsset(
                text=wrapped_text,
                duration=duration,
                style=style,
            )
            timeline.add_overlay(start=start, asset=text_asset)
        stream_url = timeline.generate_stream()
        return stream_url

    def run(
        self,
        video_id: str,
        collection_id: str,
        language: str = "english",
        notes: str = "",
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        Adds subtitles to the specified video using the provided style configuration.

        :param str video_id: The unique identifier of the video to process.
        :param str collection_id: The unique identifier of the collection containing the video.
        :param str language: A string specifying the language for the subtitles.
        :param str notes: A String specifying the style of the language used in subtitles
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: The response indicating the success or failure of the subtitle addition operation.
        :rtype: AgentResponse
        """
        try:
            self.video_id = video_id
            self.videodb_tool = VideoDBTool(collection_id=collection_id)

            self.output_message.actions.append(
                "Retrieving the subtitles in the video's original language"
            )
            video_content = VideoContent(
                agent_name=self.agent_name,
                status=MsgStatus.progress,
                status_message="Processing...",
            )
            self.output_message.content.append(video_content)
            self.output_message.push_update()

            transcript = self.videodb_tool.get_transcript(video_id, text=False)
            compact_transcript = self.get_compact_transcript(transcript=transcript)

            self.output_message.actions.append(
                f"Translating the subtitles to {language}"
            )
            self.output_message.push_update()
            translation_llm_prompt = f"{translater_prompt} Translate to {language}, additional notes : {notes} compact_list: {compact_transcript}"
            translation_llm_message = ContextMessage(
                content=translation_llm_prompt,
                role=RoleTypes.user,
            )
            llm_response = self.llm.chat_completions(
                [translation_llm_message.to_llm_msg()],
                response_format={"type": "json_object"},
            )
            translated_subtitles = json.loads(llm_response.content)

            if notes:
                self.output_message.actions.append(
                    f"Refining the language with additional notes: {notes}"
                )

            self.output_message.actions.append(
                "Overlaying the translated subtitles onto the video"
            )
            self.output_message.push_update()

            stream_url = self.add_subtitles_using_timeline(
                translated_subtitles["subtitles"]
            )
            video_content.video = VideoData(stream_url=stream_url)
            video_content.status = MsgStatus.success
            video_content.status_message = f"Subtitles in {language} have been successfully added to your video. Here is your stream."
            self.output_message.publish()

        except Exception as e:
            logger.exception(f"Error in {self.agent_name} agent: {e}")
            video_content.status = MsgStatus.error
            video_content.status_message = (
                "An error occurred while adding subtitles to the video."
            )
            self.output_message.publish()
            return AgentResponse(status=AgentStatus.ERROR, message=str(e))

        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message="Subtitles added successfully",
            data={stream_url: stream_url},
        )
```

### ./vendor/Director/backend/director/agents/profanity_remover.py
```python
import json
import logging
import os

from videodb.asset import VideoAsset, AudioAsset

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import (
    Session,
    MsgStatus,
    VideoContent,
    VideoData,
    ContextMessage,
    RoleTypes,
)
from director.llm import get_default_llm
from director.tools.videodb_tool import VideoDBTool

logger = logging.getLogger(__name__)

BEEP_AUDIO_ID = os.getenv("BEEP_AUDIO_ID")
PROFANITY_FINDER_PROMPT = """
Given the following transcript give the list of timestamps where profanity is there for censoring.
Expected output format is json like {"timestamps": [(start, end), (start, end)]} where start and end are integer in seconds
"""


class ProfanityRemoverAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "profanity_remover"
        self.description = (
            "This agent beeps the profanities in the given video and returns the updated video stream. "
            "If the user has not provided the optional parameter `beep_audio_id`, send it as `None` so defaults are picked from the environment."
        )
        self.parameters = self.get_parameters()
        self.llm = get_default_llm()
        super().__init__(session=session, **kwargs)

    def add_beep(self, videodb_tool, video_id, beep_audio_id, timestamps):
        timeline = videodb_tool.get_and_set_timeline()
        video_asset = VideoAsset(asset_id=video_id)
        timeline.add_inline(video_asset)
        for start, _ in timestamps:
            beep = AudioAsset(asset_id=beep_audio_id)
            timeline.add_overlay(start=start, asset=beep)
        stream_url = timeline.generate_stream()
        return stream_url

    def run(
        self,
        collection_id: str,
        video_id: str,
        beep_audio_id: str = None,
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        Process the video to remove the profanities by overlaying beep.

        :param str collection_id: collection id in which the source video is present.
        :param str video_id: video_id on which profanity remover needs to run.
        :param str beep_audio_id: audio id of beep asset in videodb, defaults to BEEP_AUDIO_ID
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: The response containing information about the sample processing operation.
        :rtype: AgentResponse
        """
        try:
            beep_audio_id = beep_audio_id or BEEP_AUDIO_ID
            if not beep_audio_id:
                error_message = (
                    "Please provide the beep_audio_id, or set up BEEP_AUDIO_ID in the .env file located in the backend directory."
                    "Notify user this verbatum in summary as well since it isn't displayed to them: If you don’t have a beep sound ready, you can use the audio from https://www.youtube.com/watch?v=GvXbEO5Kbgc YouTube link. Upload it as audio, obtain the audio ID, and set it as BEEP_AUDIO_ID in your .env file.'"
                    "Remember to add above line in summary."
                )
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message=error_message,
                )
            self.output_message.actions.append("Started process to remove profanity..")
            video_content = VideoContent(
                agent_name=self.agent_name, status=MsgStatus.progress
            )
            video_content.status_message = "Generating clean stream.."
            self.output_message.push_update()
            videodb_tool = VideoDBTool(collection_id=collection_id)
            try:
                transcript = videodb_tool.get_transcript(video_id, text=False)
            except Exception:
                logger.error("Failed to get transcript, indexing")
                self.output_message.actions.append("Indexing the video..")
                self.output_message.push_update()
                videodb_tool.index_spoken_words(video_id)
                transcript = videodb_tool.get_transcript(video_id, text=False)
            profanity_prompt = f"{PROFANITY_FINDER_PROMPT}\n\ntranscript: {transcript}"
            profanity_llm_message = ContextMessage(
                content=profanity_prompt,
                role=RoleTypes.user,
            )
            llm_response = self.llm.chat_completions(
                [profanity_llm_message.to_llm_msg()],
                response_format={"type": "json_object"},
            )
            profanity_timeline_response = json.loads(llm_response.content)
            profanity_timeline = profanity_timeline_response.get("timestamps")
            clean_stream = self.add_beep(
                videodb_tool, video_id, beep_audio_id, profanity_timeline
            )
            video_content.video = VideoData(stream_url=clean_stream)
            video_content.status = MsgStatus.success
            video_content.status_message = "Here is the clean stream"
            self.output_message.content.append(video_content)
            self.output_message.publish()
        except Exception as e:
            logger.exception(f"Error in {self.agent_name}")
            video_content.status = MsgStatus.error
            video_content.status_message = "Failed to generate clean stream"
            self.output_message.publish()
            error_message = f"Error in generating the clean stream due to {e}."
            return AgentResponse(status=AgentStatus.ERROR, message=error_message)
        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message=f"Agent {self.name} completed successfully.",
            data={"stream_url": clean_stream},
        )
```

### ./vendor/Director/backend/director/agents/text_to_movie.py
```python
import logging
import os
import json
import uuid
from typing import List, Optional, Dict
from dataclasses import dataclass

from videodb.asset import VideoAsset, AudioAsset
from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import (
    Session,
    ContextMessage,
    MsgStatus,
    VideoContent,
    RoleTypes,
    VideoData,
)
from director.llm import get_default_llm
from director.tools.kling import KlingAITool, PARAMS_CONFIG as KLING_PARAMS_CONFIG
from director.tools.stabilityai import (
    StabilityAITool,
    PARAMS_CONFIG as STABILITYAI_PARAMS_CONFIG,
)
from director.tools.elevenlabs import (
    ElevenLabsTool,
    PARAMS_CONFIG as ELEVENLABS_PARAMS_CONFIG,
)
from director.tools.videodb_tool import VideoDBTool
from director.constants import DOWNLOADS_PATH


logger = logging.getLogger(__name__)

SUPPORTED_ENGINES = ["stabilityai", "kling"]

TEXT_TO_MOVIE_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "collection_id": {
            "type": "string",
            "description": "Collection ID to store the video",
        },
        "engine": {
            "type": "string",
            "description": "The video generation engine to use",
            "enum": SUPPORTED_ENGINES,
            "default": "stabilityai",
        },
        "job_type": {
            "type": "string",
            "enum": ["text_to_movie"],
            "description": "The type of video generation to perform",
        },
        "text_to_movie": {
            "type": "object",
            "properties": {
                "storyline": {
                    "type": "string",
                    "description": "The storyline to generate the video",
                },
                "sound_effects_description": {
                    "type": "string",
                    "description": "Optional description for background music generation",
                    "default": None,
                },
                "video_stabilityai_config": {
                    "type": "object",
                    "description": "Optional configuration for StabilityAI engine",
                    "properties": STABILITYAI_PARAMS_CONFIG["text_to_video"],
                },
                "video_kling_config": {
                    "type": "object",
                    "description": "Optional configuration for Kling engine",
                    "properties": KLING_PARAMS_CONFIG["text_to_video"],
                },
                "audio_elevenlabs_config": {
                    "type": "object",
                    "description": "Optional configuration for ElevenLabs engine",
                    "properties": ELEVENLABS_PARAMS_CONFIG["sound_effect"],
                },
            },
            "required": ["storyline"],
        },
    },
    "required": ["job_type", "collection_id", "engine"],
}


@dataclass
class VideoGenResult:
    """Track results of video generation"""

    step_index: int
    video_path: Optional[str]
    success: bool
    error: Optional[str] = None
    video: Optional[dict] = None


@dataclass
class EngineConfig:
    name: str
    max_duration: int
    preferred_style: str
    prompt_format: str


@dataclass
class VisualStyle:
    camera_setup: str
    color_grading: str
    lighting_style: str
    movement_style: str
    film_mood: str
    director_reference: str
    character_constants: Dict
    setting_constants: Dict


class TextToMovieAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        """Initialize agent with basic parameters"""
        self.agent_name = "text_to_movie"
        self.description = (
            "Agent for generating movies from storylines using Gen AI models"
        )
        self.parameters = TEXT_TO_MOVIE_AGENT_PARAMETERS
        self.llm = get_default_llm()

        self.engine_configs = {
            "kling": EngineConfig(
                name="kling",
                max_duration=10,
                preferred_style="cinematic",
                prompt_format="detailed",
            ),
            "stabilityai": EngineConfig(
                name="stabilityai",
                max_duration=4,
                preferred_style="photorealistic",
                prompt_format="concise",
            ),
        }
        super().__init__(session=session, **kwargs)

    def run(
        self,
        collection_id: str,
        engine: str = "stabilityai",
        job_type: str = "text_to_movie",
        text_to_movie: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        Process the storyline to generate a movie.

        :param collection_id: The collection ID to store generated assets
        :param engine: Video generation engine to use
        :return: AgentResponse containing information about generated movie
        """
        try:
            self.videodb_tool = VideoDBTool(collection_id=collection_id)
            self.output_message.actions.append("Processing input...")
            video_content = VideoContent(
                agent_name=self.agent_name,
                status=MsgStatus.progress,
                status_message="Generating movie...",
            )
            self.output_message.content.append(video_content)
            self.output_message.push_update()

            if engine not in self.engine_configs:
                raise ValueError(f"Unsupported engine: {engine}")

            if engine == "stabilityai":
                STABILITY_API_KEY = os.getenv("STABILITYAI_API_KEY")
                if not STABILITY_API_KEY:
                    raise Exception("Stability AI API key not found")
                self.video_gen_tool = StabilityAITool(api_key=STABILITY_API_KEY)
                self.video_gen_config_key = "video_stabilityai_config"
            elif engine == "kling":
                KLING_API_ACCESS_KEY = os.getenv("KLING_AI_ACCESS_API_KEY")
                KLING_API_SECRET_KEY = os.getenv("KLING_AI_SECRET_API_KEY")
                if not KLING_API_ACCESS_KEY or not KLING_API_SECRET_KEY:
                    raise Exception("Kling AI API key not found")
                self.video_gen_tool = KlingAITool(
                    access_key=KLING_API_ACCESS_KEY, secret_key=KLING_API_SECRET_KEY
                )
                self.video_gen_config_key = "video_kling_config"
            else:
                raise Exception(f"{engine} not supported")

            # Initialize tools
            ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
            if not ELEVENLABS_API_KEY:
                raise Exception("ElevenLabs API key not found")
            self.audio_gen_tool = ElevenLabsTool(api_key=ELEVENLABS_API_KEY)
            self.audio_gen_config_key = "audio_elevenlabs_config"

            # Initialize steps
            if job_type == "text_to_movie":
                raw_storyline = text_to_movie.get("storyline", [])
                video_gen_config = text_to_movie.get(self.video_gen_config_key, {})
                audio_gen_config = text_to_movie.get(self.audio_gen_config_key, {})

                # Generate visual style
                visual_style = self.generate_visual_style(raw_storyline)
                print("These are visual styles", visual_style)

                # Generate scenes
                scenes = self.generate_scene_sequence(
                    raw_storyline, visual_style, engine
                )
                print("These are scenes", scenes)

                self.output_message.actions.append(
                    f"Generating {len(scenes)} videos..."
                )
                self.output_message.push_update()

                engine_config = self.engine_configs[engine]
                generated_videos_results = []

                # Generate videos sequentially
                for index, scene in enumerate(scenes):
                    self.output_message.actions.append(
                        f"Generating video for scene {index + 1}..."
                    )
                    self.output_message.push_update()

                    suggested_duration = min(
                        scene.get("suggested_duration", 5), engine_config.max_duration
                    )
                    # Generate engine-specific prompt
                    prompt = self.generate_engine_prompt(scene, visual_style, engine)

                    print(f"Generating video for scene {index + 1}...")
                    print("This is the prompt", prompt)

                    video_path = f"{DOWNLOADS_PATH}/{str(uuid.uuid4())}.mp4"
                    os.makedirs(DOWNLOADS_PATH, exist_ok=True)

                    self.video_gen_tool.text_to_video(
                        prompt=prompt,
                        save_at=video_path,
                        duration=suggested_duration,
                        config=video_gen_config,
                    )
                    generated_videos_results.append(
                        VideoGenResult(
                            step_index=index, video_path=video_path, success=True
                        )
                    )

                self.output_message.actions.append(
                    f"Uploading {len(generated_videos_results)} videos to VideoDB..."
                )
                self.output_message.push_update()

                # Process videos and track duration
                total_duration = 0
                for result in generated_videos_results:
                    if result.success:
                        self.output_message.actions.append(
                            f"Uploading video {result.step_index + 1}..."
                        )
                        self.output_message.push_update()
                        media = self.videodb_tool.upload(
                            result.video_path,
                            source_type="file_path",
                            media_type="video",
                        )
                        total_duration += float(media.get("length", 0))
                        scenes[result.step_index]["video"] = media

                        # Cleanup temporary files
                        if os.path.exists(result.video_path):
                            os.remove(result.video_path)
                    else:
                        raise Exception(
                            f"Failed to generate video {result.step_index}: {result.error}"
                        )

                # Generate audio prompt
                sound_effects_description = self.generate_audio_prompt(raw_storyline)

                self.output_message.actions.append("Generating background music...")
                self.output_message.push_update()

                # Generate and add sound effects
                os.makedirs(DOWNLOADS_PATH, exist_ok=True)
                sound_effects_path = f"{DOWNLOADS_PATH}/{str(uuid.uuid4())}.mp3"

                self.audio_gen_tool.generate_sound_effect(
                    prompt=sound_effects_description,
                    save_at=sound_effects_path,
                    duration=total_duration,
                    config=audio_gen_config,
                )

                self.output_message.actions.append(
                    "Uploading background music to VideoDB..."
                )
                self.output_message.push_update()

                sound_effects_media = self.videodb_tool.upload(
                    sound_effects_path, source_type="file_path", media_type="audio"
                )

                if os.path.exists(sound_effects_path):
                    os.remove(sound_effects_path)

                self.output_message.actions.append(
                    "Combining assets into final video..."
                )
                self.output_message.push_update()

                # Combine everything into final video
                final_video = self.combine_assets(scenes, sound_effects_media)

                video_content.video = VideoData(stream_url=final_video)
                video_content.status = MsgStatus.success
                video_content.status_message = "Movie generation complete"
                self.output_message.publish()

                return AgentResponse(
                    status=AgentStatus.SUCCESS,
                    message="Movie generated successfully",
                    data={"video_url": final_video},
                )

            else:
                raise ValueError(f"Unsupported job type: {job_type}")

        except Exception as e:
            logger.exception(f"Error in {self.agent_name} agent: {e}")
            video_content.status = MsgStatus.error
            video_content.status_message = "Error generating movie"
            self.output_message.publish()
            return AgentResponse(
                status=AgentStatus.ERROR, message=f"Agent failed with error: {str(e)}"
            )

    def generate_visual_style(self, storyline: str) -> VisualStyle:
        """Generate consistent visual style for entire film."""
        style_prompt = f"""
        As a cinematographer, define a consistent visual style for this short film:
        Storyline: {storyline}
        
        Return a JSON response with visual style parameters:
        {{
            "camera_setup": "Camera and lens combination",
            "color_grading": "Color grading style and palette",
            "lighting_style": "Core lighting approach",
            "movement_style": "Camera movement philosophy",
            "film_mood": "Overall atmospheric mood",
            "director_reference": "Key director's style to reference",
            "character_constants": {{
                "physical_description": "Consistent character details",
                "costume_details": "Consistent costume elements"
            }},
            "setting_constants": {{
                "time_period": "When this takes place",
                "environment": "Core setting elements that stay consistent"
            }}
        }}
        """

        style_message = ContextMessage(content=style_prompt, role=RoleTypes.user)
        llm_response = self.llm.chat_completions(
            [style_message.to_llm_msg()], response_format={"type": "json_object"}
        )
        return VisualStyle(**json.loads(llm_response.content))

    def generate_scene_sequence(
        self, storyline: str, style: VisualStyle, engine: str
    ) -> List[dict]:
        """Generate 3-5 scenes with visual and narrative consistency."""
        engine_config = self.engine_configs[engine]

        sequence_prompt = f"""
        Break this storyline into 3 distinct scenes maintaining visual consistency.
        Generate scene descriptions optimized for {engine} {engine_config.preferred_style} style.
        
        Visual Style:
        - Camera/Lens: {style.camera_setup}
        - Color Grade: {style.color_grading}
        - Lighting: {style.lighting_style}
        - Movement: {style.movement_style}
        - Mood: {style.film_mood}
        - Director Style: {style.director_reference}
        
        Character Constants:
        {json.dumps(style.character_constants, indent=2)}
        
        Setting Constants:
        {json.dumps(style.setting_constants, indent=2)}
        
        Maximum duration per scene: {engine_config.max_duration} seconds
        
        Storyline: {storyline}

        Return a JSON array of scenes with:
        {{
            "story_beat": "What happens in this scene",
            "scene_description": "Visual description optimized for {engine}",
            "suggested_duration": "Duration as integer in seconds (max {engine_config.max_duration})"
        }}
        Make sure suggested_duration is a number, not a string.
        """

        sequence_message = ContextMessage(content=sequence_prompt, role=RoleTypes.user)
        llm_response = self.llm.chat_completions(
            [sequence_message.to_llm_msg()], response_format={"type": "json_object"}
        )
        scenes_data = json.loads(llm_response.content)["scenes"]

        # Ensure durations are integers
        for scene in scenes_data:
            try:
                scene["suggested_duration"] = int(scene.get("suggested_duration", 5))
            except (ValueError, TypeError):
                scene["suggested_duration"] = 5

        return scenes_data

    def generate_engine_prompt(
        self, scene: dict, style: VisualStyle, engine: str
    ) -> str:
        """Generate engine-specific prompt"""
        if engine == "stabilityai":
            return f"""
            {style.director_reference} style.
            {scene['scene_description']}.
            {style.character_constants['physical_description']}.
            {style.lighting_style}, {style.color_grading}.
            Photorealistic, detailed, high quality, masterful composition.
            """
        else:  # Kling
            initial_prompt = f"""
            {style.director_reference} style shot. 
            Filmed on {style.camera_setup}.
            
            {scene['scene_description']}
            
            Character Details:
            {json.dumps(style.character_constants, indent=2)}
            
            Setting Elements:
            {json.dumps(style.setting_constants, indent=2)}
            
            {style.lighting_style} lighting.
            {style.color_grading} color palette.
            {style.movement_style} camera movement.
            
            Mood: {style.film_mood}
            """

            # Run through LLM to compress while maintaining structure
            compression_prompt = f"""
            Compress the following prompt to under 2450 characters while maintaining its structure and key information:

            {initial_prompt}
            """

            compression_message = ContextMessage(
                content=compression_prompt, role=RoleTypes.user
            )
            llm_response = self.llm.chat_completions(
                [compression_message.to_llm_msg()], response_format={"type": "text"}
            )
            return llm_response.content

    def generate_audio_prompt(self, storyline: str) -> str:
        """Generate minimal, music-focused prompt for ElevenLabs."""
        audio_prompt = f"""
        As a composer, create a simple musical description focusing ONLY on:
        - Main instrument/sound
        - One key mood change
        - Basic progression
        
        Keep it under 100 characters. No visual references or scene descriptions.
        Focus on the music.
        
        Story context: {storyline}
        """

        prompt_message = ContextMessage(content=audio_prompt, role=RoleTypes.user)
        llm_response = self.llm.chat_completions(
            [prompt_message.to_llm_msg()], response_format={"type": "text"}
        )
        return llm_response.content[:100]

    def combine_assets(self, scenes: List[dict], audio_media: Optional[dict]) -> str:
        timeline = self.videodb_tool.get_and_set_timeline()

        # Add videos sequentially
        for scene in scenes:
            video_asset = VideoAsset(asset_id=scene["video"]["id"])
            timeline.add_inline(video_asset)

        # Add background score if available
        if audio_media:
            audio_asset = AudioAsset(
                asset_id=audio_media["id"], start=0, disable_other_tracks=True
            )
            timeline.add_overlay(0, audio_asset)

        return timeline.generate_stream()
```

### ./vendor/Director/backend/director/agents/comparison.py
```python
import logging
import asyncio

from director.utils.asyncio import is_event_loop_running
from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import (
    Session,
    VideosContent,
    VideoData,
    MsgStatus,
    VideosContentUIConfig,
)
from director.agents.video_generation import VideoGenerationAgent
from director.agents.video_generation import VIDEO_GENERATION_AGENT_PARAMETERS

logger = logging.getLogger(__name__)

COMPARISON_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "job_type": {
            "type": "string",
            "enum": ["video_generation_comparison"],
            "description": "Creates videos using MULTIPLE video generation models/engines. This agent should be used in two scenarios: 1) When request contains model names connected by words like 'and', '&', 'with', ',', 'versus', 'vs' (e.g. 'using Stability and Kling'), 2) When request mentions comparing/testing multiple models even if mentioned later in the prompt (e.g. 'Generate X. Compare results from Y, Z'). If the request suggests using more than one model in any way, use this agent rather than calling video_generation agent multiple times.",
        },
        "video_generation_comparison": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Description of the video generation run, Mention the configuration picked for the run. Keep the engine name and parameter config in a separate line. Keep it short. Here's an example: 'Tokyo Sunset - Luma - Prompt: 'An aerial shot of a quiet sunset at Tokyo', Duration: 5s, Luma Dream Machine",
                    },
                    **VIDEO_GENERATION_AGENT_PARAMETERS["properties"],
                },
                "required": [
                    "description",
                    *VIDEO_GENERATION_AGENT_PARAMETERS["required"],
                ],
                "description": "Parameters to use for each video generation run, each object in this is the parameters that would be required for each @video_generation run",
            },
            "description": "List of parameters to use for each video generation run, each object in this is the parameters that would be required for each @video_generation run",
        },
    },
    "required": ["job_type", "video_generation_comparison"],
}


class ComparisonAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "comparison"
        self.description = """Primary agent for video generation from prompts. Handles all video creation requests including single and multi-model generation. If multiple models or variations are mentioned, automatically parallelizes the work. For single model requests, delegates to specialized video generation subsystem. Keywords: generate video, create video, make video, text to video. """

        self.parameters = COMPARISON_AGENT_PARAMETERS
        super().__init__(session=session, **kwargs)

    def _run_video_generation(self, index, params):
        """Helper method to run video generation with given params"""
        video_gen_agent = VideoGenerationAgent(session=self.session)
        res = video_gen_agent.run(**params, stealth_mode=True)
        return index, res

    def done_callback(self, index, result):
        if result.status == AgentStatus.SUCCESS:
            self.videos_content.videos[index] = result.data["video_content"].video
        elif result.status == AgentStatus.ERROR:
            self.videos_content.videos[index] = VideoData(
                name=f"[Error] {self.videos_content.videos[index].name}",
                stream_url="",
                id=None,
                collection_id=None,
                error=result.message,
            )
        self.output_message.push_update()

    def run_tasks(self, tasks):
        for index, task in enumerate(tasks):
            _, task_res = self._run_video_generation(index, task)
            self.done_callback(index, task_res)

    def run(
        self, job_type: str, video_generation_comparison: list, *args, **kwargs
    ) -> AgentResponse:
        """
        Compare outputs from multiple runs of video generation

        :param str job_type: Type of comparison to perform
        :param list video_generation_comparison: Parameters for video gen runs
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: Response containing comparison results
        :rtype: AgentResponse
        """
        try:
            if job_type == "video_generation_comparison":
                self.videos_content = VideosContent(
                    agent_name=self.agent_name,
                    ui_config=VideosContentUIConfig(columns=3),
                    status=MsgStatus.progress,
                    status_message="Generating videos (Usually takes 3-7 mins)",
                    videos=[],
                )
                for params in video_generation_comparison:
                    video_data = VideoData(
                        name=params["text_to_video"]["name"],
                        stream_url="",
                    )
                    self.videos_content.videos.append(video_data)

                self.output_message.content.append(self.videos_content)
                self.output_message.push_update()

                self.run_tasks(video_generation_comparison)

                self.videos_content.status = MsgStatus.success
                self.videos_content.status_message = "Here are your generated videos"
                agent_response_message = f"Video generation comparison complete with {len(self.videos_content.videos)} videos"
                for video in self.videos_content.videos:
                    if video.error:
                        agent_response_message += (
                            f"\n Failed Video : {video.name}: Reason {video.error}"
                        )
                return AgentResponse(
                    status=AgentStatus.SUCCESS,
                    message=agent_response_message,
                    data={"videos": self.videos_content},
                )
            else:
                raise Exception(f"Unsupported comparison type: {job_type}")

        except Exception as e:
            logger.exception(f"Error in {self.agent_name} agent: {e}")
            return AgentResponse(status=AgentStatus.ERROR, message=str(e))
```

### ./vendor/Director/backend/director/agents/dubbing.py
```python
import logging
import os

from director.constants import DOWNLOADS_PATH

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import Session, VideoContent, MsgStatus, VideoData
from director.tools.videodb_tool import VideoDBTool
from director.tools.elevenlabs import ElevenLabsTool

logger = logging.getLogger(__name__)

SUPPORTED_ENGINES = ["elevenlabs"]
DUBBING_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "video_id": {
            "type": "string",
            "description": "The unique identifier of the video that needs to be dubbed. This ID is used to retrieve the video from the VideoDB collection.",
        },
        "target_language": {
            "type": "string",
            "description": "The target language for dubbing (e.g. 'Spanish', 'French', 'German'). The video's audio will be translated and dubbed into this language.",
        },
        "target_language_code": {
            "type": "string",
            "description": "The target language code for dubbing (e.g. 'es' for Spanish, 'fr' for French, 'de' for German').",
        },
        "collection_id": {
            "type": "string",
            "description": "The unique identifier of the VideoDB collection containing the video. Required to locate and access the correct video library.",
        },
        "engine": {
            "type": "string",
            "description": "The dubbing engine to use. Default is 'elevenlabs'. Possible values include 'elevenlabs'.",
            "default": "elevenlabs",
        },
        "engine_params": {
            "type": "object",
            "description": "Optional parameters for the dubbing engine.",
        },
    },
    "required": [
        "video_id",
        "target_language",
        "target_language_code",
        "collection_id",
        "engine",
    ],
}


class DubbingAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "dubbing"
        self.description = (
            "This is an agent to dub the given video into a target language"
        )
        self.parameters = DUBBING_AGENT_PARAMETERS
        super().__init__(session=session, **kwargs)

    def run(
        self,
        video_id: str,
        target_language: str,
        target_language_code: str,
        collection_id: str,
        engine: str,
        engine_params: dict = {},
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        Process the video dubbing based on the given video ID.
        :param str video_id: The ID of the video to process.
        :param str target_language: The target language name for dubbing (e.g. Spanish).
        :param str target_language_code: The target language code for dubbing (e.g. es).
        :param str collection_id: The ID of the collection to process.
        :param str engine: The dubbing engine to use. Default is 'elevenlabs'.
        :param dict engine_params: Optional parameters for the dubbing engine.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: The response containing information about the dubbing operation.
        :rtype: AgentResponse
        """
        try:
            self.videodb_tool = VideoDBTool(collection_id=collection_id)

            # Get video audio file
            video = self.videodb_tool.get_video(video_id)
            if not video:
                raise Exception(f"Video {video_id} not found")

            if engine not in SUPPORTED_ENGINES:
                raise Exception(f"{engine} not supported")

            video_content = VideoContent(
                agent_name=self.agent_name,
                status=MsgStatus.progress,
                status_message="Processing...",
            )
            self.output_message.content.append(video_content)
            self.output_message.actions.append("Downloading video")
            self.output_message.push_update()

            download_response = self.videodb_tool.download(video["stream_url"])

            os.makedirs(DOWNLOADS_PATH, exist_ok=True)
            dubbed_file_path = f"{DOWNLOADS_PATH}/{video_id}_dubbed.mp4"

            if engine == "elevenlabs":
                ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
                if not ELEVENLABS_API_KEY:
                    raise Exception("Elevenlabs API key not present in .env")
                elevenlabs_tool = ElevenLabsTool(api_key=ELEVENLABS_API_KEY)
                job_id = elevenlabs_tool.create_dub_job(
                    source_url=download_response["download_url"],
                    target_language=target_language_code,
                )
                self.output_message.actions.append(
                    f"Dubbing job initiated with Job ID: {job_id}"
                )
                self.output_message.push_update()

                self.output_message.actions.append(
                    "Waiting for dubbing process to complete.."
                )
                self.output_message.push_update()
                elevenlabs_tool.wait_for_dub_job(job_id)

                self.output_message.actions.append("Downloading dubbed video")
                self.output_message.push_update()
                elevenlabs_tool.download_dub_file(
                    job_id,
                    target_language_code,
                    dubbed_file_path,
                )

                self.output_message.actions.append(
                    f"Uploading dubbed video to VideoDB as '[Dubbed in {target_language}] {video['name']}'"
                )
                self.output_message.push_update()

            dubbed_video = self.videodb_tool.upload(
                dubbed_file_path,
                source_type="file_path",
                media_type="video",
                name=f"[Dubbed in {target_language}] {video['name']}",
            )

            video_content.video = VideoData(stream_url=dubbed_video["stream_url"])
            video_content.status = MsgStatus.success
            video_content.status_message = f"Dubbed video in {target_language} has been successfully added to your video. Here is your stream."
            self.output_message.publish()

            return AgentResponse(
                status=AgentStatus.SUCCESS,
                message=f"Successfully dubbed video '{video['name']}' to {target_language}",
                data={
                    "stream_url": dubbed_video["stream_url"],
                    "video_id": dubbed_video["id"],
                },
            )

        except Exception as e:
            video_content.status = MsgStatus.error
            video_content.status_message = "An error occurred while dubbing the video."
            self.output_message.publish()
            logger.exception(f"Error in {self.agent_name} agent: {str(e)}")
            return AgentResponse(
                status=AgentStatus.ERROR, message=f"Failed to dub video: {str(e)}"
            )
```

### ./vendor/Director/backend/director/agents/audio_generation.py
```python
import logging
import os
import uuid
import base64
from typing import Optional

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import Session, TextContent, MsgStatus
from director.tools.videodb_tool import VideoDBTool
from director.tools.elevenlabs import (
    ElevenLabsTool,
    PARAMS_CONFIG as ELEVENLABS_PARAMS_CONFIG,
)

from director.constants import DOWNLOADS_PATH

logger = logging.getLogger(__name__)

SUPPORTED_ENGINES = ["elevenlabs"]

AUDIO_GENERATION_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "collection_id": {
            "type": "string",
            "description": "The unique identifier of the collection to store the audio",
        },
        "engine": {
            "type": "string",
            "description": "The engine to use",
            "default": "elevenlabs",
            "enum": ["elevenlabs"],
        },
        "job_type": {
            "type": "string",
            "enum": ["text_to_speech", "sound_effect"],
            "description": """
                The type of audio generation to perform
                Possible values:
                    - text_to_speech: converts text to speech
                    - sound_effect: creates a sound effect from a text prompt
            """,
        },
        "sound_effect": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The prompt to generate the sound effect",
                },
                "duration": {
                    "type": "number",
                    "description": "The duration of the sound effect in seconds",
                    "default": 2,
                },
                "elevenlabs_config": {
                    "type": "object",
                    "properties": ELEVENLABS_PARAMS_CONFIG["sound_effect"],
                    "description": "Config to use when elevenlabs engine is used",
                },
            },
            "required": ["prompt"],
        },
        "text_to_speech": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to convert to speech",
                },
                "elevenlabs_config": {
                    "type": "object",
                    "properties": ELEVENLABS_PARAMS_CONFIG["text_to_speech"],
                    "description": "Config to use when elevenlabs engine is used",
                },
            },
            "required": ["text"],
        },
    },
    "required": ["job_type", "collection_id", "engine"],
}


class AudioGenerationAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "audio_generation"
        self.description = "An agent designed to generate speech from text and sound effects from prompt"
        self.parameters = AUDIO_GENERATION_AGENT_PARAMETERS
        super().__init__(session=session, **kwargs)

    def run(
        self,
        collection_id: str,
        job_type: str,
        engine: str,
        sound_effect: Optional[dict] = None,
        text_to_speech: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        Generates audio using ElevenLabs API based on input text.
        :param str collection_id: The collection ID to store the generated audio
        :param job_type: The audio generation engine to use
        :param sound_effect: The sound effect to generate
        :param text_to_speech: The text to convert to speech
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: Response containing the generated audio URL
        :rtype: AgentResponse
        """
        try:
            self.videodb_tool = VideoDBTool(collection_id=collection_id)

            if engine not in SUPPORTED_ENGINES:
                raise Exception(f"{engine} not supported")

            if engine == "elevenlabs":
                ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
                if not ELEVENLABS_API_KEY:
                    raise Exception("Elevenlabs API key not present in .env")
                audio_gen_tool = ElevenLabsTool(api_key=ELEVENLABS_API_KEY)
                config_key = "elevenlabs_config"

            os.makedirs(DOWNLOADS_PATH, exist_ok=True)
            output_file_name = f"audio_{job_type}_{str(uuid.uuid4())}.mp3"
            output_path = f"{DOWNLOADS_PATH}/{output_file_name}"

            if job_type == "sound_effect":
                prompt = sound_effect.get("prompt")
                duration = sound_effect.get("duration", 5)
                config = sound_effect.get(config_key, {})
                if prompt is None:
                    raise Exception("Prompt is required for sound effect generation")
                self.output_message.actions.append(
                    f"Generating sound effect using <b>{engine}</b> for prompt <i>{prompt}</i>"
                )
                self.output_message.push_update()
                audio_gen_tool.generate_sound_effect(
                    prompt=prompt,
                    save_at=output_path,
                    duration=duration,
                    config=config,
                )
            elif job_type == "text_to_speech":
                text = text_to_speech.get("text")
                config = text_to_speech.get(config_key, {})
                self.output_message.actions.append(
                    f"Using <b> {engine} </b> to convert text <i>{text}</i> to speech"
                )
                self.output_message.push_update()
                audio_gen_tool.text_to_speech(
                    text=text,
                    save_at=output_path,
                    config=config,
                )

            self.output_message.actions.append(
                f"Generated audio saved at <i>{output_path}</i>"
            )
            self.output_message.push_update()

            # Upload to VideoDB
            media = self.videodb_tool.upload(
                output_path, source_type="file_path", media_type="audio"
            )
            self.output_message.actions.append(
                f"Uploaded generated audio to VideoDB with Audio ID {media['id']}"
            )
            with open(os.path.abspath(output_path), "rb") as file:
                data_url = f"data:audio/mpeg;base64,{base64.b64encode(file.read()).decode('utf-8')}"
                text_content = TextContent(
                    agent_name=self.agent_name,
                    status=MsgStatus.success,
                    status_message="Here is your generated audio",
                    text=f"""Click <a href='{data_url}' download='{output_file_name}' target='_blank'>here</a> to download the audio
                    """,
                )
            self.output_message.content.append(text_content)
            self.output_message.push_update()
            self.output_message.publish()

        except Exception as e:
            logger.exception(f"Error in {self.agent_name} agent: {e}")
            text_content = TextContent(
                agent_name=self.agent_name,
                status=MsgStatus.error,
                status_message="Failed to generate audio",
            )
            self.output_message.content.append(text_content)
            self.output_message.push_update()
            self.output_message.publish()
            return AgentResponse(status=AgentStatus.ERROR, message=str(e))

        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message=f"Audio generated successfully, Generated Media audio id : {media['id']}",
            data={"audio_id": media["id"]},
        )
```

### ./vendor/Director/backend/director/agents/composio.py
```python
import os
import logging
import json

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import (
    Session,
    ContextMessage,
    RoleTypes,
    TextContent,
    MsgStatus,
)

from director.tools.composio_tool import composio_tool
from director.llm import get_default_llm
from director.llm.base import LLMResponseStatus

logger = logging.getLogger(__name__)

COMPOSIO_PARAMETERS = {
    "type": "object",
    "properties": {
        "task": {
            "type": "string",
            "description": "The task to perform",
        },
    },
    "required": ["task"],
}


class ComposioAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "composio"
        self.description = f'The Composio agent is used to run tasks related to apps like {os.getenv("COMPOSIO_APPS")} '
        self.parameters = COMPOSIO_PARAMETERS
        self.llm = get_default_llm()
        super().__init__(session=session, **kwargs)

    def run(self, task: str, *args, **kwargs) -> AgentResponse:
        """
        Run the composio with the given task.

        :return: The response containing information about the sample processing operation.
        :rtype: AgentResponse
        """
        try:
            self.output_message.actions.append("Running task..")
            self.output_message.push_update()

            text_content = TextContent(
                agent_name=self.agent_name,
                status=MsgStatus.progress,
                status_message="Running task..",
            )
            self.output_message.content.append(text_content)
            self.output_message.push_update()

            composio_response = composio_tool(task=task)
            llm_prompt = (
                f"User has asked to run a task: {task} in Composio. \n"
                "Format the following reponse into text.\n"
                "Give the output which can be directly send to use \n"
                "Don't add any extra text \n"
                f"{json.dumps(composio_response)}"
            )
            composio_response = ContextMessage(content=llm_prompt, role=RoleTypes.user)
            llm_response = self.llm.chat_completions([composio_response.to_llm_msg()])
            if llm_response.status == LLMResponseStatus.ERROR:
                raise Exception(f"LLM Failed with error {llm_response.content}")

            text_content.text = llm_response.content
            text_content.status = MsgStatus.success
            text_content.status_message = "Here is the response from Composio"

        except Exception as e:
            logger.exception(f"Error in {self.agent_name}")
            text_content.status = MsgStatus.error
            self.output_message.publish()
            error_message = f"Agent failed with error {e}"
            return AgentResponse(status=AgentStatus.ERROR, message=error_message)

        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message=f"Agent {self.name} completed successfully.",
            data={"composio_response": composio_response},
        )
```

### ./vendor/Director/backend/director/agents/brandkit.py
```python
import os
import logging

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import Session, MsgStatus, VideoContent, VideoData
from director.tools.videodb_tool import VideoDBTool

logger = logging.getLogger(__name__)

INTRO_VIDEO_ID = os.getenv("INTRO_VIDEO_ID")
OUTRO_VIDEO_ID = os.getenv("OUTRO_VIDEO_ID")
BRAND_IMAGE_ID = os.getenv("BRAND_IMAGE_ID")


class BrandkitAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "brandkit"
        self.description = (
            "Agent to add brand kit elements (intro video, outro video and brand image) to the given video in VideoDB,"
            "if user has not given those optional param of intro video, outro video and brand image always try with sending them as None so that defaults are picked from env"
        )
        self.parameters = self.get_parameters()
        super().__init__(session=session, **kwargs)

    def run(
        self,
        collection_id: str,
        video_id: str,
        intro_video_id: str = None,
        outro_video_id: str = None,
        brand_image_id: str = None,
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        Generate stream of video after adding branding elements.

        :param str collection_id: collection id in which videos are available.
        :param str video_id: video id on which branding is required.
        :param str intro_video_id: VideoDB video id of intro video, defaults to INTRO_VIDEO_ID
        :param str outro_video_id: video id of outro video, defaults to OUTRO_VIDEO_ID
        :param str brand_image_id: image id of brand image for overlay over video, defaults to BRAND_IMAGE_ID
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: The response containing information about the generated brand stream.
        :rtype: AgentResponse
        """
        try:
            self.output_message.actions.append("Processing brandkit request..")
            intro_video_id = intro_video_id or INTRO_VIDEO_ID
            outro_video_id = outro_video_id or OUTRO_VIDEO_ID
            brand_image_id = brand_image_id or BRAND_IMAGE_ID
            if not any([intro_video_id, outro_video_id, brand_image_id]):
                message = (
                    "Branding elementes not provided, either you can provide provide IDs for intro video, outro video and branding image"
                    " or you can set INTRO_VIDEO_ID, OUTRO_VIDEO_ID and BRAND_IMAGE_ID in .env of backend directory."
                )
                return AgentResponse(status=AgentStatus.ERROR, message=message)
            video_content = VideoContent(
                agent_name=self.agent_name,
                status=MsgStatus.progress,
                status_message="Generating video with branding..",
            )
            self.output_message.content.append(video_content)
            self.output_message.push_update()
            videodb_tool = VideoDBTool(collection_id=collection_id)
            brandkit_stream = videodb_tool.add_brandkit(
                video_id, intro_video_id, outro_video_id, brand_image_id
            )
            video_content.video = VideoData(stream_url=brandkit_stream)
            video_content.status = MsgStatus.success
            video_content.status_message = "Here is your brandkit stream"
            self.output_message.publish()
        except Exception:
            logger.exception(f"Error in {self.agent_name}")
            video_content.status = MsgStatus.error
            error_message = "Error in adding branding."
            video_content.status_message = error_message
            self.output_message.publish()
            return AgentResponse(status=AgentStatus.ERROR, message=error_message)
        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message=f"Agent {self.name} completed successfully.",
            data={"stream_url": brandkit_stream},
        )
```

### ./vendor/Director/backend/director/agents/prompt_clip.py
```python
import logging
import json
import concurrent.futures

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import (
    Session,
    ContextMessage,
    RoleTypes,
    MsgStatus,
    VideoContent,
    VideoData,
)
from director.tools.videodb_tool import VideoDBTool
from director.llm import get_default_llm

logger = logging.getLogger(__name__)

PROMPTCLIP_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "prompt": {
            "type": "string",
            "description": "Prompt to generate clip",
        },
        "content_type": {
            "type": "string",
            "enum": ["spoken_content", "visual_content", "multimodal"],
            "description": "Type of content based on which clip is to be generated, default is spoken_content, spoken_content: based on transcript of the video, visual_content: based on visual description of the video, multimodal: based on both transcript and visual description of the video",
        },
        "video_id": {
            "type": "string",
            "description": "Video Id to generate clip",
        },
        "collection_id": {
            "type": "string",
            "description": "Collection Id to of the video",
        },
    },
    "required": ["prompt", "content_type", "video_id", "collection_id"],
}


class PromptClipAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "prompt_clip"
        # TODO: Improve this
        self.description = "Generates video clips based on user prompts. This agent uses AI to analyze the text of a video transcript and identify sentences relevant to the user prompt for making clips. It then generates video clips based on the identified sentences. Use this tool to create clips based on specific themes or topics from a video."
        self.parameters = PROMPTCLIP_AGENT_PARAMETERS
        self.llm = get_default_llm()
        super().__init__(session=session, **kwargs)

    def _chunk_docs(self, docs, chunk_size):
        """
        chunk docs to fit into context of your LLM
        :param docs:
        :param chunk_size:
        :return:
        """
        for i in range(0, len(docs), chunk_size):
            yield docs[i : i + chunk_size]  # Yield the current chunk

    def _filter_transcript(self, transcript, start, end):
        result = []
        for entry in transcript:
            if float(entry["end"]) > start and float(entry["start"]) < end:
                result.append(entry)
        return result

    def _get_multimodal_docs(self, transcript, scenes, club_on="scene"):
        # TODO: Implement club on transcript
        docs = []
        if club_on == "scene":
            for scene in scenes:
                spoken_result = self._filter_transcript(
                    transcript, float(scene["start"]), float(scene["end"])
                )
                spoken_text = " ".join(
                    entry["text"] for entry in spoken_result if entry["text"] != "-"
                )
                data = {
                    "visual": scene["description"],
                    "spoken": spoken_text,
                    "start": scene["start"],
                    "end": scene["end"],
                }
                docs.append(data)
        return docs

    def _prompt_runner(self, prompts):
        """Run the prompts in parallel."""
        matches = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(
                    self.llm.chat_completions,
                    [ContextMessage(content=prompt, role=RoleTypes.user).to_llm_msg()],
                    response_format={"type": "json_object"},
                ): i
                for i, prompt in enumerate(prompts)
            }
            for future in concurrent.futures.as_completed(future_to_index):
                try:
                    llm_response = future.result()
                    if not llm_response.status:
                        logger.error(f"LLM failed with {llm_response.content}")
                        continue
                    output = json.loads(llm_response.content)
                    matches.extend(output["sentences"])
                except Exception as e:
                    logger.exception(f"Error in getting matches: {e}")
                    continue
        return matches

    def _text_prompter(self, transcript_text, prompt):
        chunk_size = 10000
        # sentence tokenizer
        chunks = self._chunk_docs(transcript_text, chunk_size=chunk_size)
        prompts = []
        i = 0
        for chunk in chunks:
            chunk_prompt = """
            You are a video editor who uses AI. Given a user prompt and transcript of a video analyze the text to identify sentences in the transcript relevant to the user prompt for making clips. 
            - **Instructions**: 
            - Evaluate the sentences for relevance to the specified user prompt.
            - Make sure that sentences start and end properly and meaningfully complete the discussion or topic. Choose the one with the greatest relevance and longest.
            - We'll use the sentences to make video clips in future, so optimize for great viewing experience for people watching the clip of these.
            - Strictly make each result minimum 20 words long. If the match is smaller, adjust the boundries and add more context around the sentences.

            - **Output Format**: Return a JSON list of strings named 'sentences' that containes the output sentences, make sure they are exact substrings.
            - **User Prompts**: User prompts may include requests like 'find funny moments' or 'find moments for social media'. Interpret these prompts by 
            identifying keywords or themes in the transcript that match the intent of the prompt.
            """
            # pass the data
            chunk_prompt += f"""
            Transcript: {chunk}
            User Prompt: {prompt}
            """
            # Add instructions to always return JSON at the end of processing.
            chunk_prompt += """
            Ensure the final output strictly adheres to the JSON format specified without including additional text or explanations. \
            If there is no match return empty list without additional text. Use the following structure for your response:
            {
            "sentences": [
                {},
                ...
            ]
            }
            """
            prompts.append(chunk_prompt)
            i += 1

        return self._prompt_runner(prompts)

    def _scene_prompter(self, scene_index, prompt):
        chunk_size = 10000
        chunks = self._chunk_docs(scene_index, chunk_size=chunk_size)

        prompts = []
        i = 0
        for chunk in chunks:
            descriptions = [scene["description"] for scene in chunk]
            chunk_prompt = """
            You are a video editor who uses AI. Given a user prompt and AI-generated scene descriptions of a video, analyze the descriptions to identify segments relevant to the user prompt for creating clips.

            - **Instructions**: 
                - Evaluate the scene descriptions for relevance to the specified user prompt.
                - Choose description with the highest relevance and most comprehensive content.
                - Optimize for engaging viewing experiences, considering visual appeal and narrative coherence.

                - User Prompts: Interpret prompts like 'find exciting moments' or 'identify key plot points' by matching keywords or themes in the scene descriptions to the intent of the prompt.
            """

            chunk_prompt += f"""
            Descriptions: {json.dumps(descriptions)}
            User Prompt: {prompt}
            """

            chunk_prompt += """
            **Output Format**: Return a JSON list of strings named 'result' that containes the  fileds `sentence` Ensure the final output
            strictly adheres to the JSON format specified without including additional text or explanations. \
            If there is no match return empty list without additional text. Use the following structure for your response:
            {"sentences": []}
            """
            prompts.append(chunk_prompt)
            i += 1

        return self._prompt_runner(prompts)

    def _multimodal_prompter(self, transcript, scene_index, prompt):
        docs = self._get_multimodal_docs(transcript, scene_index)
        chunk_size = 80
        chunks = self._chunk_docs(docs, chunk_size=chunk_size)

        prompts = []
        i = 0
        for chunk in chunks:
            chunk_prompt = f"""
            You are given visual and spoken information of the video of each second, and a transcipt of what's being spoken along with timestamp.
            Your task is to evaluate the data for relevance to the specified user prompt.
            Corelate visual and spoken content to find the relevant video segment.

            Multimodal Data:
            video: {chunk}
            User Prompt: {prompt}

        
            """
            chunk_prompt += """
            **Output Format**: Return a JSON list of strings named 'result' that containes the  fileds `sentence`.
            sentence is from the visual section of the input.
            Ensure the final output strictly adheres to the JSON format specified without including additional text or explanations.
            If there is no match return empty list without additional text. Use the following structure for your response:
            {"sentences": []}
            """
            prompts.append(chunk_prompt)
            i += 1

        return self._prompt_runner(prompts)

    def _get_scenes(self, video_id):
        self.output_message.actions.append("Retrieving video scenes..")
        self.output_message.push_update()
        scene_index_id = None
        scene_list = self.videodb_tool.list_scene_index(video_id)
        if scene_list:
            scene_index_id = scene_list[0]["scene_index_id"]
            return scene_index_id, self.videodb_tool.get_scene_index(
                video_id=video_id, scene_id=scene_index_id
            )
        else:
            self.output_message.actions.append("Scene index not found")
            self.output_message.push_update()
            raise Exception("Scene index not found, please index the scene first.")

    def _get_transcript(self, video_id):
        self.output_message.actions.append("Retrieving video transcript..")
        self.output_message.push_update()
        try:
            return self.videodb_tool.get_transcript(
                video_id
            ), self.videodb_tool.get_transcript(video_id, text=False)
        except Exception:
            self.output_message.actions.append(
                "Transcript unavailable. Indexing spoken content."
            )
            self.output_message.push_update()
            self.videodb_tool.index_spoken_words(video_id)
            return self.videodb_tool.get_transcript(
                video_id
            ), self.videodb_tool.get_transcript(video_id, text=False)

    def run(
        self,
        prompt: str,
        content_type: str,
        video_id: str,
        collection_id: str,
        *args,
        **kwargs,
    ) -> AgentResponse:
        try:
            self.videodb_tool = VideoDBTool(collection_id=collection_id)
            result = []
            try:
                if content_type == "spoken_content":
                    transcript_text, _ = self._get_transcript(video_id=video_id)
                    result = self._text_prompter(transcript_text, prompt)

                elif content_type == "visual_content":
                    scene_index_id, scenes = self._get_scenes(video_id=video_id)
                    result = self._scene_prompter(scenes, prompt)

                else:
                    _, transcript = self._get_transcript(video_id=video_id)
                    scene_index_id, scenes = self._get_scenes(video_id=video_id)
                    result = self._multimodal_prompter(transcript, scenes, prompt)

            except Exception as e:
                logger.exception(f"Error in getting video content: {e}")
                return AgentResponse(status=AgentStatus.ERROR, message=str(e))

            self.output_message.actions.append("Identifying key moments..")
            self.output_message.push_update()
            result_timestamps = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                if content_type == "spoken_content":
                    future_to_index = {
                        executor.submit(
                            self.videodb_tool.keyword_search,
                            query=description,
                            video_id=video_id,
                        ): description
                        for description in result
                    }
                else:
                    future_to_index = {
                        executor.submit(
                            self.videodb_tool.keyword_search,
                            query=description,
                            index_type="scene",
                            video_id=video_id,
                            scene_index_id=scene_index_id,
                        ): description
                        for description in result
                    }

                for future in concurrent.futures.as_completed(future_to_index):
                    description = future_to_index[future]
                    try:
                        search_res = future.result()
                        matched_segments = search_res.get_shots()
                        video_shot = matched_segments[0]
                        result_timestamps.append(
                            (video_shot.start, video_shot.end, video_shot.text)
                        )
                    except Exception as e:
                        logger.error(
                            f"Error in getting timestamps of {description}: {e}"
                        )
                        continue
            if result_timestamps:
                try:
                    self.output_message.actions.append("Key moments identified..")
                    self.output_message.actions.append("Creating video clip..")
                    video_content = VideoContent(
                        agent_name=self.agent_name, status=MsgStatus.progress
                    )
                    self.output_message.content.append(video_content)
                    self.output_message.push_update()
                    timeline = []
                    for timestamp in result_timestamps:
                        timeline.append((timestamp[0], timestamp[1]))
                    stream_url = self.videodb_tool.generate_video_stream(
                        video_id=video_id, timeline=timeline
                    )
                    video_content.status_message = "Clip generated successfully."
                    video_content.video = VideoData(stream_url=stream_url)
                    video_content.status = MsgStatus.success
                    self.output_message.publish()

                except Exception as e:
                    logger.exception(f"Error in creating video content: {e}")
                    return AgentResponse(status=AgentStatus.ERROR, message=str(e))

                return AgentResponse(
                    status=AgentStatus.SUCCESS,
                    message=f"Agent {self.name} completed successfully.",
                    data={"stream_url": stream_url},
                )
            else:
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="No relevant moments found.",
                )

        except Exception as e:
            logger.exception(f"error in {self.agent_name}")
            return AgentResponse(status=AgentStatus.ERROR, message=str(e))
```

### ./vendor/Director/backend/director/agents/pricing.py
```python
import logging

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import (
    Session,
    MsgStatus,
    ContextMessage,
    RoleTypes,
    TextContent,
)
from director.llm import get_default_llm

logger = logging.getLogger(__name__)

PRICING_AGENT_PROMPT = """
    You are a brilliant pricing analyst working for VideoDB, a video database for AI apps. 
    You can access information from internet and also reference this sheet to provide answers to to the task your user (executive) asks. 
    VideoDB brings storage, index, retrieval and streaming at one place. 
    Programatic streams can be generated form any segment of the video and to find the right segment, indexing of content is necessary. 

    Here's the workflow:
    - Any file that gets uploaded remain in storage. 
    - Developer can generate the stream of any segment to watch using `generate_stream()`
    - Once the stream is generated, developer can use it in application to watch streams. 
    - Developer index the spoken content (one time) to search spoken content across videos. 
    - Developer index the visual content (one time) using scene indexing to search visual aspects of the content across videos.
    -  Monthly charge are for the file storage and index storage (after one time cost).
    - You can't add monthly indexing cost without one time index charges. ( for example user may upload 100 hours but choose to index only 10) 

    If user says 3000 hours of streaming that means you'll use only the streaming cost. You can add stream generation cost of uploaded content. Ask for how many new streams they might generate?

    Assume 700mb is equivalent to 1 hour. Give precise and concise answer to scenarios and use tabular format whenever useful.  When showing price always use metric as "$x/minute" or "$x/minute/month" or "$z/GB/month" type of language for clarity.
    Use following data to calculate revenue and profit for different scenarios provided by the user.  Pricing json: 
    pricing = {
        "data_storage": {
            "metric" : "GB (Size)",
            "info": "Cost to store your uploaded content",
            "price_monthly": 0.03,
        
        },
        "index_storage": {
            "metric" : "minute (Size)",
            "info": "Cost to store your indexes",
            "price_monthly": 0.0005
        },
        "spoken_index_creation": {
            "metric" : "minute (Indexed)",
        "info": "One time cost to index conversations",
            "price_one_time": 0.02
        },
    "scene_index_creation": {
        "metric" : "Scene (Indexed)",
        "info": "One time cost to index visuals in the video, perfect for security cam footage etc.", Default we can assume 1 sec as 1 scene, but you can give more calculation based on let's say if each scene is of 10 sec, 20 sec etc. You can also ask user to explore scene extraction algorithms that determines the scene and it depends on the video content.
            "price_one_time": 0.0125
        },
        "programmable_video_stream_generation": {
            "metric" : "minute ( Generated)",
            "info": "One time cost of generating any stream link, includes compilations, overlays, edits etc.",
            "price_one_time": 0.06
        },
        "simple_streaming": {
        "metric" : "minute (Streamed)",
        "info": "Depends on the number of views your streams receive",
            "price_one_time": 0.000998  
        },
        "search_queries": {
            "metric" : "count",
            "info": "# of video searches ",
            "price_one_time": 0.0025
        }
    }

    If user asks "Estimate my cost" Follow these instructions: 
    Gather Information step by step and provide a nice summary in a readable format. 
    - Assistant: "Let's start by understanding your specific needs. Could you please tell me [first aspect, e.g., 'how many hours of video content you plan to upload each month']?"
    <User provides input.>
    Assistant: "Great, now could you tell me [next aspect, e.g., 'how many of those hours you wish to index for spoken content']?"
    Repeat until all necessary information is gathered. Calculate the costs based on the provided inputs using a predefined pricing structure.
    Present the Costs in a Readable Form: For Initial and One-time Costs: Assistant: "Here's the breakdown of your initial and one-time costs:"
    Format: Use larger or bold font to emphasize the initial & one-time monthly cost.
    Format: Use bullet points or a table to list each cost with its corresponding value.
    For Recurring Monthly Costs:Prompt: "Here's the detailed breakdown of your monthly costs:"
    Format: Use a table to list each cost type with its rate and specific monthly cost.
    Emphasize Total Cost:
    Assistant: "And your Total Monthly Cost is:"
    Format: Use larger or bold font to emphasize the total monthly cost figure.
    Maintain Conciseness.
    Ensure that each part of the interaction is direct and to the point, avoiding unnecessary details or complex explanations unless requested by the user.
    Use tables where appropriate for presenting the cost breakdowns or comparing different costs, use tables for better clarity and quick readability.

    For comparative analysis use search action to get latest information.
"""


class PricingAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "pricing"
        self.description = "Agent to get information about the pricing and usage of VideoDB, helpful for running scenarios to get the estimates."
        self.parameters = self.get_parameters()
        self.llm = get_default_llm()
        super().__init__(session=session, **kwargs)

    def run(self, query: str, *args, **kwargs) -> AgentResponse:
        """
        Get the answer to the query of agent

        :param query: Query for the pricing agent for estimation scenarios.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: The response containing query output.
        :rtype: AgentResponse
        """
        try:
            text_content = TextContent(agent_name=self.agent_name)
            text_content.status_message = "Calculating pricing.."
            self.output_message.content.append(text_content)
            self.output_message.push_update()

            pricing_llm_prompt = f"{PRICING_AGENT_PROMPT} user query: {query}"
            pricing_llm_message = ContextMessage(
                content=pricing_llm_prompt, role=RoleTypes.user
            )
            llm_response = self.llm.chat_completions([pricing_llm_message.to_llm_msg()])

            if not llm_response.status:
                logger.error(f"LLM failed with {llm_response}")
                text_content.status = MsgStatus.failed
                text_content.status_message = "Failed to generate the response."
                self.output_message.publish()
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Pricing failed due to LLM error.",
                )
            text_content.text = llm_response.content
            text_content.status = MsgStatus.success
            text_content.status_message = "Pricing estimation is ready."
            self.output_message.publish()
        except Exception:
            logger.exception(f"Error in {self.agent_name}")
            error_message = "Error in calculating pricing."
            text_content.status = MsgStatus.error
            text_content.status_message = error_message
            self.output_message.publish()
            return AgentResponse(status=AgentStatus.ERROR, message=error_message)

        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message="Agent run successful",
            data={
                "response": llm_response.content
            },
        )
```

### ./vendor/Director/backend/director/agents/summarize_video.py
```python
import logging

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import ContextMessage, RoleTypes, TextContent, MsgStatus
from director.llm import get_default_llm
from director.tools.videodb_tool import VideoDBTool

logger = logging.getLogger(__name__)


class SummarizeVideoAgent(BaseAgent):
    def __init__(self, session=None, **kwargs):
        self.agent_name = "summarize_video"
        self.description = "This is an agent to summarize the given video of VideoDB, if the user wants a certain kind of summary the prompt is required."
        self.llm = get_default_llm()
        self.parameters = self.get_parameters()
        super().__init__(session=session, **kwargs)

    def run(self, collection_id: str, video_id: str, prompt: str) -> AgentResponse:
        """
        Generate summary of the given video.

        :param str collection_id: The collection_id where given video_id is available.
        :param str video_id: The id of the video for which the video player is required.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: The response containing information about the sample processing operation.
        :rtype: AgentResponse

        """
        try:
            self.output_message.actions.append("Started summary generation..")
            output_text_content = TextContent(
                agent_name=self.agent_name, status_message="Generating the summary.."
            )
            self.output_message.content.append(output_text_content)
            self.output_message.push_update()
            videodb_tool = VideoDBTool(collection_id=collection_id)
            try:
                transcript_text = videodb_tool.get_transcript(video_id)
            except Exception:
                logger.error("Failed to get transcript, indexing")
                self.output_message.actions.append("Indexing the video..")
                self.output_message.push_update()
                videodb_tool.index_spoken_words(video_id)
                transcript_text = videodb_tool.get_transcript(video_id)
            summary_llm_prompt = f"{transcript_text} {prompt}"
            summary_llm_message = ContextMessage(
                content=summary_llm_prompt, role=RoleTypes.user
            )
            llm_response = self.llm.chat_completions([summary_llm_message.to_llm_msg()])
            if not llm_response.status:
                logger.error(f"LLM failed with {llm_response}")
                output_text_content.status = MsgStatus.failed
                output_text_content.status_message = "Failed to generat the summary."
                self.output_message.publish()
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Summary failed due to LLM error.",
                )
            summary = llm_response.content
            output_text_content.text = summary
            output_text_content.status = MsgStatus.success
            output_text_content.status_message = "Here is your summary"
            self.output_message.publish()
        except Exception as e:
            logger.exception(f"Error in {self.agent_name} agent.")
            output_text_content.status = MsgStatus.error
            output_text_content.status_message = "Error in generating summary."
            self.output_message.publish()
            return AgentResponse(status=AgentStatus.ERROR, message=str(e))

        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message="Summary generated and displayed to user.",
            data={"summary": summary},
        )
```

### ./vendor/Director/backend/director/agents/sample.py
```python
import logging

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import Session, MsgStatus, TextContent

logger = logging.getLogger(__name__)


class SampleAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "sample"
        self.description = "Sample agent description"
        self.parameters = self.get_parameters()
        super().__init__(session=session, **kwargs)

    def run(self, sample_id: str, *args, **kwargs) -> AgentResponse:
        """
        Process the sample based on the given sample ID.

        :param str sample_id: The ID of the sample to process.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: The response containing information about the sample processing operation.
        :rtype: AgentResponse
        """
        try:
            self.output_message.actions.append("Processing sample..")
            text_content = TextContent(
                agent_name=self.agent_name,
                status=MsgStatus.progress,
                status_message="Generating sample response..",
            )
            self.output_message.content.append(text_content)
            self.output_message.push_update()
            text_content.text = "This is the text result of Agent."
            text_content.status = MsgStatus.success
            text_content.status_message = "Here is your response"
            self.output_message.publish()
        except Exception as e:
            logger.exception(f"Error in {self.agent_name}")
            text_content.status = MsgStatus.error
            text_content.status_message = "Error in calculating pricing."
            self.output_message.publish()
            error_message = f"Agent failed with error {e}"
            return AgentResponse(status=AgentStatus.ERROR, message=error_message)
        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message=f"Agent {self.name} completed successfully.",
            data={},
        )
```

### ./vendor/Director/backend/director/agents/search.py
```python
import logging

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.llm import get_default_llm
from director.core.session import (
    Session,
    MsgStatus,
    TextContent,
    SearchResultsContent,
    SearchData,
    ShotData,
    VideoContent,
    VideoData,
    ContextMessage,
    RoleTypes,
)
from director.tools.videodb_tool import VideoDBTool

logger = logging.getLogger(__name__)

SEARCH_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query"},
        "search_type": {
            "type": "string",
            "enum": ["semantic", "keyword"],
            "description": "Type of search, default is semantic. semantic: search based on semantic similarity, keyword: search based on keyword only avilable on video not collection",
        },
        "index_type": {
            "type": "string",
            "enum": ["spoken_word", "scene"],
            "description": "Type of indexing to perform, spoken_word: based on transcript of the video, scene: based on visual description of the video",
        },
        "result_threshold": {
            "type": "number",
            "description": " Initial filter for top N matching documents (default: 8).",
        },
        "score_threshold": {
            "type": "integer",
            "description": "Absolute threshold filter for relevance scores (default: 0.2).",
        },
        "dynamic_score_percentage": {
            "type": "integer",
            "description": "Adaptive filtering mechanism: Useful when there is a significant gap between top results and tail results after score_threshold filter. Retains top x% of the score range. Calculation: dynamic_threshold = max_score - (range * dynamic_score_percentage) (default: 20 percent)",
        },
        "video_id": {
            "type": "string",
            "description": "The ID of the video to process.",
        },
        "collection_id": {
            "type": "string",
            "description": "The ID of the collection to process.",
        },
    },
    "required": ["query", "search_type", "index_type", "collection_id"],
}


class SearchAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "search"
        self.description = "Agent to search information from VideoDB collections. Mainly used with a collection of videos."
        self.llm = get_default_llm()
        self.parameters = SEARCH_AGENT_PARAMETERS
        super().__init__(session=session, **kwargs)

    def run(
        self,
        query: str,
        search_type,
        index_type,
        collection_id: str,
        video_id: str = None,
        result_threshold=8,
        score_threshold=0.2,
        dynamic_score_percentage=20,
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        Retreive data from VideoDB collections and videos.

        :return: The response containing search results, text summary and compilation video.
        :rtype: AgentResponse
        """
        try:
            search_result_content = SearchResultsContent(
                status=MsgStatus.progress,
                status_message="Started getting search results.",
                agent_name=self.agent_name,
            )
            self.output_message.content.append(search_result_content)
            self.output_message.actions.append(f"Running {search_type} search on {index_type} index.")
            self.output_message.push_update()
            videodb_tool = VideoDBTool(collection_id=collection_id)
            scene_index_id = None
            if index_type == "scene" and video_id:
                scene_index_list = videodb_tool.list_scene_index(video_id)
                if scene_index_list:
                    scene_index_id = scene_index_list[0].get("scene_index_id")
                else:
                    self.output_message.actions.append("Scene index not found")
                    self.output_message.push_update()
                    raise ValueError("Scene index not found. Please index scene first.")

            if search_type == "semantic":
                search_results = videodb_tool.semantic_search(
                    query,
                    index_type=index_type,
                    video_id=video_id,
                    result_threshold=result_threshold,
                    score_threshold=score_threshold,
                    dynamic_score_percentage=dynamic_score_percentage,
                    scene_index_id=scene_index_id,
                )

            elif search_type == "keyword" and video_id:
                search_results = videodb_tool.keyword_search(
                    query,
                    index_type=index_type,
                    video_id=video_id,
                    result_threshold=result_threshold,
                    scene_index_id=scene_index_id,
                )
            else:
                raise ValueError(f"Invalid search type {search_type}")

            compilation_content = VideoContent(
                status=MsgStatus.progress,
                status_message="Started video compilation.",
                agent_name=self.agent_name,
            )
            self.output_message.content.append(compilation_content)
            search_summary_content = TextContent(
                status=MsgStatus.progress,
                status_message="Started generating summary of search results.",
                agent_name=self.agent_name,
            )
            self.output_message.content.append(search_summary_content)
            shots = search_results.get_shots()
            if not shots:
                search_result_content.status = MsgStatus.error
                search_result_content.status_message = "Failed to get search results."
                compilation_content.status = MsgStatus.error
                compilation_content.status_message = (
                    "Failed to create compilation of search results."
                )
                search_summary_content.status = MsgStatus.error
                search_summary_content.status_message = (
                    "Failed to generate summary of results."
                )
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message=f"Failed due to no search results found for query {query}",
                    data={
                        "message": f"Failed due to no search results found for query {query}",
                    },
                )
            search_result_videos = {}
            for shot in shots:
                video_id = shot["video_id"]
                video_title = shot["video_title"]
                if video_id in search_result_videos:
                    search_result_videos[video_id]["shots"].append(
                        {
                            "search_score": shot["search_score"],
                            "start": shot["start"],
                            "end": shot["end"],
                            "text": shot["text"],
                        }
                    )
                else:
                    video = videodb_tool.get_video(video_id)
                    search_result_videos[video_id] = {
                        "video_id": video_id,
                        "video_title": video_title,
                        "stream_url": video.get("stream_url"),
                        "duration": video.get("length"),
                        "shots": [
                            {
                                "search_score": shot["search_score"],
                                "start": shot["start"],
                                "end": shot["end"],
                                "text": shot["text"],
                            }
                        ],
                    }
            search_result_content.search_results = [
                SearchData(
                    video_id=sr["video_id"],
                    video_title=sr["video_title"],
                    stream_url=sr["stream_url"],
                    duration=sr["duration"],
                    shots=[ShotData(**shot) for shot in sr["shots"]],
                )
                for sr in search_result_videos.values()
            ]
            search_result_content.status = MsgStatus.success
            search_result_content.status_message = "Search done."
            self.output_message.actions.append(
                "Generating search result compilation clip.."
            )
            self.output_message.push_update()
            compilation_stream_url = search_results.compile()
            compilation_content.video = VideoData(stream_url=compilation_stream_url)
            compilation_content.status = MsgStatus.success
            compilation_content.status_message = "Compilation done."
            self.output_message.actions.append("Generating search result summary..")
            self.output_message.push_update()
            search_result_text_list = [shot.text for shot in shots]
            search_result_text = "\n\n".join(search_result_text_list)
            search_summary_llm_prompt = f"Summarize the search results for query: {query} search results: {search_result_text}"
            search_summary_llm_message = ContextMessage(
                content=search_summary_llm_prompt, role=RoleTypes.user
            )
            llm_response = self.llm.chat_completions(
                [search_summary_llm_message.to_llm_msg()]
            )
            search_summary_content.text = llm_response.content
            if not llm_response.status:
                search_summary_content.status = MsgStatus.error
                search_summary_content.status_message = (
                    "Failed to generate the summary of search results."
                )
                logger.error(f"LLM failed with {llm_response}")
            else:
                search_summary_content.text = llm_response.content
                search_summary_content.status = MsgStatus.success
                search_summary_content.status_message = (
                    "Here is the summary of search results."
                )
            self.output_message.publish()
        except Exception as e:
            logger.exception(f"Error in {self.agent_name}.")
            if search_result_content.status != MsgStatus.success:
                search_result_content.status = MsgStatus.error
                search_result_content.status_message = "Failed to get search results."
            elif compilation_content.status != MsgStatus.success:
                compilation_content.status = MsgStatus.error
                compilation_content.status_message = (
                    "Failed to create compilation of search results."
                )
            elif search_summary_content.status != MsgStatus.success:
                search_summary_content.status = MsgStatus.error
                search_summary_content.status_message = (
                    "Failed to generate summary of results."
                )
            return AgentResponse(
                status=AgentStatus.ERROR,
                message=f"Failed the search with exception. {e}",
            )
        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message="Search done and showed above to user.",
            data={"message": "Search done.", "stream_link": compilation_stream_url},
        )
```

### ./vendor/Director/backend/director/agents/stream_video.py
```python
import logging

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import Session, MsgStatus, VideoContent, VideoData
from director.tools.videodb_tool import VideoDBTool

logger = logging.getLogger(__name__)


class StreamVideoAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "stream_video"
        self.description = (
            "Agent to play the requested video or given m3u8 stream_url by getting the video player"
        )
        self.parameters = self.get_parameters()
        super().__init__(session=session, **kwargs)

    def run(
        self,
        collection_id: str = None,
        video_id: str = None,
        stream_url: str = None,
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        Process the collection_id, video_id or stream_url to send the video component.

        :param str collection_id: The collection_id where given video_id is available.
        :param str video_id: The id of the video for which the video player is required.
        :param str stream_url: stream_url for which video player is required.
        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: The response containing information about the sample processing operation.
        :rtype: AgentResponse
        """
        try:
            if video_id:
                self.output_message.actions.append("Processing for given video_id..")
            elif stream_url:
                self.output_message.actions.append("Processing given stream url..")
            else:
                return AgentResponse(
                    status=AgentStatus.ERROR,
                    message="Either 'video_id' or 'stream_url' is required for getting the stream in video player.",
                )
            if stream_url:
                video_content = VideoContent(
                    agent_name=self.agent_name,
                    status=MsgStatus.success,
                    status_message="Here is your stream",
                    video={"stream_url": stream_url},
                )
                self.output_message.content.append(video_content)
                self.output_message.publish()
                return AgentResponse(
                    status=AgentStatus.SUCCESS,
                    message=f"Agent {self.name} completed successfully.",
                    data={},
                )
            video_content = VideoContent(
                agent_name=self.agent_name,
                status=MsgStatus.progress,
                status_message="Loading stream for the video..",
            )
            self.output_message.content.append(video_content)
            self.output_message.push_update()
            videodb_tool = VideoDBTool(collection_id=collection_id)
            video_data = videodb_tool.get_video(video_id)
            stream_url = video_data.get("stream_url")
            video_content.video = VideoData(stream_url=stream_url)
            video_content.status = MsgStatus.success
            video_content.status_message = "Here is your stream"
            self.output_message.publish()
        except Exception as e:
            logger.exception(f"Error in {self.agent_name}")
            video_content.status = MsgStatus.error
            video_content.status_message = "Error in calculating pricing."
            self.output_message.publish()
            error_message = f"Agent failed with error {e}"
            return AgentResponse(status=AgentStatus.ERROR, message=error_message)
        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message=f"Agent {self.name} completed successfully.",
            data={
                "stream_url": stream_url
            },
        )
```

### ./vendor/Director/backend/director/agents/video_generation.py
```python
import logging
import os
import uuid

from typing import Optional

from director.agents.base import BaseAgent, AgentResponse, AgentStatus
from director.core.session import Session, VideoContent, VideoData, MsgStatus
from director.tools.videodb_tool import VideoDBTool
from director.tools.stabilityai import (
    StabilityAITool,
    PARAMS_CONFIG as STABILITYAI_PARAMS_CONFIG,
)
from director.tools.fal_video import (
    FalVideoGenerationTool,
    PARAMS_CONFIG as FAL_VIDEO_GEN_PARAMS_CONFIG,
)
from director.constants import DOWNLOADS_PATH

logger = logging.getLogger(__name__)

SUPPORTED_ENGINES = ["stabilityai", "fal"]

VIDEO_GENERATION_AGENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "collection_id": {
            "type": "string",
            "description": "Collection ID to store the video",
        },
        "engine": {
            "type": "string",
            "description": "The video generation engine to use. Use Fal by default. If the query includes any of the following: 'minimax-video, mochi-v1, hunyuan-video, luma-dream-machine, cogvideox-5b, ltx-video, fast-svd, fast-svd-lcm, t2v-turbo, kling video v 1.0, kling video v1.5 pro, fast-animatediff, fast-animatediff turbo, and animatediff-sparsectrl-lcm'- always use Fal. In case user specifies any other engine, use the supported engines like Stability.",
            "default": "fal",
            "enum": ["fal", "stabilityai"],
        },
        "job_type": {
            "type": "string",
            "enum": ["text_to_video", "image_to_video"],
            "description": """
            The type of video generation to perform
            Possible values:
                - text_to_video: generates a video from a text prompt
                - image_to_video: generates a video using an image as a base. The feature is only available in fal engine. Stability AI does not support this feature.
            """,
        },
        "text_to_video": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The text prompt to generate the video",
                },
                "name": {
                    "type": "string",
                    "description": "Description of the video generation run in two lines. Keep the engine name and parameter configuration of the engine in separate lines. Keep it short, but show the prompt in full. Here's an example: [Tokyo Sunset - Luma - Prompt: 'An aerial shot of a quiet sunset at Tokyo', Duration: 5s, Luma Dream Machine]",
                },
                "duration": {
                    "type": "number",
                    "description": "The duration of the video in seconds",
                    "default": 5,
                },
                "stabilityai_config": {
                    "type": "object",
                    "properties": STABILITYAI_PARAMS_CONFIG["text_to_video"],
                    "description": "Config to use when stabilityai engine is used",
                },
                "fal_config": {
                    "type": "object",
                    "properties": FAL_VIDEO_GEN_PARAMS_CONFIG["text_to_video"],
                    "description": "Config to use when fal engine is used",
                },
            },
            "required": ["prompt", "name"],
        },
        "image_to_video": {
            "type": "object",
            "properties": {
                "image_id": {
                    "type": "string",
                    "description": "The ID of the image in VideoDB to use for video generation",
                },
                "name": {
                    "type": "string",
                    "description": "Description of the video generation run",
                },
                "prompt": {
                    "type": "string",
                    "description": "Text prompt to guide the video generation",
                },
                "duration": {
                    "type": "number",
                    "description": "The duration of the video in seconds",
                    "default": 5,
                },
                "fal_config": {
                    "type": "object",
                    "properties": FAL_VIDEO_GEN_PARAMS_CONFIG["image_to_video"],
                    "description": "Config to use when fal engine is used",
                },
            },
            "required": ["prompt", "image_id", "name"],
        },
    },
    "required": ["job_type", "collection_id", "engine"],
}


class VideoGenerationAgent(BaseAgent):
    def __init__(self, session: Session, **kwargs):
        self.agent_name = "video_generation"
        self.description = "Creates videos using ONE specific model/engine. Only use this agent when the request mentions exactly ONE model/engine, without any comparison words like 'compare', 'test', 'versus', 'vs' and no connecting words (and/&/,) between model names. If the request mentions wanting to compare models or try multiple engines, do not use this agent - use the comparison agent instead."
        self.parameters = VIDEO_GENERATION_AGENT_PARAMETERS
        super().__init__(session=session, **kwargs)

    def run(
        self,
        collection_id: str,
        job_type: str,
        engine: str,
        text_to_video: Optional[dict] = None,
        image_to_video: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> AgentResponse:
        """
        Generates video using Stability AI's API based on input text prompt.
        :param collection_id: The collection ID to store the generated video
        :param job_type: The type of video generation job to perform
        :param engine: The engine to use for video generation
        :param text_to_video: The text to convert to video
        :param image_to_video: The image to convert to video
        :param args: Additional positional arguments
        :param kwargs: Additional keyword arguments
        :return: Response containing the generated video ID
        """
        try:
            self.videodb_tool = VideoDBTool(collection_id=collection_id)
            stealth_mode = kwargs.get("stealth_mode", False)

            if engine not in SUPPORTED_ENGINES:
                raise Exception(f"{engine} not supported")

            video_content = VideoContent(
                agent_name=self.agent_name,
                status=MsgStatus.progress,
                status_message="Processing...",
            )
            if not stealth_mode:
                self.output_message.content.append(video_content)

            if engine == "stabilityai":
                STABILITYAI_API_KEY = os.getenv("STABILITYAI_API_KEY")
                if not STABILITYAI_API_KEY:
                    raise Exception("Stability AI API key not found")
                video_gen_tool = StabilityAITool(api_key=STABILITYAI_API_KEY)
                config_key = "stabilityai_config"
            elif engine == "fal":
                FAL_KEY = os.getenv("FAL_KEY")
                if not FAL_KEY:
                    raise Exception("FAL API key not found")
                video_gen_tool = FalVideoGenerationTool(api_key=FAL_KEY)
                config_key = "fal_config"
            else:
                raise Exception(f"{engine} not supported")

            os.makedirs(DOWNLOADS_PATH, exist_ok=True)
            if not os.access(DOWNLOADS_PATH, os.W_OK):
                raise PermissionError(
                    f"No write permission for output directory: {DOWNLOADS_PATH}"
                )
            output_file_name = f"video_{job_type}_{str(uuid.uuid4())}.mp4"
            output_path = f"{DOWNLOADS_PATH}/{output_file_name}"

            if job_type == "text_to_video":
                prompt = text_to_video.get("prompt")
                video_name = text_to_video.get("name")
                duration = text_to_video.get("duration", 5)
                config = text_to_video.get(config_key, {})
                if prompt is None:
                    raise Exception("Prompt is required for video generation")
                self.output_message.actions.append(
                    f"Generating video using <b>{engine}</b> for prompt <i>{prompt}</i>"
                )
                self.output_message.push_update()
                video_gen_tool.text_to_video(
                    prompt=prompt,
                    save_at=output_path,
                    duration=duration,
                    config=config,
                )
            elif job_type == "image_to_video":
                image_id = image_to_video.get("image_id")
                video_name = image_to_video.get("name")
                duration = image_to_video.get("duration", 5)
                config = image_to_video.get(config_key, {})
                prompt = image_to_video.get("prompt")

                # Validate duration bounds
                if (
                    not isinstance(duration, (int, float))
                    or duration <= 0
                    or duration > 60
                ):
                    raise ValueError(
                        "Duration must be a positive number between 1 and 60 seconds"
                    )

                if not image_id:
                    raise ValueError(
                        "Missing required parameter: 'image_id' for image-to-video generation"
                    )

                image_data = self.videodb_tool.get_image(image_id)
                if not image_data:
                    raise ValueError(
                        f"Image with ID '{image_id}' not found in collection "
                        f"'{collection_id}'. Please verify the image ID."
                    )

                image_url = None
                if isinstance(image_data.get("url"), str) and image_data.get("url"):
                    image_url = image_data["url"]
                else:
                    image_url = self.videodb_tool.generate_image_url(image_id)

                if not image_url:
                    raise ValueError(
                        f"Image with ID '{image_id}' exists but has no "
                        f"associated URL. This might indicate data corruption."
                    )

                self.output_message.actions.append(
                    f"Generating video using <b>{engine}</b> for <a href='{image_url}'>url</a>"
                )
                if not stealth_mode:
                    self.output_message.push_update()

                video_gen_tool.image_to_video(
                    image_url=image_url,
                    save_at=output_path,
                    duration=duration,
                    config=config,
                    prompt=prompt,
                )
            else:
                raise Exception(f"{job_type} not supported")

            self.output_message.actions.append(
                f"Generated video saved at <i>{output_path}</i>"
            )
            self.output_message.push_update()

            # Upload to VideoDB
            media = self.videodb_tool.upload(
                output_path,
                source_type="file_path",
                media_type="video",
                name=video_name,
            )
            self.output_message.actions.append(
                f"Uploaded generated video to VideoDB with Video ID {media['id']}"
            )
            stream_url = media["stream_url"]
            id = media["id"]
            collection_id = media["collection_id"]
            name = media["name"]
            video_content.video = VideoData(
                stream_url=stream_url,
                id=id,
                collection_id=collection_id,
                name=name,
            )
            video_content.status = MsgStatus.success
            video_content.status_message = "Here is your generated video"
            self.output_message.push_update()
            self.output_message.publish()

        except Exception as e:
            logger.exception(f"Error in {self.agent_name} agent: {e}")
            video_content.status = MsgStatus.error
            video_content.status_message = "Failed to generate video"
            self.output_message.push_update()
            self.output_message.publish()
            return AgentResponse(status=AgentStatus.ERROR, message=str(e))

        return AgentResponse(
            status=AgentStatus.SUCCESS,
            message=f"Generated video ID {media['id']}",
            data={
                "video_id": media["id"],
                "video_stream_url": stream_url,
                "video_content": video_content,
            },
        )
```

### ./vendor/Director/backend/director/agents/base.py
```python
import logging

from abc import ABC, abstractmethod
from pydantic import BaseModel

from openai_function_calling import FunctionInferrer

from director.core.session import Session, OutputMessage

logger = logging.getLogger(__name__)


class AgentStatus:
    SUCCESS = "success"
    ERROR = "error"


class AgentResponse(BaseModel):
    """Data model for respones from agents."""

    status: str = AgentStatus.SUCCESS
    message: str = ""
    data: dict = {}


class BaseAgent(ABC):
    """Interface for all agents. All agents should inherit from this class."""

    def __init__(self, session: Session, **kwargs):
        self.session: Session = session
        self.output_message: OutputMessage = self.session.output_message

    def get_parameters(self):
        """Return the automatically inferred parameters for the function using the dcstring of the function."""
        function_inferrer = FunctionInferrer.infer_from_function_reference(self.run)
        function_json = function_inferrer.to_json_schema()
        parameters = function_json.get("parameters")
        if not parameters:
            raise Exception(
                "Failed to infere parameters, please define JSON instead of using this automated util."
            )
        return parameters

    def to_llm_format(self):
        """Convert the agent to LLM tool format."""
        return {
            "name": self.agent_name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @property
    def name(self):
        return self.agent_name

    @property
    def agent_description(self):
        return self.description

    def safe_call(self, *args, **kwargs):
        try:
            return self.run(*args, **kwargs)

        except Exception as e:
            logger.exception(f"error in {self.agent_name} agent: {e}")
            return AgentResponse(status=AgentStatus.ERROR, message=str(e))

    @abstractmethod
    def run(*args, **kwargs) -> AgentResponse:
        pass
```

### ./vendor/Director/backend/director/utils/asyncio.py
```python
import asyncio


def is_event_loop_running():
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False
```

### ./vendor/Director/backend/director/utils/__init__.py
```python
```

### ./vendor/Director/backend/director/utils/exceptions.py
```python
"""This module contains the exceptions used in the director package."""


class DirectorException(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message="An error occurred.", **kwargs):
        super(ValueError, self).__init__(message)


class AgentException(DirectorException):
    """Exception raised for errors in the agent."""

    def __init__(self, message="An error occurred in the agent", **kwargs):
        super(ValueError, self).__init__(message)


class ToolException(DirectorException):
    """Exception raised for errors in the tool."""

    def __init__(self, message="An error occurred in the tool", **kwargs):
        super(ValueError, self).__init__(message)
```

### ./vendor/Director/backend/director/entrypoint/__init__.py
```python
```

### ./vendor/Director/backend/director/entrypoint/api/server.py
```python
import os

from dotenv import load_dotenv
from director.entrypoint.api import create_app

load_dotenv()


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "engineio": {
            "handlers": ["console"],
            "level": "ERROR",
            "propagate": False,
        },
    },
}


class BaseAppConfig:
    """Base configuration for the app."""

    DEBUG: bool = 1
    """Debug mode for the app."""
    TESTING: bool = 1
    """Testing mode for the app."""
    SECRET_KEY: str = "secret"
    """Secret key for the app."""
    LOGGING_CONFIG: dict = LOGGING_CONFIG
    """Logging configuration for the app."""
    DB_TYPE: str = os.getenv("DB_TYPE", "sqlite")
    """Database type for the app."""
    HOST: str = "0.0.0.0"
    """Host for the app."""
    PORT: int = 8000
    """Port for the app."""
    ENV_PREFIX: str = "SERVER"


class LocalAppConfig(BaseAppConfig):
    """Local configuration for the app. All the default values can be change using environment variables. e.g. `SERVER_PORT=8001`"""

    TESTING: bool = 0
    """Testing mode for the app."""


class ProductionAppConfig(BaseAppConfig):
    """Production configuration for the app. All the default values can be change using environment variables. e.g. SERVER_PORT=8001"""

    DEBUG: bool = 0
    """Debug mode for the app."""
    TESTING: bool = 0
    """Testing mode for the app."""
    SECRET_KEY: str = "production"
    """Secret key for the app."""


configs = dict(local=LocalAppConfig, production=ProductionAppConfig)


# By default, the server is configured to run in development mode. To run in production mode, set the `SERVER_ENV` environment variable to `production`.
app = create_app(app_config=configs[os.getenv("SERVER_ENV", "local")])

if __name__ == "__main__":
    app.run(
        host=os.getenv("SERVER_HOST", app.config["HOST"]),
        port=os.getenv("SERVER_PORT", app.config["PORT"]),
        reloader_type="stat",
    )
```

### ./vendor/Director/backend/director/entrypoint/api/__init__.py
```python
"""
Initialize the app

Create an application factory function, which will be used to create a new app instance.

docs: https://flask.palletsprojects.com/en/2.3.x/patterns/appfactories/
"""

from flask_cors import CORS
from flask import Flask
from flask_socketio import SocketIO
from logging.config import dictConfig

from director.entrypoint.api.routes import agent_bp, session_bp, videodb_bp, config_bp
from director.entrypoint.api.socket_io import ChatNamespace

from dotenv import load_dotenv

load_dotenv()

socketio = SocketIO()


def create_app(app_config: object):
    """
    Create a Flask app using the app factory pattern.

    :param app_config: The configuration object to use.
    :return: A Flask app.
    """
    app = Flask(__name__)

    # Set the app config
    app.config.from_object(app_config)
    app.config.from_prefixed_env(app_config.ENV_PREFIX)
    CORS(app)

    # Init the socketio and attach it to the app
    socketio.init_app(
        app,
        cors_allowed_origins="*",
        logger=True,
        engineio_logger=True,
        reconnection=False if app.config["DEBUG"] else True,
    )
    app.socketio = socketio

    # Set the logging config
    dictConfig(app.config["LOGGING_CONFIG"])

    with app.app_context():
        from director.entrypoint.api import errors

    # register blueprints
    app.register_blueprint(agent_bp)
    app.register_blueprint(session_bp)
    app.register_blueprint(videodb_bp)
    app.register_blueprint(config_bp)

    # register socket namespaces
    socketio.on_namespace(ChatNamespace("/chat"))

    return app
```

### ./vendor/Director/backend/director/entrypoint/api/socket_io.py
```python
import os

from flask import current_app as app
from flask_socketio import Namespace

from director.db import load_db
from director.handler import ChatHandler


class ChatNamespace(Namespace):
    """Chat namespace for socket.io"""

    def on_chat(self, message):
        """Handle chat messages"""
        chat_handler = ChatHandler(
            db=load_db(os.getenv("SERVER_DB_TYPE", app.config["DB_TYPE"]))
        )
        chat_handler.chat(message)
```

### ./vendor/Director/backend/director/entrypoint/api/errors.py
```python
import logging


from flask import current_app as app, json, Response
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)


@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error

    logger.exception(e)

    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps(
        {
            "message": e.description,
        }
    )
    response.content_type = "application/json"
    return response


@app.errorhandler(Exception)
def handle_exception(e):
    """Return JSON instead of HTML for non-HTTP errors."""
    # pass through HTTP errors
    if isinstance(e, HTTPException):
        return e

    logger.exception(e)

    # now handling non-HTTP exceptions only
    resp = Response()
    resp.response = json.dumps(
        {
            "message": "internal server error",
            "detail": f"{type(e).__name__}: {e}",
        }
    )
    resp.status_code = 500
    resp.mimetype = "application/json"
    return resp


# handle pideantic validation error
@app.errorhandler(ValidationError)
def handle_validation_exception(error):
    """Handles all Pydantic validation errors."""

    logger.exception(error)

    resp = Response()
    resp.response = json.dumps(
        {"status": "error", "message": str(error), "detail": error.errors()}
    )
    resp.status = 400
    resp.mimetype = "application/json"
    return resp
```

### ./vendor/Director/backend/director/entrypoint/api/routes.py
```python
import os

from flask import Blueprint, request, current_app as app
from werkzeug.utils import secure_filename

from director.db import load_db
from director.handler import ChatHandler, SessionHandler, VideoDBHandler, ConfigHandler


agent_bp = Blueprint("agent", __name__, url_prefix="/agent")
session_bp = Blueprint("session", __name__, url_prefix="/session")
videodb_bp = Blueprint("videodb", __name__, url_prefix="/videodb")
config_bp = Blueprint("config", __name__, url_prefix="/config")


@agent_bp.route("/", methods=["GET"], strict_slashes=False)
def agent():
    """
    Handle the agent request
    """
    chat_handler = ChatHandler(
        db=load_db(os.getenv("SERVER_DB_TYPE", app.config["DB_TYPE"]))
    )
    return chat_handler.agents_list()


@session_bp.route("/", methods=["GET"], strict_slashes=False)
def get_sessions():
    """
    Get all the sessions
    """
    session_handler = SessionHandler(
        db=load_db(os.getenv("SERVER_DB_TYPE", app.config["DB_TYPE"]))
    )
    return session_handler.get_sessions()


@session_bp.route("/<session_id>", methods=["GET", "DELETE"])
def get_session(session_id):
    """
    Get or delete the session details
    """
    if not session_id:
        return {"message": f"Please provide {session_id}."}, 400

    session_handler = SessionHandler(
        db=load_db(os.getenv("SERVER_DB_TYPE", app.config["DB_TYPE"]))
    )
    session = session_handler.get_session(session_id)
    if not session:
        return {"message": "Session not found."}, 404

    if request.method == "GET":
        return session
    elif request.method == "DELETE":
        success, failed_components = session_handler.delete_session(session_id)
        if success:
            return {"message": "Session deleted successfully."}, 200
        else:
            return {
                "message": f"Failed to delete the entry for following components: {', '.join(failed_components)}"
            }, 500


@videodb_bp.route("/collection", defaults={"collection_id": None}, methods=["GET"])
@videodb_bp.route("/collection/<collection_id>", methods=["GET"])
def get_collection_or_all(collection_id):
    """Get a collection by ID or all collections."""
    videodb = VideoDBHandler(collection_id)
    if collection_id:
        return videodb.get_collection()
    else:
        return videodb.get_collections()


@videodb_bp.route("/collection", methods=["POST"])
def create_collection():
    try:
        data = request.get_json()

        if not data or not data.get("name"):
            return {"message": "Collection name is required"}, 400

        if not data.get("description"):
            return {"message": "Collection description is required"}, 400

        collection_name = data["name"]
        description = data["description"]

        videodb = VideoDBHandler()
        result = videodb.create_collection(collection_name, description)

        if result.get("success"):
            return {"message": "Collection created successfully", "data": result}, 201
        else:
            return {
                "message": "Failed to create collection",
                "error": result.get("error"),
            }, 400
    except Exception as e:
        return {"message": str(e)}, 500


@videodb_bp.route("/collection/<collection_id>", methods=["DELETE"])
def delete_collection(collection_id):
    try:
        if not collection_id:
            return {"message": "Collection ID is required"}, 400

        videodb = VideoDBHandler(collection_id)
        result = videodb.delete_collection()
        return result, 200
    except Exception as e:
        return {"message": str(e)}, 500


@videodb_bp.route(
    "/collection/<collection_id>/video", defaults={"video_id": None}, methods=["GET"]
)
@videodb_bp.route("/collection/<collection_id>/video/<video_id>", methods=["GET"])
def get_video_or_all(collection_id, video_id):
    """Get a video by ID or all videos in a collection."""
    videodb = VideoDBHandler(collection_id)
    if video_id:
        return videodb.get_video(video_id)
    else:
        return videodb.get_videos()


@videodb_bp.route("/collection/<collection_id>/video/<video_id>", methods=["DELETE"])
def delete_video(collection_id, video_id):
    """Delete a video by ID from a specific collection."""
    try:
        if not video_id:
            return {"message": "Video ID is required"}, 400
        videodb = VideoDBHandler(collection_id)
        result = videodb.delete_video(video_id)
        return result, 200
    except Exception as e:
        return {"message": str(e)}, 500


@videodb_bp.route(
    "/collection/<collection_id>/image/<image_id>/generate_url", methods=["GET"]
)
def generate_image_url(collection_id, image_id):
    try:
        if not collection_id:
            return {"message": "Collection ID is required"}, 400

        if not image_id:
            return {"message": "Image ID is required"}, 400

        videodb = VideoDBHandler(collection_id)
        result = videodb.generate_image_url(image_id)
        return result, 200
    except Exception as e:
        return {"message": str(e)}, 500


@videodb_bp.route("/collection/<collection_id>/upload", methods=["POST"])
def upload_video(collection_id):
    """Upload a video to a collection."""
    try:
        videodb = VideoDBHandler(collection_id)

        if "file" in request.files:
            file = request.files["file"]
            file_bytes = file.read()
            safe_filename = secure_filename(file.filename)
            if not safe_filename:
                return {"message": "Invalid filename"}, 400
            file_name = os.path.splitext(safe_filename)[0]
            media_type = file.content_type.split("/")[0]
            return videodb.upload(
                source=file_bytes,
                source_type="file",
                media_type=media_type,
                name=file_name,
            )
        elif "source" in request.json:
            source = request.json["source"]
            source_type = request.json["source_type"]
            return videodb.upload(source=source, source_type=source_type)
        else:
            return {"message": "No valid source provided"}, 400
    except Exception as e:
        return {"message": str(e)}, 500


@config_bp.route("/check", methods=["GET"])
def config_check():
    config_handler = ConfigHandler()
    return config_handler.check()
```

### ./vendor/Director/backend/director/db/__init__.py
```python
import os
from director.constants import DBType
from .base import BaseDB
from .sqlite.db import SQLiteDB
from .postgres.db import PostgresDB

db_types = {
    DBType.SQLITE: SQLiteDB,
    DBType.POSTGRES: PostgresDB,
}

def load_db(db_type: str = None) -> BaseDB:
    if db_type is None:
        db_type = os.getenv("DB_TYPE", "sqlite").lower()
    if db_type not in db_types:
        raise ValueError(
            f"Unknown DB type: {db_type}, Valid db types are: {[db_type.value for db_type in db_types]}"
        )
    return db_types[DBType(db_type)]()
```

### ./vendor/Director/backend/director/db/sqlite/db.py
```python
import json
import sqlite3
import time
import logging
import os

from typing import List

from director.constants import DBType
from director.db.base import BaseDB
from director.db.sqlite.initialize import initialize_sqlite

logger = logging.getLogger(__name__)


class SQLiteDB(BaseDB):
    def __init__(self, db_path: str = None):
        """
        :param db_path: Path to the SQLite database file.
        """
        self.db_type = DBType.SQLITE
        if db_path is None:
            self.db_path = os.getenv("SQLITE_DB_PATH", "director.db")
        else:
            self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=True)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        logger.info("Connected to SQLite DB...")

    def create_session(
        self,
        session_id: str,
        video_id: str,
        collection_id: str,
        created_at: int = None,
        updated_at: int = None,
        metadata: dict = {},
        **kwargs,
    ) -> None:
        """Create a new session.

        :param session_id: Unique session ID.
        :param video_id: ID of the video associated with the session.
        :param collection_id: ID of the collection associated with the session.
        :param created_at: Timestamp when the session was created.
        :param updated_at: Timestamp when the session was last updated.
        :param metadata: Additional metadata for the session.
        """
        created_at = created_at or int(time.time())
        updated_at = updated_at or int(time.time())

        self.cursor.execute(
            """
        INSERT OR IGNORE INTO sessions (session_id, video_id, collection_id, created_at, updated_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                video_id,
                collection_id,
                created_at,
                updated_at,
                json.dumps(metadata),
            ),
        )
        self.conn.commit()

    def get_session(self, session_id: str) -> dict:
        """Get a session by session_id.

        :param session_id: Unique session ID.
        :return: Session data as a dictionary.
        :rtype: dict
        """
        self.cursor.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        )
        row = self.cursor.fetchone()
        if row is not None:
            session = dict(row)  # Convert sqlite3.Row to dictionary
            session["metadata"] = json.loads(session["metadata"])
            return session

        else:
            return {}  # Return an empty dictionary if no data found

    def get_sessions(self) -> list:
        """Get all sessions.

        :return: List of all sessions.
        :rtype: list
        """
        self.cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
        row = self.cursor.fetchall()
        sessions = [dict(r) for r in row]
        for s in sessions:
            s["metadata"] = json.loads(s["metadata"])
        return sessions

    def add_or_update_msg_to_conv(
        self,
        session_id: str,
        conv_id: str,
        msg_id: str,
        msg_type: str,
        agents: List[str],
        actions: List[str],
        content: List[dict],
        status: str = None,
        created_at: int = None,
        updated_at: int = None,
        metadata: dict = {},
        **kwargs,
    ) -> None:
        """Add a new message (input or output) to the conversation.

        :param str session_id: Unique session ID.
        :param str conv_id: Unique conversation ID.
        :param str msg_id: Unique message ID.
        :param str msg_type: Type of message (input or output).
        :param list agents: List of agents involved in the conversation.
        :param list actions: List of actions taken by the agents.
        :param list content: List of message content.
        :param str status: Status of the message.
        :param int created_at: Timestamp when the message was created.
        :param int updated_at: Timestamp when the message was last updated.
        :param dict metadata: Additional metadata for the message.
        """
        created_at = created_at or int(time.time())
        updated_at = updated_at or int(time.time())

        self.cursor.execute(
            """
        INSERT OR REPLACE INTO conversations (session_id, conv_id, msg_id, msg_type, agents, actions, content, status, created_at, updated_at, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                conv_id,
                msg_id,
                msg_type,
                json.dumps(agents),
                json.dumps(actions),
                json.dumps(content),
                status,
                created_at,
                updated_at,
                json.dumps(metadata),
            ),
        )
        self.conn.commit()

    def get_conversations(self, session_id: str) -> list:
        self.cursor.execute(
            "SELECT * FROM conversations WHERE session_id = ?", (session_id,)
        )
        rows = self.cursor.fetchall()
        conversations = []
        for row in rows:
            if row is not None:
                conv_dict = dict(row)
                conv_dict["agents"] = json.loads(conv_dict["agents"])
                conv_dict["actions"] = json.loads(conv_dict["actions"])
                conv_dict["content"] = json.loads(conv_dict["content"])
                conv_dict["metadata"] = json.loads(conv_dict["metadata"])
                conversations.append(conv_dict)
        return conversations

    def get_context_messages(self, session_id: str) -> list:
        """Get context messages for a session.

        :param str session_id: Unique session ID.
        :return: List of context messages.
        :rtype: list
        """
        self.cursor.execute(
            "SELECT context_data FROM context_messages WHERE session_id = ?",
            (session_id,),
        )
        result = self.cursor.fetchone()
        return json.loads(result[0]) if result else {}

    def add_or_update_context_msg(
        self,
        session_id: str,
        context_messages: list,
        created_at: int = None,
        updated_at: int = None,
        metadata: dict = {},
        **kwargs,
    ) -> None:
        """Update context messages for a session.

        :param str session_id: Unique session ID.
        :param List context_messages: List of context messages.
        :param int created_at: Timestamp when the context messages were created.
        :param int updated_at: Timestamp when the context messages were last updated.
        :param dict metadata: Additional metadata for the context messages.
        """
        created_at = created_at or int(time.time())
        updated_at = updated_at or int(time.time())

        self.cursor.execute(
            """
        INSERT OR REPLACE INTO context_messages (context_data, session_id, created_at, updated_at, metadata)
        VALUES (?, ?, ?, ?, ?)
        """,
            (
                json.dumps(context_messages),
                session_id,
                created_at,
                updated_at,
                json.dumps(metadata),
            ),
        )
        self.conn.commit()

    def delete_conversation(self, session_id: str) -> bool:
        """Delete all conversations for a given session.

        :param str session_id: Unique session ID.
        :return: True if conversations were deleted, False otherwise.
        """
        self.cursor.execute(
            "DELETE FROM conversations WHERE session_id = ?", (session_id,)
        )
        self.conn.commit()
        return self.cursor.rowcount > 0

    def delete_context(self, session_id: str) -> bool:
        """Delete context messages for a given session.

        :param str session_id: Unique session ID.
        :return: True if context messages were deleted, False otherwise.
        """
        self.cursor.execute(
            "DELETE FROM context_messages WHERE session_id = ?", (session_id,)
        )
        self.conn.commit()
        return self.cursor.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its associated data.

        :param str session_id: Unique session ID.
        :return: True if the session was deleted, False otherwise.
        """
        failed_components = []
        if not self.delete_conversation(session_id):
            failed_components.append("conversation")
        if not self.delete_context(session_id):
            failed_components.append("context")
        self.cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        self.conn.commit()
        if not self.cursor.rowcount > 0:
            failed_components.append("session")
        success = len(failed_components) < 3
        return success, failed_components

    def health_check(self) -> bool:
        """Check if the SQLite database is healthy and the necessary tables exist. If not, create them."""
        try:
            query = """
                SELECT COUNT(name)
                FROM sqlite_master
                WHERE type='table'
                AND name IN ('sessions', 'conversations', 'context_messages');
            """
            self.cursor.execute(query)
            table_count = self.cursor.fetchone()[0]
            if table_count < 3:
                logger.info("Tables not found. Initializing SQLite DB...")
                initialize_sqlite(self.db_path)
            return True

        except Exception as e:
            logger.exception(f"SQLite health check failed: {e}")
            return False

    def __del__(self):
        self.conn.close()
```

### ./vendor/Director/backend/director/db/sqlite/initialize.py
```python
import sqlite3
import os

# SQL to create the sessions table
CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    video_id TEXT,
    collection_id TEXT,
    created_at INTEGER,
    updated_at INTEGER,
    metadata JSON
)
"""

# SQL to create the conversations table
CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    session_id TEXT,
    conv_id TEXT,
    msg_id TEXT PRIMARY KEY,
    msg_type TEXT,
    agents JSON,
    actions JSON,
    content JSON,
    status TEXT,
    created_at INTEGER,
    updated_at INTEGER,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
)
"""

# SQL to create the context_messages table
CREATE_CONTEXT_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS context_messages (
    session_id TEXT PRIMARY KEY,
    context_data JSON,
    created_at INTEGER,
    updated_at INTEGER,
    metadata JSON,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
)
"""


def initialize_sqlite(db_name="director.db"):
    """Initialize the SQLite database by creating the necessary tables."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute(CREATE_SESSIONS_TABLE)
    cursor.execute(CREATE_CONVERSATIONS_TABLE)
    cursor.execute(CREATE_CONTEXT_MESSAGES_TABLE)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    db_path = os.getenv("SQLITE_DB_PATH", "director.db")
    initialize_sqlite(db_path)
```

### ./vendor/Director/backend/director/db/sqlite/__init__.py
```python
```

### ./vendor/Director/backend/director/db/postgres/db.py
```python
import json
import time
import logging
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List

from director.constants import DBType
from director.db.base import BaseDB
from director.db.postgres.initialize import initialize_postgres

logger = logging.getLogger(__name__)


class PostgresDB(BaseDB):
    def __init__(self):
        """Initialize PostgreSQL connection using environment variables."""
        self.db_type = DBType.POSTGRES
        self.conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", "postgres"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "postgres"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", "5432"),
        )
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)

        logger.info("Connected to PostgreSQL DB..........")

    def create_session(
        self,
        session_id: str,
        video_id: str,
        collection_id: str,
        created_at: int = None,
        updated_at: int = None,
        metadata: dict = {},
        **kwargs,
    ) -> None:
        created_at = created_at or int(time.time())
        updated_at = updated_at or int(time.time())

        self.cursor.execute(
            """
            INSERT INTO sessions (session_id, video_id, collection_id, created_at, updated_at, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO NOTHING
            """,
            (
                session_id,
                video_id,
                collection_id,
                created_at,
                updated_at,
                json.dumps(metadata),
            ),
        )
        self.conn.commit()

    def get_session(self, session_id: str) -> dict:
        self.cursor.execute(
            "SELECT * FROM sessions WHERE session_id = %s", (session_id,)
        )
        row = self.cursor.fetchone()
        if row is not None:
            session = dict(row)
            return session
        return {}

    def get_sessions(self) -> list:
        self.cursor.execute("SELECT * FROM sessions ORDER BY updated_at DESC")
        rows = self.cursor.fetchall()
        return [dict(r) for r in rows]

    def add_or_update_msg_to_conv(
        self,
        session_id: str,
        conv_id: str,
        msg_id: str,
        msg_type: str,
        agents: List[str],
        actions: List[str],
        content: List[dict],
        status: str = None,
        created_at: int = None,
        updated_at: int = None,
        metadata: dict = {},
        **kwargs,
    ) -> None:
        created_at = created_at or int(time.time())
        updated_at = updated_at or int(time.time())

        self.cursor.execute(
            """
            INSERT INTO conversations (
                session_id, conv_id, msg_id, msg_type, agents, actions,
                content, status, created_at, updated_at, metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (msg_id) DO UPDATE SET
                session_id = EXCLUDED.session_id,
                conv_id = EXCLUDED.conv_id,
                msg_type = EXCLUDED.msg_type,
                agents = EXCLUDED.agents,
                actions = EXCLUDED.actions,
                content = EXCLUDED.content,
                status = EXCLUDED.status,
                updated_at = EXCLUDED.updated_at,
                metadata = EXCLUDED.metadata
            """,
            (
                session_id,
                conv_id,
                msg_id,
                msg_type,
                json.dumps(agents),
                json.dumps(actions),
                json.dumps(content),
                status,
                created_at,
                updated_at,
                json.dumps(metadata),
            ),
        )
        self.conn.commit()

    def get_conversations(self, session_id: str) -> list:
        self.cursor.execute(
            "SELECT * FROM conversations WHERE session_id = %s", (session_id,)
        )
        rows = self.cursor.fetchall()
        conversations = []
        for row in rows:
            if row is not None:
                conv_dict = dict(row)
                conversations.append(conv_dict)
        return conversations

    def get_context_messages(self, session_id: str) -> list:
        self.cursor.execute(
            "SELECT context_data FROM context_messages WHERE session_id = %s",
            (session_id,),
        )
        result = self.cursor.fetchone()
        return result["context_data"] if result else {}

    def add_or_update_context_msg(
        self,
        session_id: str,
        context_messages: list,
        created_at: int = None,
        updated_at: int = None,
        metadata: dict = {},
        **kwargs,
    ) -> None:
        created_at = created_at or int(time.time())
        updated_at = updated_at or int(time.time())

        self.cursor.execute(
            """
            INSERT INTO context_messages (context_data, session_id, created_at, updated_at, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (session_id) DO UPDATE SET
                context_data = EXCLUDED.context_data,
                updated_at = EXCLUDED.updated_at,
                metadata = EXCLUDED.metadata
            """,
            (
                json.dumps(context_messages),
                session_id,
                created_at,
                updated_at,
                json.dumps(metadata),
            ),
        )
        self.conn.commit()

    def delete_conversation(self, session_id: str) -> bool:
        self.cursor.execute(
            "DELETE FROM conversations WHERE session_id = %s", (session_id,)
        )
        self.conn.commit()
        return self.cursor.rowcount > 0

    def delete_context(self, session_id: str) -> bool:
        self.cursor.execute(
            "DELETE FROM context_messages WHERE session_id = %s", (session_id,)
        )
        self.conn.commit()
        return self.cursor.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        failed_components = []
        if not self.delete_conversation(session_id):
            failed_components.append("conversation")
        if not self.delete_context(session_id):
            failed_components.append("context")

        self.cursor.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
        self.conn.commit()
        if not self.cursor.rowcount > 0:
            failed_components.append("session")

        success = len(failed_components) < 3
        return success, failed_components

    def health_check(self) -> bool:
        try:
            query = """
                SELECT COUNT(table_name)
                FROM information_schema.tables
                WHERE table_name IN ('sessions', 'conversations', 'context_messages')
                AND table_schema = 'public';
            """
            self.cursor.execute(query)
            table_count = self.cursor.fetchone()["count"]

            if table_count < 3:
                logger.info("Tables not found. Initializing PostgreSQL DB...")
                initialize_postgres()
            return True

        except Exception as e:
            logger.exception(f"PostgreSQL health check failed: {e}")
            return False

    def __del__(self):
        if hasattr(self, "conn") and self.conn:
            self.conn.close()
```

### ./vendor/Director/backend/director/db/postgres/initialize.py
```python
import os
import psycopg2
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    video_id TEXT,
    collection_id TEXT,
    created_at BIGINT,
    updated_at BIGINT,
    metadata JSONB
);
"""

CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    session_id TEXT,
    conv_id TEXT,
    msg_id TEXT PRIMARY KEY,
    msg_type TEXT,
    agents JSONB,
    actions JSONB,
    content JSONB,
    status TEXT,
    created_at BIGINT,
    updated_at BIGINT,
    metadata JSONB,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""

CREATE_CONTEXT_MESSAGES_TABLE = """
CREATE TABLE IF NOT EXISTS context_messages (
    session_id TEXT PRIMARY KEY,
    context_data JSONB,
    created_at BIGINT,
    updated_at BIGINT,
    metadata JSONB,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""

def initialize_postgres():
    """Initialize the PostgreSQL database by creating the necessary tables."""
    conn = psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "postgres"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432")
    )
    cursor = conn.cursor()

    try:
        cursor.execute(CREATE_SESSIONS_TABLE)
        cursor.execute(CREATE_CONVERSATIONS_TABLE)
        cursor.execute(CREATE_CONTEXT_MESSAGES_TABLE)
        conn.commit()
        logger.info("PostgreSQL tables created successfully")
    except Exception as e:
        logger.exception(f"Error creating PostgreSQL tables: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    initialize_postgres()
```

### ./vendor/Director/backend/director/db/base.py
```python
from abc import ABC, abstractmethod


class BaseDB(ABC):
    """Interface for all databases. It provides a common interface for all databases to follow."""

    @abstractmethod
    def create_session(
        self, session_id: str, video_id: str = None, collection_id: str = None
    ) -> None:
        """Create a new session."""
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> dict:
        """Get a session by session_id."""
        pass

    @abstractmethod
    def get_sessions(self) -> list:
        """Get all sessions."""
        pass

    @abstractmethod
    def add_or_update_msg_to_conv() -> None:
        """Add a new message (input or output) to the conversation."""
        pass

    @abstractmethod
    def get_conversations(self, session_id: str) -> list:
        """Get all conversations for a given session."""
        pass

    @abstractmethod
    def get_context_messages(self, session_id: str) -> list:
        """Get context messages for a session."""
        pass

    @abstractmethod
    def add_or_update_context_msg(
        self, session_id: str, context_messages: list
    ) -> None:
        """Update context messages for a session."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the database is healthy."""
        pass
```

### ./vendor/Director/docs/hooks/copyright.py
```python
from datetime import datetime


def on_config(config, **kwargs):
    config.copyright = f"Copyright © {datetime.now().year} VideoDB"
```

### ./src/director/tools/music_tool.py
```python
from typing import Dict, Any, List, Optional
import random
import os
from ..core.base_tool import BaseTool

class MusicTool(BaseTool):
    """Tool for selecting and managing background music for videos."""
    
    def __init__(self):
        super().__init__()
        self.music_dir = os.path.join(os.getcwd(), "assets/audio/music")
        os.makedirs(self.music_dir, exist_ok=True)
        
        self.music_library = {
            "modern": {
                "upbeat": [
                    {"name": "Urban Energy", "duration": 60, "bpm": 128, "key": "C", "mood": "energetic", "path": os.path.join(self.music_dir, "urban_energy.mp3")},
                    {"name": "City Pulse", "duration": 45, "bpm": 120, "key": "G", "mood": "dynamic", "path": os.path.join(self.music_dir, "city_pulse.mp3")},
                    {"name": "Modern Flow", "duration": 55, "bpm": 125, "key": "Am", "mood": "groovy", "path": os.path.join(self.music_dir, "modern_flow.mp3")}
                ],
                "chill": [
                    {"name": "Lofi Dreams", "duration": 50, "bpm": 90, "key": "Dm", "mood": "relaxed", "path": os.path.join(self.music_dir, "lofi_dreams.mp3")},
                    {"name": "Urban Calm", "duration": 65, "bpm": 85, "key": "Em", "mood": "peaceful", "path": os.path.join(self.music_dir, "urban_calm.mp3")},
                    {"name": "City Nights", "duration": 48, "bpm": 95, "key": "F", "mood": "smooth", "path": os.path.join(self.music_dir, "city_nights.mp3")}
                ]
            },
            "luxury": {
                "elegant": [
                    {"name": "Golden Hour", "duration": 55, "bpm": 110, "key": "D", "mood": "sophisticated", "path": os.path.join(self.music_dir, "golden_hour.mp3")},
                    {"name": "Luxury Suite", "duration": 62, "bpm": 105, "key": "Em", "mood": "refined", "path": os.path.join(self.music_dir, "luxury_suite.mp3")},
                    {"name": "Elite Status", "duration": 58, "bpm": 108, "key": "Bm", "mood": "prestigious", "path": os.path.join(self.music_dir, "elite_status.mp3")}
                ],
                "ambient": [
                    {"name": "Velvet Dreams", "duration": 70, "bpm": 80, "key": "C", "mood": "atmospheric", "path": os.path.join(self.music_dir, "velvet_dreams.mp3")},
                    {"name": "Premium Space", "duration": 65, "bpm": 75, "key": "G", "mood": "ethereal", "path": os.path.join(self.music_dir, "premium_space.mp3")},
                    {"name": "High End", "duration": 60, "bpm": 85, "key": "Am", "mood": "luxurious", "path": os.path.join(self.music_dir, "high_end.mp3")}
                ]
            },
            "minimal": {
                "ambient": [
                    {"name": "Pure Space", "duration": 45, "bpm": 70, "key": "C", "mood": "minimal", "path": os.path.join(self.music_dir, "pure_space.mp3")},
                    {"name": "Clean Lines", "duration": 50, "bpm": 65, "key": "Em", "mood": "spacious", "path": os.path.join(self.music_dir, "clean_lines.mp3")},
                    {"name": "Simple Beauty", "duration": 55, "bpm": 72, "key": "G", "mood": "clean", "path": os.path.join(self.music_dir, "simple_beauty.mp3")}
                ],
                "soft": [
                    {"name": "Gentle Flow", "duration": 58, "bpm": 80, "key": "Am", "mood": "subtle", "path": os.path.join(self.music_dir, "gentle_flow.mp3")},
                    {"name": "Soft Touch", "duration": 52, "bpm": 75, "key": "Dm", "mood": "delicate", "path": os.path.join(self.music_dir, "soft_touch.mp3")},
                    {"name": "Light Air", "duration": 48, "bpm": 78, "key": "F", "mood": "airy", "path": os.path.join(self.music_dir, "light_air.mp3")}
                ]
            }
        }
        
        # Mapping of template pacing to music categories
        self.pacing_music_map = {
            "rapid": ["upbeat", "elegant"],
            "balanced": ["upbeat", "elegant", "ambient"],
            "relaxed": ["chill", "ambient", "soft"]
        }
    
    def select_music(
        self,
        template_name: str,
        duration: float,
        mood_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Select appropriate background music for a video.
        
        Args:
            template_name: Name of the template being used
            duration: Desired duration of the video
            mood_override: Optional override for mood selection
            
        Returns:
            Dictionary containing selected music track information
        """
        # For now, use a default track for testing
        track = {
            "name": "Lofi Dreams",
            "duration": 50,
            "bpm": 90,
            "key": "Dm",
            "mood": "relaxed",
            "path": os.path.join(self.music_dir, "lofi_dreams.mp3")
        }
        
        # Calculate number of loops needed
        loop_count = int(duration / track["duration"]) + 1
        
        # Add fade out if looping
        fade_out = loop_count > 1
        
        return {
            **track,
            "loop_count": loop_count,
            "fade_out": fade_out
        }
        
        if not matching_tracks:
            raise ValueError(f"No suitable music found for template {template_name}")
        
        # Select random track from matches
        selected_track = random.choice(matching_tracks)
        
        # Add any necessary adjustments for duration
        result = selected_track.copy()
        result["loop_count"] = max(1, int(duration / selected_track["duration"]))
        result["fade_out"] = True if result["loop_count"] > 1 else False
        
        return result
    
    def get_available_moods(self, template_name: str) -> List[str]:
        """Get available music moods for a template."""
        style = template_name.split('_')[0]
        moods = set()
        
        if style in self.music_library:
            for category in self.music_library[style].values():
                for track in category:
                    moods.add(track["mood"])
        
        return sorted(list(moods))
    
    def get_track_by_name(self, track_name: str) -> Optional[Dict[str, Any]]:
        """Get track information by name."""
        for style in self.music_library.values():
            for category in style.values():
                for track in category:
                    if track["name"] == track_name:
                        return track
        return None
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Get tool capabilities and requirements."""
        return {
            "name": "music_tool",
            "description": "Selects and manages background music for videos",
            "supported_styles": list(self.music_library.keys()),
            "features": [
                "style_based_selection",
                "mood_matching",
                "tempo_matching",
                "duration_adjustment"
            ]
        }
```

### ./src/director/tools/voiceover_tool.py
```python
from typing import Dict, Any, Optional
import os
import logging
from ..core.base_tool import BaseTool

logger = logging.getLogger(__name__)

class VoiceoverTool(BaseTool):
    """Tool for generating voiceovers using ElevenLabs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        super().__init__(config)
        if not self.api_key:
            logger.warning("ELEVENLABS_API_KEY not found in environment")
        
        # Load configuration from environment
        self.model = os.getenv("ELEVENLABS_MODEL", "eleven_monolingual_v1")
        self.default_stability = float(os.getenv("ELEVENLABS_VOICE_STABILITY", "0.75"))
        self.default_similarity = float(os.getenv("ELEVENLABS_VOICE_SIMILARITY", "0.75"))
            
        self.voice_profiles = {
            "luxury": {
                "voice_id": "ErXwobaYiN019PkySvjV",  # Antoni - Professional male voice
                "settings": {
                    "stability": self.default_stability,
                    "similarity_boost": self.default_similarity,
                    "style": 0.5,
                    "use_speaker_boost": True
                }
            },
            "friendly": {
                "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Bella - Female voice
                "settings": {
                    "stability": self.default_stability,
                    "similarity_boost": self.default_similarity,
                    "style": 0.3,
                    "use_speaker_boost": True
                }
            },
            "professional": {
                "voice_id": "VR6AewLTigWG4xSOukaG",  # Adam - Professional male voice
                "settings": {
                    "stability": self.default_stability,
                    "similarity_boost": self.default_similarity,
                    "style": 0.4,
                    "use_speaker_boost": True
                }
            }
        }
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Get tool capabilities and requirements."""
        return {
            "name": "voiceover_tool",
            "description": "Generates professional voiceovers using ElevenLabs API",
            "requires_api_key": True,
            "supported_voices": list(self.voice_profiles.keys()),
            "max_text_length": 5000,
            "supported_formats": ["mp3", "wav"]
        }
        
    def generate_voiceover(
        self,
        text: str,
        style: str = "professional",
        output_format: str = "mp3"
    ) -> Dict[str, Any]:
        """
        Generate a voiceover for the given text using ElevenLabs API.
        
        Args:
            text: Text to convert to speech
            style: Voice style to use
            output_format: Output audio format
            
        Returns:
            Dictionary containing voiceover metadata and file path
        """
        if not self.api_key:
            raise ValueError("ElevenLabs API key not configured")
            
        if style not in self.voice_profiles:
            raise ValueError(f"Unknown voice style: {style}")
            
        if len(text) > self.get_capabilities()["max_text_length"]:
            raise ValueError("Text exceeds maximum length")
            
        try:
            import requests
            import uuid
            
            # Get voice profile
            profile = self.voice_profiles[style]
            
            # Prepare request
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{profile['voice_id']}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": self.model,
                "voice_settings": profile["settings"]
            }
            
            # Make API request
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                # Save audio file
                output_dir = os.path.join(os.getcwd(), "assets/audio/voiceover")
                os.makedirs(output_dir, exist_ok=True)
                
                file_name = f"voiceover_{uuid.uuid4()}.{output_format}"
                file_path = os.path.join(output_dir, file_name)
                
                with open(file_path, "wb") as f:
                    f.write(response.content)
                
                return {
                    "status": "success",
                    "file_path": file_path,
                    "text": text,
                    "voice_id": profile["voice_id"],
                    "settings": profile["settings"]
                }
            else:
                error_msg = f"ElevenLabs API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
        except Exception as e:
            logger.error(f"Voiceover generation failed: {str(e)}")
            raise
            
    def validate_config(self) -> None:
        """Validate tool configuration."""
        if not self.api_key and not self.config.get("mock_mode", False):
            logger.warning("No API key provided and mock mode is disabled")
```

### ./src/director/tools/video_db_tool.py
```python
from typing import Dict, Any, List, Optional
import random

class VideoDBTool:
    """Tool for video processing and transitions using VideoDBTool."""
    
    def __init__(self):
        self.transition_library = {
            "modern": {
                "standard": [
                    {"name": "slide", "duration": 0.5, "direction": "left"},
                    {"name": "fade", "duration": 0.4, "ease": "smooth"},
                    {"name": "zoom", "duration": 0.6, "scale": 1.2}
                ],
                "dynamic": [
                    {"name": "swipe", "duration": 0.4, "direction": "up"},
                    {"name": "reveal", "duration": 0.5, "direction": "right"},
                    {"name": "blur", "duration": 0.6, "intensity": 0.8}
                ]
            },
            "luxury": {
                "elegant": [
                    {"name": "dissolve", "duration": 0.8, "ease": "smooth"},
                    {"name": "glide", "duration": 0.7, "direction": "up"},
                    {"name": "fade", "duration": 0.6, "ease": "elegant"}
                ],
                "premium": [
                    {"name": "reveal", "duration": 0.9, "direction": "diagonal"},
                    {"name": "sweep", "duration": 0.8, "direction": "right"},
                    {"name": "blur", "duration": 0.7, "intensity": 0.6}
                ]
            },
            "minimal": {
                "clean": [
                    {"name": "fade", "duration": 0.4, "ease": "linear"},
                    {"name": "slide", "duration": 0.5, "direction": "left"},
                    {"name": "dissolve", "duration": 0.4, "ease": "smooth"}
                ],
                "simple": [
                    {"name": "cut", "duration": 0.3},
                    {"name": "fade", "duration": 0.5, "ease": "linear"},
                    {"name": "slide", "duration": 0.4, "direction": "up"}
                ]
            }
        }
        
        self.effect_library = {
            "modern": {
                "color": {
                    "contrast": 1.1,
                    "saturation": 1.2,
                    "brightness": 1.05
                },
                "filters": ["vibrant", "sharp"],
                "overlays": ["grain", "light_leak"]
            },
            "luxury": {
                "color": {
                    "contrast": 1.15,
                    "saturation": 0.95,
                    "brightness": 1.0
                },
                "filters": ["cinematic", "warm"],
                "overlays": ["film", "subtle_flare"]
            },
            "minimal": {
                "color": {
                    "contrast": 1.05,
                    "saturation": 0.9,
                    "brightness": 1.1
                },
                "filters": ["clean", "crisp"],
                "overlays": ["none"]
            }
        }
    
    def get_transitions(
        self,
        template_name: str,
        count: int = 3,
        variation: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Get a list of transitions for a video.
        
        Args:
            template_name: Name of the template being used
            count: Number of transitions to return
            variation: Amount of random variation to apply (0-1)
            
        Returns:
            List of transition configurations
        """
        style = template_name.split('_')[0]
        
        if style not in self.transition_library:
            style = "modern"  # fallback
            
        # Get all transitions for this style
        available_transitions = []
        for category in self.transition_library[style].values():
            available_transitions.extend(category)
            
        # Select random transitions
        selected = []
        for _ in range(count):
            transition = random.choice(available_transitions).copy()
            
            # Add variation
            if "duration" in transition:
                base_duration = transition["duration"]
                min_duration = base_duration * (1 - variation)
                max_duration = base_duration * (1 + variation)
                transition["duration"] = random.uniform(min_duration, max_duration)
                
            selected.append(transition)
            
        return selected
    
    def get_effects(
        self,
        template_name: str,
        intensity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Get video effects configuration.
        
        Args:
            template_name: Name of the template being used
            intensity: Intensity of effects (0-1)
            
        Returns:
            Effects configuration
        """
        style = template_name.split('_')[0]
        
        if style not in self.effect_library:
            style = "modern"  # fallback
            
        effects = self.effect_library[style].copy()
        
        # Adjust effect intensity
        color = effects["color"].copy()
        for key in color:
            deviation = abs(1 - color[key])
            color[key] = 1 + (deviation * intensity * (1 if color[key] > 1 else -1))
            
        effects["color"] = color
        effects["intensity"] = intensity
        
        return effects
    
    def process_segment(
        self,
        segment: Dict[str, Any],
        effects: Dict[str, Any],
        transition: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a video segment with effects and transition.
        
        Args:
            segment: Segment configuration
            effects: Effects to apply
            transition: Optional transition configuration
            
        Returns:
            Processed segment configuration
        """
        processed = segment.copy()
        processed["effects"] = effects
        if transition:
            processed["transition"] = transition
        return processed
```

### ./src/director/tools/video_renderer.py
```python
from typing import Dict, Any, List, Optional
import os
import logging
import moviepy.editor as mp
from moviepy.editor import VideoFileClip, ImageClip, TextClip, CompositeVideoClip, AudioFileClip, CompositeAudioClip
import firebase_admin
from firebase_admin import storage
import tempfile
import uuid

logger = logging.getLogger(__name__)

class VideoRenderer:
    """Tool for rendering video segments into a final video."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize Firebase if not already initialized
        try:
            self.bucket = storage.bucket()
        except ValueError:
            cred = firebase_admin.credentials.Certificate("config/firebase-service-account.json")
            firebase_admin.initialize_app(cred, {
                'storageBucket': 'project3-ziltok.firebasestorage.app'
            })
            self.bucket = storage.bucket()
    
    def download_image(self, image_url: str) -> str:
        """Download image from Firebase Storage."""
        if not image_url.startswith("gs://"):
            return image_url
            
        # Extract blob path from gs:// URL
        blob_path = image_url.replace("gs://project3-ziltok.firebasestorage.app/", "")
        blob = self.bucket.blob(blob_path)
        
        # Create temp file
        _, temp_path = tempfile.mkstemp(suffix=os.path.splitext(blob_path)[1])
        
        # Download to temp file
        blob.download_to_filename(temp_path)
        return temp_path
    
    def apply_effects(self, clip: VideoFileClip, effects: Dict[str, Any], is_first: bool = False) -> VideoFileClip:
        """Apply video effects to a clip."""
        # Skip effects for first clip
        if is_first:
            return clip
            
        # Apply color adjustments
        if "color" in effects:
            color = effects["color"]
            if "contrast" in color:
                clip = clip.fx(mp.vfx.colorx, color["contrast"])
            if "brightness" in color:
                clip = clip.fx(mp.vfx.colorx, color["brightness"])
            if "saturation" in color:
                clip = clip.fx(mp.vfx.colorx, color["saturation"])
        
        # Apply filters
        if "filters" in effects:
            for filter_name in effects["filters"]:
                if filter_name == "vibrant":
                    clip = clip.fx(mp.vfx.colorx, 1.2)
                elif filter_name == "warm":
                    clip = clip.fx(mp.vfx.colorx, 1.1)
                elif filter_name == "cinematic":
                    clip = clip.fx(mp.vfx.lum_contrast)
        
        # Apply transitions if specified
        if "transition" in effects:
            transition = effects["transition"]
            if transition["name"] == "fade":
                clip = clip.crossfadein(transition["duration"])
            elif transition["name"] == "slide":
                # TODO: Implement slide transition
                pass
            
        return clip
    
    def create_text_overlay(self, text: str, duration: float, size: int = 30) -> TextClip:
        """Create a text overlay clip."""
        return TextClip(text, fontsize=size, color='white', font='Arial', 
                       size=(900, None), method='caption', align='center',  # Wider text for vertical video
                       stroke_color='black', stroke_width=2)\
               .set_duration(duration)\
               .set_position(('center', 'bottom'))\
               .crossfadein(0.3)\
               .crossfadeout(0.3)
    
    def render_video(self, segments: List[Dict[str, Any]], music: Dict[str, Any], 
                    output_path: Optional[str] = None) -> str:
        """
        Render the final video from segments and music.
        
        Args:
            segments: List of video segments with effects
            music: Music configuration
            output_path: Optional custom output path
            
        Returns:
            Path to the rendered video file
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"video_{uuid.uuid4()}.mp4")
            
        clips = []
        
        # Process each segment
        for segment in segments:
            # Download and create optimized image clip
            image_path = self.download_image(segment["image_url"])
            
            # Create clip with TikTok resolution (1080x1920)
            clip = ImageClip(image_path)
            
            # Calculate dimensions to fill TikTok aspect ratio while maintaining center focus
            img_w, img_h = clip.size
            target_ratio = 9/16  # TikTok aspect ratio
            current_ratio = img_w/img_h
            
            if current_ratio > target_ratio:  # Image is too wide
                # Resize based on height
                new_h = 1920
                new_w = int(img_w * (new_h/img_h))
                clip = clip.resize((new_w, new_h))
                # Crop to width
                x_center = new_w/2
                clip = clip.crop(x1=int(x_center-540), x2=int(x_center+540))  # 1080px wide
            else:  # Image is too tall or perfect
                # Resize based on width
                new_w = 1080
                new_h = int(img_h * (new_w/img_w))
                clip = clip.resize((new_w, new_h))
                # Crop to height if needed
                if new_h > 1920:
                    y_center = new_h/2
                    clip = clip.crop(y1=int(y_center-960), y2=int(y_center+960))  # 1920px tall
            
            # Set duration
            clip = clip.set_duration(segment["duration"])
            
            # Apply effects (pass is_first flag)
            if "effects" in segment:
                is_first = (segment == segments[0])
                clip = self.apply_effects(clip, segment["effects"], is_first)
            
            # Add text overlay
            if "text" in segment:
                text_clip = self.create_text_overlay(segment["text"], segment["duration"])
                clip = CompositeVideoClip([clip, text_clip])
            
            clips.append(clip)
        
        # Concatenate all clips with no transitions
        final_clip = mp.concatenate_videoclips(clips, method="compose")
        
        # Get total duration from segments
        total_duration = sum(segment["duration"] for segment in segments)
        # Ensure final clip duration matches segments
        final_clip = final_clip.set_duration(total_duration)
        
        # Process audio tracks
        audio_tracks = []
        
        # Add voiceovers for each segment
        current_time = 0
        for segment in segments:
            if "voiceover_path" in segment:
                try:
                    voiceover = AudioFileClip(segment["voiceover_path"])
                    # Set the start time for this voiceover
                    voiceover = voiceover.set_start(current_time)
                    audio_tracks.append(voiceover)
                except Exception as e:
                    self.logger.warning(f"Failed to load voiceover: {e}")
            current_time += segment["duration"]
        
        # Add background music if specified
        if music and "path" in music:
            try:
                background_music = AudioFileClip(music["path"])
                # Trim or loop music to match video duration
                total_duration = sum(segment["duration"] for segment in segments)
                if background_music.duration > total_duration:
                    background_music = background_music.subclip(0, total_duration)
                elif background_music.duration < total_duration:
                    # Calculate how many loops we need
                    loops_needed = int(total_duration / background_music.duration) + 1
                    loops = [background_music] * loops_needed
                    background_music = concatenate_audioclips(loops)
                    background_music = background_music.subclip(0, total_duration)
                if music.get("fade_out"):
                    background_music = background_music.audio_fadeout(2)
                # Lower the volume of background music
                background_music = background_music.volumex(0.3)
                audio_tracks.append(background_music)
            except Exception as e:
                self.logger.warning(f"Failed to load background music: {e}")
        
        # Combine all audio tracks if we have any
        if audio_tracks:
            try:
                final_audio = CompositeAudioClip(audio_tracks)
                final_clip = final_clip.set_audio(final_audio)
            except Exception as e:
                self.logger.error(f"Failed to combine audio tracks: {e}")
        
        # Write final video with optimized settings
        # Use more compatible encoding settings
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=30,
            preset='medium',  # Better compatibility
            threads=4,
            bitrate='4000k',
            ffmpeg_params=[
                "-profile:v", "baseline",
                "-level", "3.0",
                "-pix_fmt", "yuv420p",  # Required for QuickTime
                "-movflags", "+faststart"  # Enable streaming
            ]
        )
        
        # Cleanup temp files
        for clip in clips:
            clip.close()
        
        return output_path
```

### ./src/director/tools/music_library_tool.py
```python
from typing import Dict, Any, List, Optional
import random
from ..core.base_tool import BaseTool

class MusicLibraryTool(BaseTool):
    """Tool for music selection and management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.music_library = {
            "luxury": {
                "tracks": [
                    {
                        "id": "lux_001",
                        "name": "Elegant Journey",
                        "duration": 120,
                        "tempo": "moderate",
                        "mood": "sophisticated",
                        "key": "C major"
                    },
                    {
                        "id": "lux_002",
                        "name": "Premium Lifestyle",
                        "duration": 180,
                        "tempo": "slow",
                        "mood": "upscale",
                        "key": "G major"
                    }
                ],
                "transitions": [
                    {"name": "smooth_fade", "duration": 2.0},
                    {"name": "orchestral_swell", "duration": 3.0}
                ]
            },
            "modern": {
                "tracks": [
                    {
                        "id": "mod_001",
                        "name": "Urban Pulse",
                        "duration": 90,
                        "tempo": "fast",
                        "mood": "energetic",
                        "key": "D minor"
                    },
                    {
                        "id": "mod_002",
                        "name": "City Vibes",
                        "duration": 150,
                        "tempo": "moderate",
                        "mood": "trendy",
                        "key": "A minor"
                    }
                ],
                "transitions": [
                    {"name": "quick_cut", "duration": 0.5},
                    {"name": "beat_sync", "duration": 1.0}
                ]
            }
        }
        
    def get_capabilities(self) -> Dict[str, Any]:
        """Get tool capabilities and requirements."""
        return {
            "name": "music_library_tool",
            "description": "Manages music selection and background tracks",
            "supported_styles": list(self.music_library.keys()),
            "features": [
                "track_selection",
                "tempo_matching",
                "mood_analysis",
                "transition_generation"
            ]
        }
        
    def select_track(
        self,
        style: str,
        duration: float,
        mood: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Select an appropriate music track.
        
        Args:
            style: Music style to use
            duration: Target duration
            mood: Optional mood preference
            
        Returns:
            Selected track information
        """
        if style not in self.music_library:
            raise ValueError(f"Unknown music style: {style}")
            
        available_tracks = self.music_library[style]["tracks"]
        
        # Filter by mood if specified
        if mood:
            available_tracks = [
                t for t in available_tracks
                if mood.lower() in t["mood"].lower()
            ]
            
        if not available_tracks:
            raise ValueError(f"No tracks found for style={style}, mood={mood}")
            
        # Select best matching track
        track = random.choice(available_tracks)
        
        return {
            "track": track,
            "start_time": 0.0,
            "end_time": min(duration, track["duration"]),
            "fade_in": 1.0,
            "fade_out": 2.0
        }
        
    def generate_transition(
        self,
        style: str,
        duration: float
    ) -> Dict[str, Any]:
        """
        Generate a music transition.
        
        Args:
            style: Music style to use
            duration: Transition duration
            
        Returns:
            Transition configuration
        """
        if style not in self.music_library:
            raise ValueError(f"Unknown music style: {style}")
            
        transitions = self.music_library[style]["transitions"]
        transition = random.choice(transitions)
        
        return {
            "name": transition["name"],
            "duration": min(duration, transition["duration"]),
            "fade_curve": "smooth"
        }
        
    def analyze_mood(self, property_data: Dict[str, Any]) -> str:
        """Analyze property data to determine appropriate music mood."""
        price = property_data.get("price", 0)
        features = property_data.get("features", [])
        
        if price > 1000000 or "luxury" in " ".join(features).lower():
            return "sophisticated"
        elif "modern" in " ".join(features).lower():
            return "trendy"
        else:
            return "neutral"
```

### ./src/director/core/reasoning_engine.py
```python
from typing import Dict, Any, List, Optional, Callable
import asyncio
from datetime import datetime
import uuid
import os
from src.utils.logging import setup_logger
from ..agents.script_agent import ScriptAgent
from ..agents.pacing_agent import PacingAgent
from ..agents.effect_agent import EffectAgent
from ..tools.video_db_tool import VideoDBTool
from ..tools.video_renderer import VideoRenderer
from ..tools.music_tool import MusicTool
from ..tools.voiceover_tool import VoiceoverTool
from .task import Task

class ReasoningEngine:
    """Core reasoning and workflow orchestration engine."""
    
    def __init__(self):
        self.logger = setup_logger("reasoning_engine")
        self.logger.info("Initializing ReasoningEngine...")
        
        self.script_agent = ScriptAgent()
        self.pacing_agent = PacingAgent()
        self.effect_agent = EffectAgent()
        self.video_db_tool = VideoDBTool()
        self.video_renderer = VideoRenderer()
        self.music_tool = MusicTool()
        self.voiceover_tool = VoiceoverTool()
        
        # Create audio directory if it doesn't exist
        self.audio_dir = os.path.join(os.getcwd(), "assets/audio")
        os.makedirs(os.path.join(self.audio_dir, "voiceover"), exist_ok=True)
        os.makedirs(os.path.join(self.audio_dir, "music"), exist_ok=True)
        
        self.logger.info("All components initialized successfully")
        
        self.workflow_hooks: Dict[str, List[Callable]] = {
            "pre_generation": [],
            "post_script": [],
            "post_effects": [],
            "post_generation": []
        }
        
    async def generate_video(
        self,
        listing_data: Dict[str, Any],
        image_urls: List[str],
        style: str = "modern",
        duration: float = 30.0,
        session_id: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Task:
        """
        Generate a complete real estate video.
        
        Args:
            property_data: Property listing data
            style: Video style to use
            duration: Target video duration
            session_id: Optional session ID for tracking
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary containing video metadata and output paths
        """
        try:
            # Run pre-generation hooks
            await self._run_hooks("pre_generation", listing_data)
            
            # 1. Generate script
            self.logger.info(f"Generating script for property {listing_data.get('id', 'unknown')}")
            if progress_callback:
                progress_callback(0.1, "Generating script")
                
            script = self.script_agent.generate_script(
                property_data=listing_data,
                style=style,
                duration=duration
            )
            
            # Run post-script hooks
            await self._run_hooks("post_script", script)
            
            # 2. Analyze and optimize pacing
            if progress_callback:
                progress_callback(0.2, "Optimizing pacing")
                
            content_analysis = self.pacing_agent.analyze_content(script["segments"])
            pacing = self.pacing_agent.get_optimal_pacing(
                content_analysis=content_analysis,
                target_duration=duration,
                style=style
            )
            
            segments = self.pacing_agent.apply_pacing(
                segments=script["segments"],
                pacing=pacing,
                target_duration=duration
            )
            
            # 3. Generate effects and transitions
            if progress_callback:
                progress_callback(0.3, "Generating effects")
                
            # Get base effects for the style
            base_effects = self.video_db_tool.get_effects(
                template_name=f"{style}_standard",
                intensity=0.8  # Slightly reduced intensity for better results
            )
            
            # Get transitions (one less than segments since we don't need transition for first)
            transitions = self.video_db_tool.get_transitions(
                template_name=f"{style}_standard",
                count=len(segments) - 1,
                variation=0.2
            )
            
            # Process each segment with effects and transitions
            processed_segments = []
            for i, segment in enumerate(segments):
                # Don't add transition for first segment
                transition = None if i == 0 else transitions[i-1]
                
                # Process segment
                processed = self.video_db_tool.process_segment(
                    segment=segment,
                    effects=base_effects,
                    transition=transition
                )
                processed_segments.append(processed)
            
            # Replace segments with processed ones
            segments = processed_segments
                
            # Run post-effects hooks
            await self._run_hooks("post_effects", segments)
            
            # 4. Generate voiceovers for each segment
            if progress_callback:
                progress_callback(0.4, "Generating voiceover")
                
            for segment in segments:
                if "text" in segment:
                    voiceover = self.voiceover_tool.generate_voiceover(
                        text=segment["text"],
                        style=style,
                        output_format="mp3"
                    )
                    segment["voiceover_path"] = voiceover["file_path"]
            
            # 5. Select music
            if progress_callback:
                progress_callback(0.5, "Selecting music")
                
            music = self.music_tool.select_music(
                template_name=f"{style}_balanced_standard",
                duration=duration
            )
            
            # 5. Process video
            if progress_callback:
                progress_callback(0.5, "Processing video")
                
            # 6. Finalize video
            if progress_callback:
                progress_callback(0.9, "Finalizing video")
                
            result = {
                "session_id": session_id,
                "style": style,
                "duration": duration,
                "segments": segments,  # Use processed segments
                "music": music,
                "generated_at": datetime.utcnow().isoformat(),
                "property_id": listing_data.get("id")
            }
            
            # Add image URLs to segments
            for i, segment in enumerate(segments):  # Use processed segments
                # Cycle through images if we have more segments than images
                image_index = i % len(image_urls)
                segment["image_url"] = image_urls[image_index]
            
            # Render final video
            if progress_callback:
                progress_callback(0.9, "Rendering video")
                
            output_path = self.video_renderer.render_video(
                segments=segments,  # Use processed segments
                music=music
            )
            
            result["output_path"] = output_path
            
            # Run post-generation hooks
            await self._run_hooks("post_generation", result)
            
            if progress_callback:
                progress_callback(1.0, "Complete")
                
            # Create and complete task
            task = Task(session_id or str(uuid.uuid4()))
            task.complete(result)
            return task
            
        except Exception as e:
            self.logger.error(f"Video generation failed: {str(e)}")
            # Create failed task
            task = Task(session_id or str(uuid.uuid4()))
            task.fail(str(e))
            return task
            
    def register_hook(
        self,
        hook_type: str,
        callback: Callable
    ) -> None:
        """Register a workflow hook."""
        if hook_type not in self.workflow_hooks:
            raise ValueError(f"Invalid hook type: {hook_type}")
        self.workflow_hooks[hook_type].append(callback)
        
    async def _run_hooks(
        self,
        hook_type: str,
        data: Any
    ) -> None:
        """Run all registered hooks of a given type."""
        for hook in self.workflow_hooks[hook_type]:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(data)
                else:
                    hook(data)
            except Exception as e:
                logger.error(f"Hook execution failed: {str(e)}")
                # Continue with other hooks
```

### ./src/director/core/task.py
```python
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

class Task:
    """Represents a video generation task."""
    
    def __init__(self, session_id: str):
        """Initialize a new task.
        
        Args:
            session_id: Associated session ID
        """
        self.id = str(uuid.uuid4())
        self.session_id = session_id
        self.status = "processing"
        self.progress = 0.0
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.created_at = datetime.utcnow()
        self.updated_at = self.created_at
        
    def update_progress(self, progress: float, status: Optional[str] = None):
        """Update task progress.
        
        Args:
            progress: Progress value between 0 and 1
            status: Optional new status
        """
        self.progress = max(0.0, min(1.0, progress))
        if status:
            self.status = status
        self.updated_at = datetime.utcnow()
        
    def complete(self, result: Dict[str, Any]):
        """Mark task as complete with result.
        
        Args:
            result: Task result data
        """
        self.status = "completed"
        self.progress = 1.0
        self.result = result
        self.updated_at = datetime.utcnow()
        
    def fail(self, error: str):
        """Mark task as failed with error.
        
        Args:
            error: Error message
        """
        self.status = "failed"
        self.error = error
        self.updated_at = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
```

### ./src/director/core/config.py
```python
from typing import Dict, Any, Optional
import os
from pathlib import Path
import yaml

class DirectorConfig:
    """Configuration manager for Director integration."""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path('config/director')
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.configs: Dict[str, Any] = {}
        
    def load_config(self, name: str) -> Dict[Any, Any]:
        """Load a specific configuration file."""
        if name in self.configs:
            return self.configs[name]
            
        config_path = self.config_dir / f"{name}.yaml"
        if not config_path.exists():
            return {}
            
        with open(config_path, 'r') as f:
            self.configs[name] = yaml.safe_load(f)
            return self.configs[name]
            
    def save_config(self, name: str, config: Dict[Any, Any]) -> None:
        """Save a configuration to file."""
        config_path = self.config_dir / f"{name}.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        self.configs[name] = config
        
    def get_value(self, name: str, key: str, default: Any = None) -> Any:
        """Get a specific configuration value."""
        config = self.load_config(name)
        return config.get(key, default)
        
    def set_value(self, name: str, key: str, value: Any) -> None:
        """Set a specific configuration value."""
        config = self.load_config(name)
        config[key] = value
        self.save_config(name, config)
```

### ./src/director/core/base_agent.py
```python
from typing import Dict, Any

class BaseAgent:
    """Base class for all Director agents."""
    
    def __init__(self):
        """Initialize the base agent."""
        pass
        
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate agent configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if configuration is valid
        """
        return True
```

### ./src/director/core/session_manager.py
```python
from typing import Dict, Any, Optional
import uuid
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager
from src.utils.logging import setup_logger

class SessionManager:
    """Manages video generation sessions and state tracking."""
    
    def __init__(self):
        self.logger = setup_logger("session_manager")
        self.logger.info("Initializing SessionManager...")
        
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.cleanup_interval = timedelta(hours=1)
        self._cleanup_task = None
        
        self.logger.info("SessionManager initialized successfully")
        
    async def start(self):
        """Start the session manager and cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop the session manager and cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            
    @asynccontextmanager
    async def session(self, session_id: Optional[str] = None) -> str:
        """
        Context manager for video generation sessions.
        
        Args:
            session_id: Optional session ID to use
            
        Returns:
            Session ID string
        """
        session_id = session_id or str(uuid.uuid4())
        try:
            await self.create_session(session_id)
            yield session_id
        finally:
            await self.end_session(session_id)
            
    async def create_session(self, session_id: str) -> None:
        """Create a new video generation session."""
        self.logger.info(f"Creating new session with ID: {session_id}")
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")
            
        self.sessions[session_id] = {
            "created_at": datetime.utcnow(),
            "status": "initializing",
            "progress": 0.0,
            "message": "Session created",
            "artifacts": {},
            "errors": []
        }
        
    async def end_session(self, session_id: str) -> None:
        """End a video generation session."""
        if session_id in self.sessions:
            self.sessions[session_id]["status"] = "completed"
            self.sessions[session_id]["ended_at"] = datetime.utcnow()
            
    async def update_progress(
        self,
        session_id: str,
        progress: float,
        message: Optional[str] = None
    ) -> None:
        """Update session progress."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        self.sessions[session_id].update({
            "progress": progress,
            "message": message or self.sessions[session_id]["message"],
            "updated_at": datetime.utcnow()
        })
        
    async def add_artifact(
        self,
        session_id: str,
        artifact_type: str,
        artifact_data: Any
    ) -> None:
        """Add an artifact to the session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        self.sessions[session_id]["artifacts"][artifact_type] = {
            "data": artifact_data,
            "created_at": datetime.utcnow()
        }
        
    async def add_error(
        self,
        session_id: str,
        error: Exception
    ) -> None:
        """Add an error to the session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        self.sessions[session_id]["errors"].append({
            "error": str(error),
            "type": type(error).__name__,
            "timestamp": datetime.utcnow()
        })
        
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session information."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        return self.sessions[session_id]
        
    async def _cleanup_loop(self):
        """Background task to clean up old sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval.total_seconds())
                await self._cleanup_old_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session cleanup failed: {str(e)}")
                
    async def _cleanup_old_sessions(self):
        """Clean up completed sessions older than cleanup_interval."""
        now = datetime.utcnow()
        to_remove = []
        
        for session_id, session in self.sessions.items():
            if (
                session["status"] == "completed"
                and now - session["ended_at"] > self.cleanup_interval
            ):
                to_remove.append(session_id)
                
        for session_id in to_remove:
            del self.sessions[session_id]
```

### ./src/director/core/session.py
```python
from typing import Dict, Optional, Any
import uuid
from datetime import datetime

class Session:
    """Represents a video generation session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.utcnow()
        self.status = "initialized"
        self.data: Dict[str, Any] = {}
        self.progress = 0
        self.error: Optional[str] = None
        
    def update_progress(self, progress: int, status: Optional[str] = None) -> None:
        """Update session progress."""
        self.progress = progress
        if status:
            self.status = status
            
    def set_error(self, error: str) -> None:
        """Set error state for session."""
        self.status = "error"
        self.error = error
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "data": self.data
        }

class SessionManager:
    """Manages video generation sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        
    def create_session(self) -> Session:
        """Create a new session."""
        session_id = str(uuid.uuid4())
        session = Session(session_id)
        self.sessions[session_id] = session
        return session
        
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        return self.sessions.get(session_id)
        
    def update_session(self, session_id: str, progress: int, status: Optional[str] = None) -> None:
        """Update session progress."""
        session = self.get_session(session_id)
        if session:
            session.update_progress(progress, status)
            
    def remove_session(self, session_id: str) -> None:
        """Remove a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            
    def cleanup_old_sessions(self, max_age_hours: int = 24) -> None:
        """Remove sessions older than max_age_hours."""
        now = datetime.utcnow()
        to_remove = []
        
        for session_id, session in self.sessions.items():
            age = now - session.created_at
            if age.total_seconds() > max_age_hours * 3600:
                to_remove.append(session_id)
                
        for session_id in to_remove:
            self.remove_session(session_id)
```

### ./src/director/core/__init__.py
```python
```

### ./src/director/core/base_tool.py
```python
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Base class for all Director tools."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the base tool with optional configuration."""
        self.config = config or {}
        self.validate_config()
        
    def validate_config(self) -> None:
        """Validate tool configuration."""
        pass
        
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get tool capabilities and requirements."""
        pass
        
    def cleanup(self) -> None:
        """Clean up any resources used by the tool."""
        pass
```

### ./src/director/core/engine.py
```python
from typing import Dict, Any, Optional
import os
from pathlib import Path
import yaml

class ReasoningEngine:
    """
    Wrapper for Director's reasoning engine, customized for real estate video generation.
    """
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.session_manager = None  # Will be initialized with SessionManager
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[Any, Any]:
        """Load engine configuration from YAML file."""
        if not config_path:
            config_path = os.getenv('DIRECTOR_CONFIG_PATH', 'config/director/engine.yaml')
        
        config_path = Path(config_path)
        if not config_path.exists():
            return self._create_default_config(config_path)
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self, config_path: Path) -> Dict[Any, Any]:
        """Create default configuration if none exists."""
        default_config = {
            'llm': {
                'model': 'gpt-4',
                'temperature': 0.7,
                'max_tokens': 2000
            },
            'video': {
                'max_duration': 60,  # seconds
                'resolution': '1080p',
                'fps': 30
            },
            'agents': {
                'script_generator': {
                    'enabled': True,
                    'style': 'tiktok'
                },
                'voice_generator': {
                    'enabled': True,
                    'provider': 'elevenlabs'
                },
                'music_selector': {
                    'enabled': True,
                    'style': 'upbeat'
                }
            }
        }
        
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save default config
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
            
        return default_config
    
    def initialize(self, session_manager: 'SessionManager') -> None:
        """Initialize the engine with a session manager."""
        self.session_manager = session_manager
        
    async def process_request(self, session_id: str, request_data: Dict[Any, Any]) -> Dict[Any, Any]:
        """
        Process a video generation request.
        
        Args:
            session_id: Unique session identifier
            request_data: Video generation parameters
            
        Returns:
            Dict containing processing results
        """
        if not self.session_manager:
            raise RuntimeError("Engine not initialized with SessionManager")
            
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"No session found for ID: {session_id}")
            
        # TODO: Implement actual processing logic using Director's components
        return {
            "status": "processing",
            "session_id": session_id,
            "message": "Video generation started"
        }
```

### ./src/director/variation/style_engine.py
```python
from typing import Dict, Any, List, Optional, Tuple
import random
import numpy as np
from dataclasses import dataclass

@dataclass
class StyleComponent:
    """Component of a video style."""
    name: str
    weight: float
    attributes: Dict[str, Any]

class StyleCombination:
    """Combination of multiple style components."""
    
    def __init__(self, components: List[StyleComponent]):
        self.components = components
        self._normalize_weights()
        
    def _normalize_weights(self) -> None:
        """Normalize component weights to sum to 1."""
        total_weight = sum(c.weight for c in self.components)
        if total_weight > 0:
            for component in self.components:
                component.weight /= total_weight
                
    def get_weighted_value(self, attribute: str) -> Any:
        """Get weighted average of an attribute across components."""
        values = []
        weights = []
        
        for component in self.components:
            if attribute in component.attributes:
                values.append(component.attributes[attribute])
                weights.append(component.weight)
                
        if not values:
            return None
            
        # Handle different value types
        if isinstance(values[0], (int, float)):
            return np.average(values, weights=weights)
        elif isinstance(values[0], str):
            # For strings, return the value with highest weight
            max_weight_idx = np.argmax(weights)
            return values[max_weight_idx]
        elif isinstance(values[0], dict):
            # Merge dictionaries with weights
            result = {}
            for value, weight in zip(values, weights):
                for k, v in value.items():
                    if k not in result:
                        result[k] = v if isinstance(v, str) else 0
                    if isinstance(v, (int, float)):
                        if isinstance(result[k], (int, float)):
                            result[k] += v * weight
                    else:
                        # For non-numeric values (like colors), use weighted selection
                        if weight > result.get(f'{k}_weight', 0):
                            result[k] = v
                            result[f'{k}_weight'] = weight
            
            # Clean up temporary weight keys
            for k in list(result.keys()):
                if k.endswith('_weight'):
                    del result[k]
            return result
            
        return values[0]

class StyleEngine:
    """Engine for combining and varying video styles."""
    
    def __init__(self):
        self.base_styles = self._create_base_styles()
        
    def _create_base_styles(self) -> Dict[str, Dict[str, Any]]:
        """Create base video styles."""
        return {
            'modern': {
                'color_scheme': {
                    'primary': '#FF5733',
                    'secondary': '#33FF57',
                    'text': '#FFFFFF'
                },
                'transitions': {
                    'types': ['slide_left', 'slide_right', 'zoom_in'],
                    'duration': 0.5
                },
                'effects': {
                    'color_enhance': 0.7,
                    'sharpness': 0.5
                },
                'text': {
                    'font': 'Roboto',
                    'animation': 'slide'
                }
            },
            'luxury': {
                'color_scheme': {
                    'primary': '#FFD700',
                    'secondary': '#000000',
                    'text': '#FFFFFF'
                },
                'transitions': {
                    'types': ['dissolve', 'zoom_out'],
                    'duration': 1.0
                },
                'effects': {
                    'vignette': 0.3,
                    'film_grain': 0.2
                },
                'text': {
                    'font': 'Playfair Display',
                    'animation': 'fade'
                }
            },
            'minimalist': {
                'color_scheme': {
                    'primary': '#FFFFFF',
                    'secondary': '#000000',
                    'text': '#000000'
                },
                'transitions': {
                    'types': ['dissolve'],
                    'duration': 0.7
                },
                'effects': {
                    'sharpness': 0.3
                },
                'text': {
                    'font': 'Helvetica',
                    'animation': 'fade'
                }
            }
        }
        
    def create_style_combination(self,
                               styles: List[Tuple[str, float]],
                               variation: float = 0.2) -> StyleCombination:
        """
        Create combination of styles with weights.
        
        Args:
            styles: List of (style_name, weight) tuples
            variation: Amount of random variation (0-1)
        """
        components = []
        
        for style_name, weight in styles:
            if style_name not in self.base_styles:
                raise ValueError(f"Unknown style: {style_name}")
                
            # Deep copy attributes with variation
            attributes = {}
            base_attrs = self.base_styles[style_name]
            
            for key, value in base_attrs.items():
                if isinstance(value, (int, float)):
                    # Add random variation to numeric values
                    variation_amount = value * variation
                    attributes[key] = value + random.uniform(-variation_amount, variation_amount)
                elif isinstance(value, dict):
                    # Recursively handle nested dictionaries
                    attributes[key] = {}
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            variation_amount = v * variation
                            attributes[key][k] = v + random.uniform(-variation_amount, variation_amount)
                        else:
                            attributes[key][k] = v
                else:
                    attributes[key] = value
                    
            components.append(StyleComponent(style_name, weight, attributes))
            
        return StyleCombination(components)
        
    def interpolate_styles(self,
                          style1: StyleCombination,
                          style2: StyleCombination,
                          t: float) -> StyleCombination:
        """
        Interpolate between two style combinations.
        
        Args:
            style1: First style combination
            style2: Second style combination
            t: Interpolation factor (0-1)
        """
        # Combine components from both styles
        components = []
        
        # Add components from style1 with adjusted weights
        for comp in style1.components:
            components.append(StyleComponent(
                comp.name,
                comp.weight * (1 - t),
                comp.attributes
            ))
            
        # Add components from style2 with adjusted weights
        for comp in style2.components:
            components.append(StyleComponent(
                comp.name,
                comp.weight * t,
                comp.attributes
            ))
            
        return StyleCombination(components)
```

### ./src/director/variation/template_engine.py
```python
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import random

class VideoTemplate:
    """Template for video structure and styling."""
    
    def __init__(self, config: Dict[str, Any]):
        self.name = config.get('name', 'default')
        self.duration = config.get('duration', 60)
        self.segments = config.get('segments', [])
        self.transitions = config.get('transitions', [])
        self.effects = config.get('effects', [])
        self.text_styles = config.get('text_styles', {})
        self.music_style = config.get('music_style', {})
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert template to dictionary."""
        return {
            'name': self.name,
            'duration': self.duration,
            'segments': self.segments,
            'transitions': self.transitions,
            'effects': self.effects,
            'text_styles': self.text_styles,
            'music_style': self.music_style
        }

class TemplateLibrary:
    """Library of video templates."""
    
    def __init__(self, template_path: Optional[str] = None):
        self.templates = self._load_templates(template_path)
        
    def _load_templates(self, template_path: Optional[str]) -> Dict[str, VideoTemplate]:
        """Load templates from JSON file."""
        if not template_path:
            return self._default_templates()
            
        template_path = Path(template_path)
        if template_path.exists():
            with open(template_path, 'r') as f:
                templates_data = json.load(f)
                return {
                    name: VideoTemplate(data)
                    for name, data in templates_data.items()
                }
                
        return self._default_templates()
        
    def _default_templates(self) -> Dict[str, VideoTemplate]:
        """Default video templates with extensive variations."""
        templates = {}
        
        # Base styles
        styles = {
            'modern': {
                'transitions': ['slide_left', 'slide_right', 'zoom_in', 'whip_pan'],
                'effects': ['color_enhance', 'sharpness'],
                'text_animations': ['pop', 'slide'],
                'music_genres': ['electronic', 'pop', 'upbeat'],
                'color_schemes': ['vibrant', 'neon', 'pastel']
            },
            'luxury': {
                'transitions': ['dissolve', 'zoom_out', 'fade'],
                'effects': ['cinematic_bars', 'film_grain', 'vignette'],
                'text_animations': ['fade', 'elegant_slide'],
                'music_genres': ['ambient', 'classical', 'jazz'],
                'color_schemes': ['gold', 'monochrome', 'earth']
            },
            'minimal': {
                'transitions': ['cut', 'dissolve', 'slide_minimal'],
                'effects': ['subtle_sharpen', 'light_contrast'],
                'text_animations': ['fade', 'minimal_pop'],
                'music_genres': ['minimal', 'acoustic', 'soft'],
                'color_schemes': ['clean', 'subtle', 'neutral']
            },
            'urban': {
                'transitions': ['glitch', 'swipe', 'bounce'],
                'effects': ['rgb_split', 'grain', 'contrast'],
                'text_animations': ['glitch', 'bounce', 'swipe'],
                'music_genres': ['hip_hop', 'electronic', 'urban'],
                'color_schemes': ['street', 'neon', 'bold']
            },
            'cinematic': {
                'transitions': ['cinematic_fade', 'letterbox', 'pan'],
                'effects': ['anamorphic', 'film_look', 'lens_blur'],
                'text_animations': ['cinematic_fade', 'smooth_reveal'],
                'music_genres': ['orchestral', 'soundtrack', 'epic'],
                'color_schemes': ['film', 'dramatic', 'moody']
            }
        }
        
        # Pacing variations
        pacings = {
            'rapid': {'segment_multiplier': 0.7, 'transition_duration': 0.3},
            'balanced': {'segment_multiplier': 1.0, 'transition_duration': 0.7},
            'relaxed': {'segment_multiplier': 1.3, 'transition_duration': 1.0}
        }
        
        # Segment structures
        structures = {
            'standard': [
                {'type': 'intro', 'duration': 5},
                {'type': 'features', 'duration': 30},
                {'type': 'highlights', 'duration': 20},
                {'type': 'outro', 'duration': 5}
            ],
            'story': [
                {'type': 'hook', 'duration': 3},
                {'type': 'context', 'duration': 12},
                {'type': 'reveal', 'duration': 35},
                {'type': 'close', 'duration': 10}
            ],
            'showcase': [
                {'type': 'aerial', 'duration': 10},
                {'type': 'exterior', 'duration': 15},
                {'type': 'interior', 'duration': 25},
                {'type': 'details', 'duration': 10}
            ],
            'feature_focus': [
                {'type': 'intro', 'duration': 5},
                {'type': 'feature_1', 'duration': 20},
                {'type': 'feature_2', 'duration': 20},
                {'type': 'feature_3', 'duration': 15}
            ],
            'quick': [
                {'type': 'hook', 'duration': 3},
                {'type': 'highlights', 'duration': 42},
                {'type': 'call_to_action', 'duration': 15}
            ]
        }
        
        # Generate templates by combining styles, pacings, and structures
        for style_name, style in styles.items():
            for pacing_name, pacing in pacings.items():
                for structure_name, base_segments in structures.items():
                    # Create variations of segments with pacing
                    segments = []
                    for segment in base_segments:
                        segments.append({
                            'type': segment['type'],
                            'duration': segment['duration'] * pacing['segment_multiplier']
                        })
                    
                    # Create template variations with different combinations
                    for color_scheme in style['color_schemes']:
                        # Ensure consistent naming format
                        safe_structure_name = structure_name.replace('_', '')
                        template_name = f'{style_name}_{pacing_name}_{safe_structure_name}_{color_scheme}'
                        
                        # Select random variations while maintaining style consistency
                        import random
                        transitions = [
                            {'type': t, 'duration': pacing['transition_duration']}
                            for t in random.sample(style['transitions'], 
                                                min(3, len(style['transitions'])))
                        ]
                        
                        effects = [
                            {'name': e, 'intensity': random.uniform(0.3, 0.8)}
                            for e in random.sample(style['effects'],
                                                min(2, len(style['effects'])))
                        ]
                        
                        templates[template_name] = VideoTemplate({
                            'name': template_name,
                            'duration': 60,
                            'segments': segments,
                            'transitions': transitions,
                            'effects': effects,
                            'text_styles': {
                                'title': {
                                    'font_scale': 1.2,
                                    'animation': random.choice(style['text_animations'])
                                },
                                'feature': {
                                    'font_scale': 1.0,
                                    'animation': random.choice(style['text_animations'])
                                }
                            },
                            'music_style': {
                                'genre': random.choice(style['music_genres']),
                                'tempo': pacing_name
                            }
                        })
        
        return templates
        
    def get_template(self, name: str) -> VideoTemplate:
        """Get template by name."""
        if name not in self.templates:
            raise ValueError(f"Unknown template: {name}")
        return self.templates[name]
        
    def list_templates(self) -> List[str]:
        """List available templates."""
        return list(self.templates.keys())
        
    def get_template_info(self, name: str) -> Dict[str, Any]:
        """Get detailed information about a template."""
        template = self.get_template(name)
        return template.to_dict()
        
    def save_templates(self, path: str) -> None:
        """Save templates to JSON file."""
        templates_data = {
            name: template.to_dict()
            for name, template in self.templates.items()
        }
        
        with open(path, 'w') as f:
            json.dump(templates_data, f, indent=2)
            
    def add_template(self, name: str, config: Dict[str, Any]) -> None:
        """Add new template to library."""
        self.templates[name] = VideoTemplate(config)
```

### ./src/director/variation/__init__.py
```python
```

### ./src/director/variation/pacing_engine.py
```python
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class PacingPoint:
    """Represents a point in the video with specific pacing."""
    time: float
    intensity: float
    duration: float

class PacingCurve:
    """Generates dynamic pacing curves for video timing."""
    
    def __init__(self, duration: float, base_intensity: float = 0.5):
        self.duration = duration
        self.base_intensity = base_intensity
        self.points: List[PacingPoint] = []
        
    def add_point(self, time: float, intensity: float, duration: float) -> None:
        """Add a pacing point."""
        self.points.append(PacingPoint(time, intensity, duration))
        # Sort points by time
        self.points.sort(key=lambda p: p.time)
        
    def get_intensity(self, time: float) -> float:
        """Get intensity at specific time."""
        if not self.points:
            return self.base_intensity
            
        # Find surrounding points
        prev_point = None
        next_point = None
        
        for point in self.points:
            if point.time <= time:
                prev_point = point
            else:
                next_point = point
                break
                
        if not prev_point:
            return self.points[0].intensity
        if not next_point:
            return prev_point.intensity
            
        # Interpolate between points
        t = (time - prev_point.time) / (next_point.time - prev_point.time)
        return prev_point.intensity * (1 - t) + next_point.intensity * t

class PacingEngine:
    """Engine for managing video pacing."""
    
    def __init__(self):
        self.pacing_presets = self._create_presets()
        
    def _create_presets(self) -> Dict[str, Dict[str, Any]]:
        """Create pacing presets."""
        return {
            'energetic': {
                'base_intensity': 0.7,
                'points': [
                    {'time': 0, 'intensity': 0.8, 'duration': 3},
                    {'time': 15, 'intensity': 0.9, 'duration': 2},
                    {'time': 30, 'intensity': 1.0, 'duration': 1},
                    {'time': 45, 'intensity': 0.9, 'duration': 2}
                ],
                'transition_frequency': 'high',
                'effect_intensity': 'high'
            },
            'balanced': {
                'base_intensity': 0.5,
                'points': [
                    {'time': 0, 'intensity': 0.6, 'duration': 4},
                    {'time': 20, 'intensity': 0.7, 'duration': 3},
                    {'time': 40, 'intensity': 0.6, 'duration': 3}
                ],
                'transition_frequency': 'medium',
                'effect_intensity': 'medium'
            },
            'cinematic': {
                'base_intensity': 0.3,
                'points': [
                    {'time': 0, 'intensity': 0.4, 'duration': 5},
                    {'time': 25, 'intensity': 0.5, 'duration': 4},
                    {'time': 50, 'intensity': 0.4, 'duration': 4}
                ],
                'transition_frequency': 'low',
                'effect_intensity': 'low'
            }
        }
        
    def create_pacing_curve(self, 
                          preset: str,
                          duration: float,
                          variation: float = 0.2) -> PacingCurve:
        """
        Create pacing curve from preset.
        
        Args:
            preset: Preset name
            duration: Video duration
            variation: Amount of random variation (0-1)
        """
        if preset not in self.pacing_presets:
            raise ValueError(f"Unknown preset: {preset}")
            
        preset_data = self.pacing_presets[preset]
        base_intensity = preset_data['base_intensity']
        
        # Add variation to base intensity
        if variation > 0:
            base_intensity += np.random.uniform(-variation, variation)
            base_intensity = max(0.1, min(1.0, base_intensity))
            
        curve = PacingCurve(duration, base_intensity)
        
        # Add points with variation
        for point in preset_data['points']:
            time_ratio = point['time'] / 60  # Original points based on 60s
            time = time_ratio * duration
            
            # Add random variation to each point
            intensity = point['intensity']
            if variation > 0:
                variation_scale = np.random.uniform(1 - variation, 1 + variation)
                intensity *= variation_scale
                intensity = max(0.1, min(1.0, intensity))
                
            curve.add_point(time, intensity, point['duration'])
            
        return curve
        
    def get_segment_duration(self,
                           base_duration: float,
                           intensity: float) -> float:
        """
        Calculate segment duration based on intensity.
        
        Args:
            base_duration: Base segment duration
            intensity: Current pacing intensity
            
        Returns:
            Adjusted duration
        """
        # Higher intensity = shorter duration
        return base_duration * (1.5 - intensity)
        
    def get_transition_duration(self,
                              base_duration: float,
                              intensity: float) -> float:
        """
        Calculate transition duration based on intensity.
        
        Args:
            base_duration: Base transition duration
            intensity: Current pacing intensity
            
        Returns:
            Adjusted duration
        """
        # Higher intensity = faster transitions
        return base_duration * (2 - intensity)
        
    def get_effect_intensity(self,
                           base_intensity: float,
                           pacing_intensity: float) -> float:
        """
        Calculate effect intensity based on pacing.
        
        Args:
            base_intensity: Base effect intensity
            pacing_intensity: Current pacing intensity
            
        Returns:
            Adjusted intensity
        """
        # Higher pacing = stronger effects
        return base_intensity * (0.5 + pacing_intensity * 0.5)
```

### ./src/director/__init__.py
```python
```

### ./src/director/agents/pacing_agent.py
```python
from typing import Dict, Any, List, Optional
import numpy as np
from ..core.base_agent import BaseAgent

class PacingAgent(BaseAgent):
    """Agent for adapting video pacing based on content and engagement."""
    
    def __init__(self):
        self.pacing_profiles = {
            "rapid": {
                "segment_duration": 2.5,
                "transition_duration": 0.3,
                "text_duration": 2.0,
                "music_bpm_range": (120, 140)
            },
            "balanced": {
                "segment_duration": 3.5,
                "transition_duration": 0.5,
                "text_duration": 3.0,
                "music_bpm_range": (100, 120)
            },
            "relaxed": {
                "segment_duration": 4.5,
                "transition_duration": 0.7,
                "text_duration": 4.0,
                "music_bpm_range": (80, 100)
            }
        }
        
        self.content_weights = {
            "exterior": 1.0,
            "interior": 1.2,
            "feature": 1.5,
            "amenity": 1.3,
            "view": 1.4
        }
        
    def analyze_content(
        self,
        segments: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Analyze content to determine optimal pacing.
        
        Args:
            segments: List of video segments with content type
            
        Returns:
            Dictionary with content analysis metrics
        """
        analysis = {
            "complexity": 0.0,
            "visual_interest": 0.0,
            "information_density": 0.0
        }
        
        for segment in segments:
            content_type = segment.get("content_type", "interior")
            weight = self.content_weights.get(content_type, 1.0)
            
            # Calculate complexity based on content
            if "text" in segment:
                analysis["complexity"] += len(segment["text"].split()) * 0.1 * weight
            
            # Calculate visual interest
            if "features" in segment:
                analysis["visual_interest"] += len(segment["features"]) * 0.2 * weight
                
            # Calculate information density
            if "data_points" in segment:
                analysis["information_density"] += len(segment["data_points"]) * 0.15 * weight
                
        # Normalize metrics
        total_segments = len(segments)
        for key in analysis:
            analysis[key] /= total_segments
            
        return analysis
        
    def get_optimal_pacing(
        self,
        content_analysis: Dict[str, float],
        target_duration: float,
        style: str = "modern"
    ) -> Dict[str, Any]:
        """
        Determine optimal pacing based on content analysis.
        
        Args:
            content_analysis: Content analysis metrics
            target_duration: Target video duration
            style: Video style
            
        Returns:
            Pacing configuration
        """
        # Calculate base pacing profile
        complexity_score = content_analysis["complexity"]
        if complexity_score > 0.7:
            base_profile = "relaxed"
        elif complexity_score < 0.3:
            base_profile = "rapid"
        else:
            base_profile = "balanced"
            
        profile = self.pacing_profiles[base_profile].copy()
        
        # Adjust for style
        if style == "luxury":
            profile["segment_duration"] *= 1.2
            profile["transition_duration"] *= 1.3
        elif style == "minimal":
            profile["segment_duration"] *= 0.9
            profile["transition_duration"] *= 0.8
            
        # Adjust for content metrics
        visual_score = content_analysis["visual_interest"]
        info_score = content_analysis["information_density"]
        
        profile["segment_duration"] *= 1 + (info_score - 0.5) * 0.3
        profile["transition_duration"] *= 1 + (visual_score - 0.5) * 0.2
        
        return profile
        
    def apply_pacing(
        self,
        segments: List[Dict[str, Any]],
        pacing: Dict[str, Any],
        target_duration: float
    ) -> List[Dict[str, Any]]:
        """
        Apply pacing configuration to video segments.
        
        Args:
            segments: List of video segments
            pacing: Pacing configuration
            target_duration: Target video duration
            
        Returns:
            List of segments with adjusted timing
        """
        adjusted_segments = []
        current_time = 0.0
        
        # Calculate initial durations
        base_durations = []
        for segment in segments:
            content_type = segment.get("content_type", "interior")
            weight = self.content_weights.get(content_type, 1.0)
            duration = pacing["segment_duration"] * weight
            base_durations.append(duration)
            
        # Scale durations to match target
        total_duration = sum(base_durations)
        scale_factor = (target_duration - 
                       (len(segments) - 1) * pacing["transition_duration"]) / total_duration
        
        # Apply scaled durations
        for segment, base_duration in zip(segments, base_durations):
            adjusted_segment = segment.copy()
            adjusted_segment["duration"] = base_duration * scale_factor
            adjusted_segment["start_time"] = current_time
            
            # Add transition if not last segment
            if len(adjusted_segments) < len(segments) - 1:
                adjusted_segment["transition_duration"] = pacing["transition_duration"]
                current_time += adjusted_segment["duration"] + pacing["transition_duration"]
            else:
                current_time += adjusted_segment["duration"]
                
            adjusted_segments.append(adjusted_segment)
            
        return adjusted_segments
        
    def optimize_engagement(
        self,
        segments: List[Dict[str, Any]],
        engagement_data: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Optimize pacing based on engagement data.
        
        Args:
            segments: List of video segments
            engagement_data: Optional engagement metrics
            
        Returns:
            Optimized segments
        """
        if not engagement_data:
            return segments
            
        optimized = []
        total_duration = sum(s["duration"] for s in segments)
        
        # Calculate engagement-based duration adjustments
        for segment in segments:
            adjusted = segment.copy()
            content_type = segment.get("content_type", "interior")
            
            if content_type in engagement_data:
                # Adjust duration based on engagement (±25% max)
                engagement_score = engagement_data[content_type]
                adjustment = np.clip(engagement_score - 0.5, -0.25, 0.25)
                adjusted["duration"] *= (1 + adjustment)
                
            optimized.append(adjusted)
            
        # Normalize durations
        current_total = sum(s["duration"] for s in optimized)
        scale_factor = total_duration / current_total
        
        current_time = 0.0
        for segment in optimized:
            segment["duration"] *= scale_factor
            segment["start_time"] = current_time
            current_time += segment["duration"]
            
        return optimized
```

### ./src/director/agents/music_agent.py
```python
from typing import Dict, Any, List, Optional
import os
from pathlib import Path
import json
import random

class MusicAgent:
    """
    Agent for selecting and managing background music.
    """
    
    def __init__(self, music_dir: Optional[str] = None):
        self.music_dir = Path(music_dir or 'assets/audio/music')
        self.music_dir.mkdir(parents=True, exist_ok=True)
        self._load_music_catalog()
        
    def _load_music_catalog(self) -> None:
        """Load music catalog from JSON file or create default."""
        catalog_file = self.music_dir / 'catalog.json'
        
        if catalog_file.exists():
            with open(catalog_file, 'r') as f:
                self.catalog = json.load(f)
        else:
            self.catalog = self._create_default_catalog()
            with open(catalog_file, 'w') as f:
                json.dump(self.catalog, f, indent=2)
                
    def _create_default_catalog(self) -> Dict[str, Any]:
        """Create default music catalog."""
        return {
            'genres': {
                'upbeat_electronic': {
                    'mood': 'energetic',
                    'tempo': 'fast',
                    'tracks': [
                        {
                            'title': 'Modern Momentum',
                            'duration': 60,
                            'file': 'modern_momentum.mp3',
                            'tags': ['modern', 'upbeat', 'electronic']
                        }
                    ]
                },
                'corporate': {
                    'mood': 'professional',
                    'tempo': 'medium',
                    'tracks': [
                        {
                            'title': 'Business Progress',
                            'duration': 60,
                            'file': 'business_progress.mp3',
                            'tags': ['corporate', 'professional', 'clean']
                        }
                    ]
                },
                'ambient': {
                    'mood': 'luxury',
                    'tempo': 'slow',
                    'tracks': [
                        {
                            'title': 'Elegant Atmosphere',
                            'duration': 60,
                            'file': 'elegant_atmosphere.mp3',
                            'tags': ['ambient', 'luxury', 'sophisticated']
                        }
                    ]
                }
            },
            'moods': ['energetic', 'professional', 'luxury'],
            'tempos': ['slow', 'medium', 'fast']
        }
        
    async def select_music(self,
                         style: str,
                         duration: int,
                         mood: Optional[str] = None) -> Dict[str, Any]:
        """
        Select appropriate background music.
        
        Args:
            style: Video style
            duration: Required duration in seconds
            mood: Optional specific mood
            
        Returns:
            Selected track information
        """
        # Map style to genre
        style_genre_map = {
            'energetic': 'upbeat_electronic',
            'professional': 'corporate',
            'luxury': 'ambient'
        }
        
        genre = style_genre_map.get(style, 'upbeat_electronic')
        genre_info = self.catalog['genres'][genre]
        
        # Filter tracks by duration
        suitable_tracks = [
            track for track in genre_info['tracks']
            if track['duration'] >= duration
        ]
        
        if not suitable_tracks:
            # If no tracks match duration, use any track from genre
            suitable_tracks = genre_info['tracks']
            
        # Select random track from suitable ones
        selected_track = random.choice(suitable_tracks)
        
        return {
            'track': selected_track,
            'genre': genre,
            'mood': genre_info['mood'],
            'tempo': genre_info['tempo']
        }
        
    async def process_music(self,
                          track_path: str,
                          duration: int,
                          volume: float = 0.5) -> str:
        """
        Process music track for video.
        
        Args:
            track_path: Path to music track
            duration: Required duration
            volume: Background music volume (0-1)
            
        Returns:
            Path to processed music file
        """
        # TODO: Implement music processing
        # - Trim to duration
        # - Adjust volume
        # - Add fade in/out
        return track_path
        
    def get_available_tracks(self, 
                           genre: Optional[str] = None, 
                           mood: Optional[str] = None,
                           tempo: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of available tracks with optional filters."""
        tracks = []
        
        for genre_name, genre_info in self.catalog['genres'].items():
            if genre and genre != genre_name:
                continue
                
            if mood and genre_info['mood'] != mood:
                continue
                
            if tempo and genre_info['tempo'] != tempo:
                continue
                
            tracks.extend(genre_info['tracks'])
            
        return tracks
```

### ./src/director/agents/style_variation_agent.py
```python
from typing import Dict, Any, List
import random
from ..core.base_agent import BaseAgent

class StyleVariationAgent(BaseAgent):
    """Agent for generating TikTok-style variations."""
    
    def __init__(self):
        super().__init__()
        self.style_templates = {
            "energetic": {
                "transitions": ["quick_cut", "whip_pan", "zoom_blur"],
                "effects": ["shake", "glitch", "color_pulse"],
                "music_style": "upbeat",
                "text_style": "bold",
                "duration_range": (15, 30)
            },
            "cinematic": {
                "transitions": ["smooth_fade", "dolly_zoom", "parallax"],
                "effects": ["film_grain", "letterbox", "color_grade"],
                "music_style": "dramatic",
                "text_style": "elegant",
                "duration_range": (30, 60)
            },
            "minimal": {
                "transitions": ["simple_cut", "fade", "slide"],
                "effects": ["subtle_zoom", "soft_light", "clean"],
                "music_style": "ambient",
                "text_style": "modern",
                "duration_range": (20, 45)
            }
        }
        
    def generate_variation(
        self,
        base_style: str,
        target_audience: List[str],
        duration: float
    ) -> Dict[str, Any]:
        """
        Generate a TikTok-style variation.
        
        Args:
            base_style: Base style to build from
            target_audience: List of target audience segments
            duration: Target duration
            
        Returns:
            Style variation configuration
        """
        if base_style not in self.style_templates:
            raise ValueError(f"Unknown base style: {base_style}")
            
        base_template = self.style_templates[base_style]
        
        # Adjust style based on target audience
        variation = {
            "style": base_style,
            "transitions": list(base_template["transitions"]),
            "effects": list(base_template["effects"]),
            "music_style": base_template["music_style"],
            "text_style": base_template["text_style"]
        }
        
        # Audience-specific adjustments
        if "young_professionals" in target_audience:
            variation["effects"].extend(["dynamic_text", "emoji_pop"])
            variation["music_style"] = "trendy"
        elif "families" in target_audience:
            variation["effects"].extend(["warm_glow", "family_friendly"])
            variation["music_style"] = "uplifting"
        elif "luxury_buyers" in target_audience:
            variation["effects"].extend(["premium_grade", "elegant_text"])
            variation["music_style"] = "sophisticated"
            
        # Duration-based adjustments
        if duration <= 15:
            # Short-form optimizations
            variation["transitions"] = [t for t in variation["transitions"] if "quick" in t or "fast" in t]
            variation["effects"].append("attention_grab")
        elif duration >= 45:
            # Long-form enhancements
            variation["transitions"].extend(["story_transition", "mood_shift"])
            variation["effects"].append("narrative_enhance")
            
        return variation
        
    def optimize_engagement(
        self,
        variation: Dict[str, Any],
        platform_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Optimize variation based on platform engagement metrics.
        
        Args:
            variation: Current style variation
            platform_metrics: Engagement metrics from the platform
            
        Returns:
            Optimized variation
        """
        optimized = dict(variation)
        
        # Adjust based on view duration metrics
        if platform_metrics.get("avg_view_duration", 0) < 0.5:
            # Users dropping off quickly, make it more engaging
            optimized["effects"].extend(["pattern_interrupt", "hook_enhance"])
            optimized["transitions"] = [t for t in optimized["transitions"] if "slow" not in t]
            
        # Adjust based on engagement rate
        if platform_metrics.get("engagement_rate", 0) < 0.1:
            # Low engagement, add more interactive elements
            optimized["effects"].extend(["text_callout", "highlight_zoom"])
            
        # Adjust based on completion rate
        if platform_metrics.get("completion_rate", 0) < 0.7:
            # Users not finishing, optimize for retention
            optimized["effects"].append("retention_hook")
            optimized["transitions"].append("curiosity_gap")
            
        return optimized
        
    def get_trending_elements(self) -> Dict[str, List[str]]:
        """Get currently trending style elements."""
        # In a real implementation, this would fetch from a trends API
        return {
            "transitions": ["creative_zoom", "glitch_transition"],
            "effects": ["viral_text_style", "trending_filter"],
            "music_styles": ["viral_beats", "trending_sounds"],
            "hooks": ["question_hook", "statistic_hook"]
        }
```

### ./src/director/agents/script_agent.py
```python
from typing import Dict, Any, List, Optional
import json
from ..core.base_agent import BaseAgent

class ScriptAgent(BaseAgent):
    """Agent for generating dynamic video scripts using LLMs."""
    
    def __init__(self):
        self.script_templates = {
            "modern": {
                "hooks": [
                    "WAIT UNTIL YOU SEE THIS {location} DREAM HOME! 🏠✨",
                    "THIS {style} HOME IN {location} WILL BLOW YOUR MIND! 🔥",
                    "INSANE {price_range} PROPERTY ALERT! 🚨 {location} GEM! 💎"
                ],
                "features": [
                    "OBSESSED with this {feature}! 😍",
                    "THE MOST INCREDIBLE {feature} for {use_case}! 🤯",
                    "YOU WON'T BELIEVE this {feature}! ✨"
                ],
                "closings": [
                    "RUN don't walk! DM now! 🏃‍♂️",
                    "This won't last! Save this! ⭐️",
                    "Tag someone who needs to see this! 🔥"
                ]
            },
            "luxury": {
                "hooks": [
                    "EXCLUSIVE FIRST LOOK: ${price_range} {location} MANSION 👑",
                    "INSIDE THE MOST STUNNING {location} ESTATE YET! ✨",
                    "THIS {style} MASTERPIECE WILL LEAVE YOU SPEECHLESS! 🏰"
                ],
                "features": [
                    "LOOK AT THIS INCREDIBLE {feature}! Perfect for {use_case} 🤩",
                    "THE MOST INSANE {feature} you'll ever see! 🔥",
                    "I CAN'T BELIEVE this {feature}! Absolutely unreal! ✨"
                ],
                "closings": [
                    "Serious inquiries only - DM for private tour 🔑",
                    "Save this dream home! You won't regret it! ⭐️",
                    "Like & Share if this is your dream home! 🏠"
                ]
            },
            "minimal": {
                "hooks": [
                    "THIS {location} GEM IS EVERYTHING! ✨",
                    "PERFECTLY DESIGNED {style} HOME! 🏠",
                    "THE MOST AESTHETIC {location} FIND! 🎯"
                ],
                "features": [
                    "OBSESSED with this {feature} for {use_case}! 😍",
                    "Clean lines define this {feature}",
                    "Smart design in the {feature}"
                ],
                "closings": [
                    "Schedule your visit",
                    "Available for viewing",
                    "Connect for details"
                ]
            }
        }
        
    def _analyze_property(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze property data to determine key selling points."""
        analysis = {
            "price_range": self._get_price_range(property_data["price"]),
            "key_features": [],
            "unique_selling_points": [],
            "target_audience": [],
            "style_keywords": []
        }
        
        # Analyze price and location
        if property_data["price"] > 1000000:
            analysis["target_audience"].extend(["luxury-buyers", "investors"])
            analysis["style_keywords"].extend(["luxury", "premium", "exclusive"])
        else:
            analysis["target_audience"].extend(["homeowners", "families"])
            analysis["style_keywords"].extend(["modern", "comfortable", "practical"])
            
        # Analyze features
        # Check bedrooms
        if property_data.get("bedrooms", 0) >= 4:
            analysis["key_features"].append(("spacious layout", "family living"))
            analysis["unique_selling_points"].append("Large family home")
            
        # Check bathrooms
        if property_data.get("bathrooms", 0) >= 2.5:
            analysis["key_features"].append(("multiple bathrooms", "convenience"))
            
        # Check square footage
        sqft = property_data.get("sqft", 0)
        if sqft > 3000:
            analysis["key_features"].append(("expansive space", "luxury living"))
        elif sqft > 2000:
            analysis["key_features"].append(("generous space", "comfortable living"))
            
        # Check lot size
        if property_data.get("lot_size", 0) > 10000:  # 10,000 sqft
            analysis["key_features"].append(("large lot", "outdoor living"))
            
        # Process custom features
        for feature in property_data.get("features", []):
            feature_lower = feature.lower()
            if "kitchen" in feature_lower:
                analysis["key_features"].append((feature, "modern living"))
            elif "floor" in feature_lower:
                analysis["key_features"].append((feature, "design"))
            elif "yard" in feature_lower or "outdoor" in feature_lower:
                analysis["key_features"].append((feature, "outdoor living"))
            else:
                analysis["key_features"].append((feature, "lifestyle"))
            
        return analysis
        
    def _get_price_range(self, price: float) -> str:
        """Convert price to descriptive range."""
        if price < 300000:
            return "starter"
        elif price < 600000:
            return "mid-range"
        elif price < 1000000:
            return "upscale"
        else:
            return "luxury"
            
    MIN_DURATION = 10.0  # TikTok minimum video duration
    MAX_DURATION = 60.0  # TikTok maximum video duration
    
    def generate_script(
        self,
        property_data: Dict[str, Any],
        style: str = "modern",
        duration: float = 30.0,
        target_segments: int = 5
    ) -> Dict[str, Any]:
        """
        Generate a dynamic video script based on property data.
        
        Args:
            property_data: Dictionary containing property information
            style: Script style to use
            duration: Target video duration
            target_segments: Number of segments to generate
            
        Returns:
            Dictionary containing script segments and metadata
        """
        # Validate duration
        if duration < self.MIN_DURATION:
            duration = self.MIN_DURATION
        elif duration > self.MAX_DURATION:
            duration = self.MAX_DURATION
            
        # Adjust segments based on duration
        if duration <= 15.0:
            target_segments = min(target_segments, 3)  # Shorter videos need fewer segments
        elif duration >= 45.0:
            target_segments = 7  # Longer videos need more segments
            # Add extra segments for very long videos
            if duration >= 55.0:
                target_segments = 8
            
        analysis = self._analyze_property(property_data)
        templates = self.script_templates[style]
        
        # Calculate timing
        segment_duration = duration / target_segments
        current_time = 0.0
        
        script = {
            "style": style,
            "duration": duration,
            "segments": [],
            "metadata": {
                "price_range": analysis["price_range"],
                "target_audience": analysis["target_audience"],
                "key_features": analysis["key_features"]
            }
        }
        
        # Generate hook
        hook_text = templates["hooks"][0].format(
            location=property_data["location"],
            style=property_data.get("style", "modern"),
            price_range=analysis["price_range"]
        )
        script["segments"].append({
            "type": "hook",
            "text": hook_text,
            "start_time": current_time,
            "duration": segment_duration,
            "focus": "exterior"
        })
        current_time += segment_duration
        
        # Generate feature segments
        # Calculate how many feature segments we need
        feature_count = target_segments - 2  # -2 for hook and closing
        
        # If we need more features than available, duplicate some
        features = analysis["key_features"]
        while len(features) < feature_count:
            features.extend(analysis["key_features"])
        features = features[:feature_count]
        
        # Generate feature segments
        for feature, use_case in features:
            feature_text = templates["features"][0].format(
                feature=feature,
                use_case=use_case
            )
            script["segments"].append({
                "type": "feature",
                "text": feature_text,
                "start_time": current_time,
                "duration": segment_duration,
                "focus": feature.replace(" ", "_")
            })
            current_time += segment_duration
            
        # Generate closing
        closing_text = templates["closings"][0]
        script["segments"].append({
            "type": "closing",
            "text": closing_text,
            "start_time": current_time,
            "duration": segment_duration,
            "focus": "exterior"
        })
        
        return script
        
    def adjust_script_pacing(
        self,
        script: Dict[str, Any],
        engagement_data: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Adjust script pacing based on engagement data.
        
        Args:
            script: Original script
            engagement_data: Optional dictionary with engagement metrics
            
        Returns:
            Adjusted script
        """
        if not engagement_data:
            return script
            
        adjusted = script.copy()
        segments = adjusted["segments"]
        
        # Calculate engagement-based duration adjustments
        total_engagement = sum(engagement_data.values())
        for segment in segments:
            segment_type = segment["type"]
            if segment_type in engagement_data:
                engagement_ratio = engagement_data[segment_type] / total_engagement
                # Adjust duration based on engagement (±20% max)
                base_duration = segment["duration"]
                adjustment = (engagement_ratio - 1/len(engagement_data)) * 0.2
                segment["duration"] = base_duration * (1 + adjustment)
                
        # Normalize durations to match target duration
        total_duration = sum(s["duration"] for s in segments)
        scale_factor = script["duration"] / total_duration
        for segment in segments:
            segment["duration"] *= scale_factor
            
        # Update start times
        current_time = 0.0
        for segment in segments:
            segment["start_time"] = current_time
            current_time += segment["duration"]
            
        return adjusted
```

### ./src/director/agents/__init__.py
```python
```

### ./src/director/agents/real_estate_video_agent.py
```python
from typing import Dict, Any, List
from ..core.base_agent import BaseAgent
from .script_agent import ScriptAgent
from .pacing_agent import PacingAgent
from .effect_agent import EffectAgent

class RealEstateVideoAgent(BaseAgent):
    """Agent for property-specific video generation."""
    
    def __init__(self):
        super().__init__()
        self.script_agent = ScriptAgent()
        self.pacing_agent = PacingAgent()
        self.effect_agent = EffectAgent()
        
    def analyze_property(self, property_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze property data to determine key selling points and features.
        
        Args:
            property_data: Property listing data
            
        Returns:
            Analysis results including key features, target audience, etc.
        """
        analysis = {
            "key_features": [],
            "target_audience": [],
            "price_segment": "",
            "unique_selling_points": []
        }
        
        # Analyze price segment
        price = property_data.get("price", 0)
        if price > 1000000:
            analysis["price_segment"] = "luxury"
        elif price > 500000:
            analysis["price_segment"] = "premium"
        else:
            analysis["price_segment"] = "standard"
            
        # Extract key features
        features = property_data.get("features", [])
        amenities = property_data.get("amenities", [])
        all_features = features + amenities
        
        # Prioritize features based on price segment
        if analysis["price_segment"] == "luxury":
            priority_features = ["pool", "spa", "view", "smart_home"]
        else:
            priority_features = ["location", "space", "updates", "parking"]
            
        for feature in all_features:
            if any(p in feature.lower() for p in priority_features):
                analysis["key_features"].append(feature)
                
        # Determine target audience
        if property_data.get("bedrooms", 0) >= 4:
            analysis["target_audience"].append("families")
        if "modern" in " ".join(features).lower():
            analysis["target_audience"].append("young_professionals")
        if "retirement" in " ".join(features).lower():
            analysis["target_audience"].append("retirees")
            
        # Identify unique selling points
        if property_data.get("year_built", 0) > 2020:
            analysis["unique_selling_points"].append("new_construction")
        if property_data.get("lot_size", 0) > 10000:
            analysis["unique_selling_points"].append("large_lot")
        if "view" in " ".join(features).lower():
            analysis["unique_selling_points"].append("scenic_view")
            
        return analysis
        
    def generate_video_plan(
        self,
        property_data: Dict[str, Any],
        style: str = "modern",
        duration: float = 30.0
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive video plan for the property.
        
        Args:
            property_data: Property listing data
            style: Video style preference
            duration: Target video duration
            
        Returns:
            Video plan including script, pacing, and effects
        """
        # Analyze property
        analysis = self.analyze_property(property_data)
        
        # Generate script
        script = self.script_agent.generate_script(
            property_data=property_data,
            style=style,
            duration=duration
        )
        
        # Optimize pacing
        content_analysis = self.pacing_agent.analyze_content(script["segments"])
        pacing = self.pacing_agent.get_optimal_pacing(
            content_analysis=content_analysis,
            target_duration=duration,
            style=style
        )
        
        # Apply pacing
        segments = self.pacing_agent.apply_pacing(
            segments=script["segments"],
            pacing=pacing,
            target_duration=duration
        )
        
        # Generate effects
        effects = []
        for segment in segments:
            segment_effects = self.effect_agent.generate_effect_combination(
                style=style,
                content_type=segment.get("content_type", "interior")
            )
            effects.append(segment_effects)
            
        return {
            "analysis": analysis,
            "script": script,
            "segments": segments,
            "effects": effects,
            "style": style,
            "duration": duration
        }
```

### ./src/director/agents/voice_agent.py
```python
from typing import Dict, Any, Optional
import os
import json
import aiohttp
from pathlib import Path

class VoiceAgent:
    """
    Agent for generating voiceovers using ElevenLabs API.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.api_key = os.getenv('ELEVENLABS_API_KEY')
        self.base_url = 'https://api.elevenlabs.io/v1'
        
    def _default_config(self) -> Dict[str, Any]:
        """Default voice configuration."""
        return {
            'voices': {
                'professional': {
                    'voice_id': 'ErXwobaYiN019PkySvjV',  # Antoni voice
                    'settings': {
                        'stability': 0.75,
                        'similarity_boost': 0.75
                    }
                },
                'friendly': {
                    'voice_id': 'MF3mGyEYCl7XYWbV9V6O',  # Elli voice
                    'settings': {
                        'stability': 0.65,
                        'similarity_boost': 0.85
                    }
                },
                'luxury': {
                    'voice_id': 'VR6AewLTigWG4xSOukaG',  # Adam voice
                    'settings': {
                        'stability': 0.85,
                        'similarity_boost': 0.65
                    }
                }
            },
            'default_voice': 'professional'
        }
        
    async def generate_voiceover(self,
                               script: str,
                               style: str = 'professional',
                               output_path: Optional[str] = None) -> str:
        """
        Generate voiceover for script using ElevenLabs API.
        
        Args:
            script: Text to convert to speech
            style: Voice style to use
            output_path: Optional path to save audio file
            
        Returns:
            Path to generated audio file
        """
        if not self.api_key:
            raise ValueError("ElevenLabs API key not found in environment variables")
            
        voice_config = self.config['voices'].get(style, 
                                               self.config['voices'][self.config['default_voice']])
                                               
        headers = {
            'xi-api-key': self.api_key,
            'Content-Type': 'application/json'
        }
        
        data = {
            'text': script,
            'voice_settings': voice_config['settings']
        }
        
        url = f"{self.base_url}/text-to-speech/{voice_config['voice_id']}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status != 200:
                    raise Exception(f"ElevenLabs API error: {await response.text()}")
                    
                audio_data = await response.read()
                
        if not output_path:
            output_path = f"temp_voiceover_{style}_{hash(script)}.mp3"
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(audio_data)
            
        return str(output_path)
        
    async def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """Get information about a specific voice."""
        if not self.api_key:
            raise ValueError("ElevenLabs API key not found in environment variables")
            
        headers = {'xi-api-key': self.api_key}
        url = f"{self.base_url}/voices/{voice_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"ElevenLabs API error: {await response.text()}")
                return await response.json()
                
    def get_voice_settings(self, style: str) -> Dict[str, Any]:
        """Get voice settings for a specific style."""
        return self.config['voices'].get(style, 
                                      self.config['voices'][self.config['default_voice']])
```

### ./src/director/agents/style_agent.py
```python
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

class StyleAgent:
    """
    Agent for managing and varying video styles.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
    def _default_config(self) -> Dict[str, Any]:
        """Default style configuration."""
        return {
            'styles': {
                'energetic': {
                    'video': {
                        'transitions': ['fast_fade', 'zoom_blur', 'slide'],
                        'effects': ['zoom', 'shake', 'flash'],
                        'pacing': 'fast',
                        'color_grade': 'vibrant'
                    },
                    'text': {
                        'font': 'Impact',
                        'animation': 'pop',
                        'color_scheme': ['#FF4B4B', '#FFFFFF'],
                        'placement': 'dynamic'
                    },
                    'audio': {
                        'music_style': 'upbeat_electronic',
                        'voice_style': 'friendly',
                        'sound_effects': True
                    }
                },
                'professional': {
                    'video': {
                        'transitions': ['smooth_fade', 'dissolve', 'wipe'],
                        'effects': ['pan', 'tilt', 'focus'],
                        'pacing': 'medium',
                        'color_grade': 'natural'
                    },
                    'text': {
                        'font': 'Helvetica',
                        'animation': 'fade',
                        'color_scheme': ['#2C3E50', '#ECF0F1'],
                        'placement': 'structured'
                    },
                    'audio': {
                        'music_style': 'corporate',
                        'voice_style': 'professional',
                        'sound_effects': False
                    }
                },
                'luxury': {
                    'video': {
                        'transitions': ['cinematic_fade', 'blur_dissolve', 'elegant_slide'],
                        'effects': ['depth_blur', 'light_rays', 'film_grain'],
                        'pacing': 'slow',
                        'color_grade': 'cinematic'
                    },
                    'text': {
                        'font': 'Didot',
                        'animation': 'elegant',
                        'color_scheme': ['#B8860B', '#FFFFFF'],
                        'placement': 'minimal'
                    },
                    'audio': {
                        'music_style': 'ambient',
                        'voice_style': 'luxury',
                        'sound_effects': False
                    }
                }
            },
            'variations': {
                'color_grades': ['natural', 'vibrant', 'cinematic', 'moody', 'warm'],
                'text_animations': ['fade', 'pop', 'slide', 'elegant', 'dynamic'],
                'transition_types': ['fade', 'dissolve', 'slide', 'zoom', 'blur']
            }
        }
        
    def get_style(self, style_name: str) -> Dict[str, Any]:
        """Get complete style configuration."""
        return self.config['styles'].get(style_name, self.config['styles']['professional'])
        
    def generate_variation(self,
                         base_style: str,
                         variation_strength: float = 0.3) -> Dict[str, Any]:
        """
        Generate a variation of a base style.
        
        Args:
            base_style: Name of base style
            variation_strength: How much to vary (0-1)
            
        Returns:
            Modified style configuration
        """
        base_config = self.get_style(base_style).copy()
        
        if variation_strength <= 0:
            return base_config
            
        # Vary video effects
        if random.random() < variation_strength:
            base_config['video']['effects'] = random.sample(
                base_config['video']['effects'],
                k=max(1, len(base_config['video']['effects']) - 1)
            )
            
        # Vary color grade
        if random.random() < variation_strength:
            base_config['video']['color_grade'] = random.choice(
                self.config['variations']['color_grades']
            )
            
        # Vary text animation
        if random.random() < variation_strength:
            base_config['text']['animation'] = random.choice(
                self.config['variations']['text_animations']
            )
            
        return base_config
        
    def adapt_style_to_content(self,
                             content_type: str,
                             property_value: float,
                             target_market: str) -> str:
        """
        Adapt style based on content and target market.
        
        Args:
            content_type: Type of property
            property_value: Property value
            target_market: Target market description
            
        Returns:
            Recommended style name
        """
        # Luxury properties
        if property_value > 1000000 or 'luxury' in target_market.lower():
            return 'luxury'
            
        # Young/urban market
        if any(term in target_market.lower() for term in ['young', 'urban', 'professional']):
            return 'energetic'
            
        # Default to professional
        return 'professional'
        
    def get_style_elements(self,
                         style_name: str,
                         element_type: str) -> Dict[str, Any]:
        """Get specific elements of a style."""
        style = self.get_style(style_name)
        return style.get(element_type, {})
```

### ./src/director/agents/text_overlay_agent.py
```python
from typing import Dict, Any, List
from ..core.base_agent import BaseAgent

class TextOverlayAgent(BaseAgent):
    """Agent for generating and styling text overlays in TikTok-style videos."""
    
    def __init__(self):
        self.style_presets = {
            "modern": {
                "title": {
                    "font": "Helvetica Neue",
                    "weight": "bold",
                    "size": "large",
                    "color": "#FFFFFF",
                    "shadow": True,
                    "animation": "slide_up"
                },
                "price": {
                    "font": "Helvetica Neue",
                    "weight": "bold",
                    "size": "xlarge",
                    "color": "#00FF88",
                    "shadow": True,
                    "animation": "pop"
                },
                "features": {
                    "font": "Helvetica Neue",
                    "weight": "medium",
                    "size": "medium",
                    "color": "#FFFFFF",
                    "shadow": True,
                    "animation": "fade_in"
                }
            },
            "luxury": {
                "title": {
                    "font": "Didot",
                    "weight": "bold",
                    "size": "large",
                    "color": "#FFFFFF",
                    "shadow": False,
                    "animation": "elegant_fade"
                },
                "price": {
                    "font": "Didot",
                    "weight": "bold",
                    "size": "xlarge",
                    "color": "#FFD700",
                    "shadow": False,
                    "animation": "elegant_reveal"
                },
                "features": {
                    "font": "Didot",
                    "weight": "medium",
                    "size": "medium",
                    "color": "#FFFFFF",
                    "shadow": False,
                    "animation": "smooth_fade"
                }
            },
            "minimal": {
                "title": {
                    "font": "SF Pro Display",
                    "weight": "regular",
                    "size": "large",
                    "color": "#FFFFFF",
                    "shadow": False,
                    "animation": "minimal_fade"
                },
                "price": {
                    "font": "SF Pro Display",
                    "weight": "light",
                    "size": "xlarge",
                    "color": "#FFFFFF",
                    "shadow": False,
                    "animation": "minimal_slide"
                },
                "features": {
                    "font": "SF Pro Display",
                    "weight": "light",
                    "size": "medium",
                    "color": "#FFFFFF",
                    "shadow": False,
                    "animation": "minimal_reveal"
                }
            }
        }
        
        self.animations = {
            "slide_up": {"duration": 0.5, "ease": "cubic-bezier(0.4, 0, 0.2, 1)"},
            "pop": {"duration": 0.3, "ease": "cubic-bezier(0.34, 1.56, 0.64, 1)"},
            "fade_in": {"duration": 0.4, "ease": "cubic-bezier(0.4, 0, 0.2, 1)"},
            "elegant_fade": {"duration": 0.8, "ease": "cubic-bezier(0.4, 0, 0.2, 1)"},
            "elegant_reveal": {"duration": 1.0, "ease": "cubic-bezier(0.4, 0, 0.2, 1)"},
            "smooth_fade": {"duration": 0.6, "ease": "cubic-bezier(0.4, 0, 0.2, 1)"},
            "minimal_fade": {"duration": 0.4, "ease": "linear"},
            "minimal_slide": {"duration": 0.5, "ease": "cubic-bezier(0.4, 0, 0.2, 1)"},
            "minimal_reveal": {"duration": 0.3, "ease": "cubic-bezier(0.4, 0, 0.2, 1)"}
        }
    
    def get_style_for_template(self, template_name: str) -> Dict[str, Any]:
        """Get text style configuration for a template."""
        style_name = template_name.split('_')[0]  # e.g., 'modern' from 'modern_rapid_standard'
        return self.style_presets.get(style_name, self.style_presets["modern"])
    
    def generate_overlay_config(
        self,
        template_name: str,
        text_content: Dict[str, str],
        timing: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Generate text overlay configuration for a video.
        
        Args:
            template_name: Name of the template being used
            text_content: Dictionary of text content (title, price, features, etc.)
            timing: Dictionary of timing information for each text element
            
        Returns:
            List of overlay configurations with styling and animation
        """
        style = self.get_style_for_template(template_name)
        overlays = []
        
        for text_type, content in text_content.items():
            if text_type in style:
                text_style = style[text_type]
                animation = self.animations[text_style["animation"]]
                
                overlay = {
                    "content": content,
                    "type": text_type,
                    "style": {
                        "font_family": text_style["font"],
                        "font_weight": text_style["weight"],
                        "font_size": text_style["size"],
                        "color": text_style["color"],
                        "text_shadow": text_style["shadow"]
                    },
                    "animation": {
                        "type": text_style["animation"],
                        "duration": animation["duration"],
                        "easing": animation["ease"]
                    },
                    "timing": {
                        "start": timing.get(text_type, 0),
                        "duration": timing.get(f"{text_type}_duration", 3.0)
                    }
                }
                overlays.append(overlay)
        
        return overlays
```

### ./src/director/agents/effect_agent.py
```python
from typing import Dict, Any, List, Optional
import random
from ..core.base_agent import BaseAgent

class EffectAgent(BaseAgent):
    """Agent for generating and combining video effects."""
    
    def __init__(self):
        self.effect_library = {
            "color": {
                "modern": [
                    {"name": "vibrant", "contrast": 1.2, "saturation": 1.3, "brightness": 1.1},
                    {"name": "cool", "contrast": 1.1, "saturation": 0.9, "temperature": -10},
                    {"name": "warm", "contrast": 1.1, "saturation": 1.1, "temperature": 10}
                ],
                "luxury": [
                    {"name": "cinematic", "contrast": 1.3, "saturation": 0.9, "shadows": 0.8},
                    {"name": "rich", "contrast": 1.2, "saturation": 1.1, "highlights": 0.9},
                    {"name": "moody", "contrast": 1.4, "saturation": 0.8, "shadows": 0.7}
                ],
                "minimal": [
                    {"name": "clean", "contrast": 1.1, "saturation": 0.9, "brightness": 1.05},
                    {"name": "bright", "contrast": 1.0, "saturation": 0.8, "brightness": 1.2},
                    {"name": "pure", "contrast": 1.05, "saturation": 0.85, "highlights": 1.1}
                ]
            },
            "overlay": {
                "modern": [
                    {"name": "light_leak", "opacity": 0.2, "blend": "screen"},
                    {"name": "grain", "opacity": 0.1, "blend": "overlay"},
                    {"name": "vignette", "opacity": 0.15, "blend": "multiply"}
                ],
                "luxury": [
                    {"name": "film_grain", "opacity": 0.08, "blend": "overlay"},
                    {"name": "soft_light", "opacity": 0.15, "blend": "soft_light"},
                    {"name": "cinematic_bars", "opacity": 1.0, "blend": "normal"}
                ],
                "minimal": [
                    {"name": "subtle_grain", "opacity": 0.05, "blend": "overlay"},
                    {"name": "soft_vignette", "opacity": 0.1, "blend": "multiply"},
                    {"name": "clean_overlay", "opacity": 0.08, "blend": "screen"}
                ]
            },
            "filter": {
                "modern": [
                    {"name": "sharpen", "amount": 0.3, "radius": 1.0},
                    {"name": "clarity", "amount": 0.4, "radius": 1.5},
                    {"name": "dehaze", "amount": 0.2}
                ],
                "luxury": [
                    {"name": "bloom", "amount": 0.3, "radius": 2.0},
                    {"name": "glow", "amount": 0.25, "radius": 1.8},
                    {"name": "soft_focus", "amount": 0.2, "radius": 1.5}
                ],
                "minimal": [
                    {"name": "clarity", "amount": 0.2, "radius": 1.0},
                    {"name": "sharpen", "amount": 0.15, "radius": 0.8},
                    {"name": "clean", "amount": 0.1, "radius": 1.0}
                ]
            }
        }
        
        self.combination_rules = {
            "modern": {
                "max_overlays": 2,
                "required_effects": ["color", "filter"],
                "optional_effects": ["overlay"],
                "compatibility": {
                    "light_leak": ["grain"],
                    "vignette": ["grain", "light_leak"]
                }
            },
            "luxury": {
                "max_overlays": 2,
                "required_effects": ["color", "filter", "overlay"],
                "optional_effects": [],
                "compatibility": {
                    "film_grain": ["soft_light"],
                    "cinematic_bars": ["film_grain", "soft_light"]
                }
            },
            "minimal": {
                "max_overlays": 1,
                "required_effects": ["color"],
                "optional_effects": ["filter", "overlay"],
                "compatibility": {
                    "subtle_grain": ["soft_vignette"],
                    "clean_overlay": ["subtle_grain"]
                }
            }
        }
        
    def generate_effect_combination(
        self,
        style: str,
        content_type: str,
        intensity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generate a compatible combination of effects.
        
        Args:
            style: Video style
            content_type: Type of content (exterior, interior, etc.)
            intensity: Effect intensity multiplier
            
        Returns:
            Dictionary of combined effects
        """
        rules = self.combination_rules[style]
        effects = {
            "color": None,
            "filter": None,
            "overlay": []
        }
        
        # Apply required effects
        for effect_type in rules["required_effects"]:
            if effect_type == "overlay":
                continue  # Handled separately
            effects[effect_type] = self._select_effect(
                effect_type, style, intensity
            )
            
        # Apply optional effects
        for effect_type in rules["optional_effects"]:
            if random.random() < 0.7:  # 70% chance to apply optional effect
                if effect_type == "overlay":
                    continue  # Handled separately
                effects[effect_type] = self._select_effect(
                    effect_type, style, intensity
                )
                
        # Handle overlays with compatibility rules
        available_overlays = self.effect_library["overlay"][style]
        selected_overlays = []
        
        while (len(selected_overlays) < rules["max_overlays"] and 
               len(available_overlays) > 0):
            overlay = random.choice(available_overlays)
            available_overlays.remove(overlay)
            
            # Check compatibility
            compatible = True
            for selected in selected_overlays:
                if (selected["name"] in rules["compatibility"] and
                    overlay["name"] not in rules["compatibility"][selected["name"]]):
                    compatible = False
                    break
                    
            if compatible:
                # Adjust opacity based on intensity
                adjusted = overlay.copy()
                adjusted["opacity"] *= intensity
                selected_overlays.append(adjusted)
                
        effects["overlay"] = selected_overlays
        
        # Add content-specific adjustments
        self._adjust_for_content(effects, content_type)
        
        return effects
        
    def _select_effect(
        self,
        effect_type: str,
        style: str,
        intensity: float
    ) -> Dict[str, Any]:
        """Select and adjust an effect based on style and intensity."""
        options = self.effect_library[effect_type][style]
        effect = random.choice(options).copy()
        
        # Adjust effect parameters based on intensity
        if effect_type == "color":
            for key in ["contrast", "saturation", "brightness"]:
                if key in effect:
                    deviation = abs(1 - effect[key])
                    effect[key] = 1 + (deviation * intensity * 
                                     (1 if effect[key] > 1 else -1))
        elif effect_type == "filter":
            effect["amount"] *= intensity
            
        return effect
        
    def _adjust_for_content(
        self,
        effects: Dict[str, Any],
        content_type: str
    ) -> None:
        """Adjust effects based on content type."""
        if content_type == "exterior":
            # Enhance contrast and saturation for exterior shots
            if effects["color"]:
                effects["color"]["contrast"] *= 1.1
                effects["color"]["saturation"] *= 1.1
        elif content_type == "interior":
            # Boost brightness for interior shots
            if effects["color"]:
                effects["color"]["brightness"] = max(
                    1.1,
                    effects["color"].get("brightness", 1.0)
                )
        elif content_type == "feature":
            # Enhance clarity for feature shots
            if effects["filter"]:
                effects["filter"]["amount"] *= 1.2
                
    def blend_effects(
        self,
        effects_a: Dict[str, Any],
        effects_b: Dict[str, Any],
        blend_factor: float
    ) -> Dict[str, Any]:
        """
        Blend two effect combinations.
        
        Args:
            effects_a: First effect combination
            effects_b: Second effect combination
            blend_factor: Blend factor (0-1, 0=full A, 1=full B)
            
        Returns:
            Blended effect combination
        """
        blended = {
            "color": {},
            "filter": {},
            "overlay": []
        }
        
        # Blend color effects
        if effects_a.get("color") and effects_b.get("color"):
            for key in ["contrast", "saturation", "brightness", "temperature"]:
                if key in effects_a["color"] and key in effects_b["color"]:
                    blended["color"][key] = (
                        effects_a["color"][key] * (1 - blend_factor) +
                        effects_b["color"][key] * blend_factor
                    )
                    
        # Blend filter effects
        if effects_a.get("filter") and effects_b.get("filter"):
            for key in ["amount", "radius"]:
                if key in effects_a["filter"] and key in effects_b["filter"]:
                    blended["filter"][key] = (
                        effects_a["filter"][key] * (1 - blend_factor) +
                        effects_b["filter"][key] * blend_factor
                    )
                    
        # Select overlays based on blend factor
        all_overlays = (effects_a.get("overlay", []) + 
                       effects_b.get("overlay", []))
        if all_overlays:
            # Prioritize overlays from the dominant effect set
            dominant = effects_a if blend_factor < 0.5 else effects_b
            secondary = effects_b if blend_factor < 0.5 else effects_a
            
            blended["overlay"] = (
                dominant.get("overlay", [])[:1] +
                secondary.get("overlay", [])[:1]
            )
            
        return blended
```

### ./src/director/ai/__init__.py
```python
```

### ./src/director/ai/llm_manager.py
```python
from typing import Dict, Any, List, Optional
import os
from pathlib import Path
import yaml
import json

class LLMManager:
    """
    Manages LLM interactions for content generation, using Director's LLM integration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        # TODO: Initialize Director's LLM system
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[Any, Any]:
        """Load LLM configuration."""
        if not config_path:
            config_path = os.getenv('DIRECTOR_LLM_CONFIG', 'config/director/llm.yaml')
            
        config_path = Path(config_path)
        if not config_path.exists():
            return self._create_default_config(config_path)
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _create_default_config(self, config_path: Path) -> Dict[Any, Any]:
        """Create default LLM configuration."""
        default_config = {
            'model': 'gpt-4',
            'temperature': 0.7,
            'max_tokens': 2000,
            'style_presets': {
                'energetic': {
                    'tone': 'upbeat',
                    'pacing': 'fast',
                    'language': 'casual'
                },
                'professional': {
                    'tone': 'formal',
                    'pacing': 'moderate',
                    'language': 'professional'
                },
                'luxury': {
                    'tone': 'sophisticated',
                    'pacing': 'slow',
                    'language': 'upscale'
                }
            },
            'prompt_templates': {
                'script': '''Create an engaging TikTok-style script for a {duration} second real estate video.
                Property Details:
                - Title: {title}
                - Price: {price}
                - Location: {location}
                - Features: {features}
                
                Style: {style}
                Target Audience: {audience}
                Key Highlights: {highlights}
                
                The script should be dynamic and attention-grabbing, following TikTok best practices.
                Include timestamps for transitions and effects.''',
                
                'style_selection': '''Analyze the following property details and recommend the best video style:
                Property: {property_details}
                Price Range: {price_range}
                Target Market: {target_market}
                
                Consider the property's unique features and target audience to select the most effective presentation style.'''
            }
        }
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
            
        return default_config
        
    async def generate_video_script(self,
                                  listing_data: Dict[str, Any],
                                  style_preset: str = 'energetic',
                                  duration: int = 60) -> Dict[str, Any]:
        """
        Generate a video script based on listing data.
        
        Args:
            listing_data: Property listing information
            style_preset: Style preset to use
            duration: Video duration in seconds
            
        Returns:
            Dictionary containing:
            - script: Generated script text
            - segments: List of timed segments
            - style_directions: Style and effect directions
        """
        prompt = self.config['prompt_templates']['script'].format(
            duration=duration,
            title=listing_data.get('title', ''),
            price=listing_data.get('price', ''),
            location=listing_data.get('location', ''),
            features=', '.join(listing_data.get('features', [])),
            style=self.config['style_presets'][style_preset]['tone'],
            audience=listing_data.get('target_market', 'general'),
            highlights=listing_data.get('highlights', '')
        )
        
        # TODO: Use Director's LLM to generate script
        # For now, return a mock response
        return {
            'script': 'Mock script content',
            'segments': [
                {'time': 0, 'content': 'Introduction'},
                {'time': 15, 'content': 'Features'},
                {'time': 30, 'content': 'Highlights'},
                {'time': 45, 'content': 'Call to Action'}
            ],
            'style_directions': {
                'tone': self.config['style_presets'][style_preset]['tone'],
                'pacing': self.config['style_presets'][style_preset]['pacing'],
                'language': self.config['style_presets'][style_preset]['language']
            }
        }
        
    async def select_video_style(self, listing_data: Dict[str, Any]) -> str:
        """
        Analyze listing data and select the most appropriate video style.
        
        Args:
            listing_data: Property listing information
            
        Returns:
            Selected style preset name
        """
        prompt = self.config['prompt_templates']['style_selection'].format(
            property_details=json.dumps(listing_data, indent=2),
            price_range=f"${listing_data.get('price', 0):,}",
            target_market=listing_data.get('target_market', 'general')
        )
        
        # TODO: Use Director's LLM to select style
        # For now, return a default style
        return 'energetic'
        
    def get_style_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get style preset configuration."""
        return self.config['style_presets'].get(preset_name, self.config['style_presets']['energetic'])
```

### ./src/director/ai/content_generator.py
```python
from typing import Dict, Any, List, Optional
from .llm_manager import LLMManager

class ContentGenerator:
    """
    Generates video content using LLM-driven decisions.
    """
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        
    async def generate_video_content(self,
                                   listing_data: Dict[str, Any],
                                   duration: int = 60) -> Dict[str, Any]:
        """
        Generate complete video content including script, style, and segments.
        
        Args:
            listing_data: Property listing information
            duration: Video duration in seconds
            
        Returns:
            Dictionary containing all content elements:
            - script: Generated script
            - style: Selected style
            - segments: Timed content segments
            - directions: Production directions
        """
        # Select appropriate style based on property details
        style_preset = await self.llm_manager.select_video_style(listing_data)
        
        # Generate script with selected style
        script_data = await self.llm_manager.generate_video_script(
            listing_data,
            style_preset,
            duration
        )
        
        # Enhance the content with additional elements
        return {
            'script': script_data['script'],
            'style': {
                'preset': style_preset,
                'settings': self.llm_manager.get_style_preset(style_preset)
            },
            'segments': script_data['segments'],
            'directions': {
                'style': script_data['style_directions'],
                'transitions': self._generate_transitions(script_data['segments']),
                'effects': self._generate_effects(style_preset),
                'music': self._select_music_style(style_preset)
            }
        }
        
    def _generate_transitions(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate transition effects between segments."""
        transitions = []
        for i in range(len(segments) - 1):
            transitions.append({
                'time': segments[i+1]['time'],
                'type': 'smooth_fade',  # Default transition
                'duration': 0.5
            })
        return transitions
        
    def _generate_effects(self, style_preset: str) -> List[Dict[str, Any]]:
        """Generate video effects based on style."""
        effects = []
        if style_preset == 'energetic':
            effects = [
                {'type': 'zoom', 'intensity': 'medium'},
                {'type': 'text_pop', 'intensity': 'high'},
                {'type': 'color_enhance', 'intensity': 'medium'}
            ]
        elif style_preset == 'luxury':
            effects = [
                {'type': 'smooth_pan', 'intensity': 'low'},
                {'type': 'cinematic_fade', 'intensity': 'medium'},
                {'type': 'depth_blur', 'intensity': 'low'}
            ]
        return effects
        
    def _select_music_style(self, style_preset: str) -> Dict[str, Any]:
        """Select music style based on video style."""
        music_styles = {
            'energetic': {
                'genre': 'upbeat_electronic',
                'tempo': 'fast',
                'intensity': 'high'
            },
            'professional': {
                'genre': 'corporate',
                'tempo': 'medium',
                'intensity': 'low'
            },
            'luxury': {
                'genre': 'ambient',
                'tempo': 'slow',
                'intensity': 'medium'
            }
        }
        return music_styles.get(style_preset, music_styles['energetic'])
```

### ./src/director/video_generator.py
```python
from typing import List, Callable, Optional
from .variation.template_engine import TemplateLibrary
from ..api.models import ListingData

class VideoGenerator:
    """Video generator class for creating real estate videos."""
    
    def __init__(self):
        """Initialize video generator."""
        self.template_library = TemplateLibrary()
    
    async def generate(
        self,
        template_name: str,
        listing_data: ListingData,
        images: List[str],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> str:
        """
        Generate a video using the specified template and data.
        
        Args:
            template_name: Name of the template to use
            listing_data: Listing data for the video
            images: List of image URLs or base64 strings
            progress_callback: Optional callback for progress updates
            
        Returns:
            URL of the generated video
            
        Raises:
            ValueError: If template not found or invalid data
        """
        try:
            # Get template
            template = self.template_library.get_template(template_name)
            if not template:
                raise ValueError(f"Template {template_name} not found")
                
            # Update progress
            if progress_callback:
                progress_callback(10)
                
            # TODO: Download/process images
            if progress_callback:
                progress_callback(30)
                
            # TODO: Generate video segments
            if progress_callback:
                progress_callback(60)
                
            # TODO: Combine segments
            if progress_callback:
                progress_callback(80)
                
            # TODO: Upload to storage
            if progress_callback:
                progress_callback(90)
            
            # Return mock URL for now
            result_url = f"https://storage.example.com/videos/{template_name}.mp4"
            
            if progress_callback:
                progress_callback(100)
                
            return result_url
            
        except Exception as e:
            raise ValueError(f"Video generation failed: {str(e)}")
```

### ./src/director/effects/__init__.py
```python
```

### ./src/director/effects/visual_effects.py
```python
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from pathlib import Path

class VisualEffect:
    """Base class for visual effects."""
    
    def __init__(self, intensity: float = 1.0):
        self.intensity = max(0.0, min(1.0, intensity))
        
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply effect to frame."""
        raise NotImplementedError

class ColorEnhance(VisualEffect):
    """Enhance colors in frame."""
    
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Enhance saturation
        hsv[:, :, 1] = hsv[:, :, 1] * (1 + self.intensity * 0.5)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back to BGR
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

class Sharpness(VisualEffect):
    """Adjust image sharpness."""
    
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]]) * self.intensity
        return cv2.filter2D(frame, -1, kernel)

class Vignette(VisualEffect):
    """Add vignette effect."""
    
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        rows, cols = frame.shape[:2]
        
        # Generate vignette mask
        X_resultant, Y_resultant = np.meshgrid(np.linspace(-1, 1, cols),
                                             np.linspace(-1, 1, rows))
        mask = np.sqrt(X_resultant**2 + Y_resultant**2)
        mask = 1 - mask
        mask = np.clip(mask + (1 - self.intensity), 0, 1)
        
        # Expand mask to match frame channels
        mask = np.dstack([mask] * 3)
        
        return (frame * mask).astype(np.uint8)

class FilmGrain(VisualEffect):
    """Add film grain effect."""
    
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, 25 * self.intensity, frame.shape).astype(np.uint8)
        return cv2.add(frame, noise)

class CinematicBars(VisualEffect):
    """Add cinematic bars effect."""
    
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        height = frame.shape[0]
        bar_height = int(height * 0.1 * self.intensity)
        
        frame[:bar_height] = 0  # Top bar
        frame[-bar_height:] = 0  # Bottom bar
        return frame

class EffectLibrary:
    """Library of available visual effects."""
    
    def __init__(self):
        self.effects = {
            'color_enhance': ColorEnhance,
            'sharpness': Sharpness,
            'vignette': Vignette,
            'film_grain': FilmGrain,
            'cinematic_bars': CinematicBars
        }
        
    def get_effect(self, name: str, intensity: float = 1.0) -> VisualEffect:
        """Get effect by name."""
        if name not in self.effects:
            raise ValueError(f"Unknown effect: {name}")
            
        return self.effects[name](intensity)
        
    def list_effects(self) -> List[str]:
        """List available effects."""
        return list(self.effects.keys())
        
    def get_effect_info(self, name: str) -> Dict[str, Any]:
        """Get information about an effect."""
        if name not in self.effects:
            raise ValueError(f"Unknown effect: {name}")
            
        effect_class = self.effects[name]
        return {
            'name': name,
            'description': effect_class.__doc__,
            'parameters': {
                'intensity': 'Effect intensity (0-1)'
            }
        }

class EffectChain:
    """Chain multiple visual effects together."""
    
    def __init__(self):
        self.effects: List[VisualEffect] = []
        self.library = EffectLibrary()
        
    def add_effect(self, name: str, intensity: float = 1.0) -> 'EffectChain':
        """Add effect to chain."""
        effect = self.library.get_effect(name, intensity)
        self.effects.append(effect)
        return self
        
    async def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply all effects in chain."""
        result = frame.copy()
        for effect in self.effects:
            result = await effect.apply(result)
        return result
        
    def clear(self) -> None:
        """Clear all effects from chain."""
        self.effects.clear()
```

### ./src/director/effects/transitions.py
```python
from typing import Dict, Any, List, Optional
import numpy as np
from pathlib import Path
import cv2

class TransitionEffect:
    """Base class for video transitions."""
    
    def __init__(self, duration: float = 0.5):
        self.duration = duration
        
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        """Apply transition effect between two frames."""
        raise NotImplementedError

class CrossDissolve(TransitionEffect):
    """Smooth fade transition between frames."""
    
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        return cv2.addWeighted(frame1, 1 - progress, frame2, progress, 0)

class SlideTransition(TransitionEffect):
    """Slide one frame over another."""
    
    def __init__(self, direction: str = 'left', duration: float = 0.5):
        super().__init__(duration)
        self.direction = direction
        
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        height, width = frame1.shape[:2]
        result = np.zeros_like(frame1)
        
        if self.direction == 'left':
            offset = int(width * progress)
            result[:, :width-offset] = frame1[:, offset:]
            result[:, width-offset:] = frame2[:, :offset]
        elif self.direction == 'right':
            offset = int(width * progress)
            result[:, offset:] = frame1[:, :width-offset]
            result[:, :offset] = frame2[:, width-offset:]
        
        return result

class ZoomTransition(TransitionEffect):
    """Zoom in/out transition effect."""
    
    def __init__(self, zoom_in: bool = True, duration: float = 0.5):
        super().__init__(duration)
        self.zoom_in = zoom_in
        
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        if self.zoom_in:
            scale = 1 + progress
            frame1_scaled = cv2.resize(frame1, None, fx=scale, fy=scale)
            h1, w1 = frame1_scaled.shape[:2]
            y1 = int((h1 - frame1.shape[0]) / 2)
            x1 = int((w1 - frame1.shape[1]) / 2)
            frame1_crop = frame1_scaled[y1:y1+frame1.shape[0], x1:x1+frame1.shape[1]]
            return cv2.addWeighted(frame1_crop, 1 - progress, frame2, progress, 0)
        else:
            scale = 2 - progress
            frame2_scaled = cv2.resize(frame2, None, fx=scale, fy=scale)
            h2, w2 = frame2_scaled.shape[:2]
            y2 = int((h2 - frame2.shape[0]) / 2)
            x2 = int((w2 - frame2.shape[1]) / 2)
            frame2_crop = frame2_scaled[y2:y2+frame2.shape[0], x2:x2+frame2.shape[1]]
            return cv2.addWeighted(frame1, 1 - progress, frame2_crop, progress, 0)

class WhipPanTransition(TransitionEffect):
    """Fast whip pan transition effect."""
    
    async def apply(self, frame1: np.ndarray, frame2: np.ndarray, progress: float) -> np.ndarray:
        if progress < 0.5:
            # Apply motion blur to first frame
            kernel_size = int((progress * 2) * 50)
            if kernel_size > 0:
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
                kernel = kernel / kernel_size
                frame1 = cv2.filter2D(frame1, -1, kernel)
            return frame1
        else:
            # Apply motion blur to second frame
            kernel_size = int((1 - progress) * 2 * 50)
            if kernel_size > 0:
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
                kernel = kernel / kernel_size
                frame2 = cv2.filter2D(frame2, -1, kernel)
            return frame2

class TransitionLibrary:
    """Library of available transition effects."""
    
    def __init__(self):
        self.transitions = {
            'dissolve': CrossDissolve,
            'slide_left': lambda: SlideTransition('left'),
            'slide_right': lambda: SlideTransition('right'),
            'zoom_in': lambda: ZoomTransition(True),
            'zoom_out': lambda: ZoomTransition(False),
            'whip_pan': WhipPanTransition
        }
        
    def get_transition(self, name: str, **kwargs) -> TransitionEffect:
        """Get transition effect by name."""
        if name not in self.transitions:
            raise ValueError(f"Unknown transition effect: {name}")
            
        transition_class = self.transitions[name]
        if callable(transition_class):
            return transition_class()
        return transition_class(**kwargs)
        
    def list_transitions(self) -> List[str]:
        """List available transition effects."""
        return list(self.transitions.keys())
        
    def get_transition_info(self, name: str) -> Dict[str, Any]:
        """Get information about a transition effect."""
        if name not in self.transitions:
            raise ValueError(f"Unknown transition effect: {name}")
            
        transition_class = self.transitions[name]
        return {
            'name': name,
            'description': transition_class.__doc__,
            'parameters': {
                'duration': 'Duration of transition in seconds'
            }
        }
```

### ./src/director/effects/text_overlay.py
```python
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from pathlib import Path
import json

class TextStyle:
    """Text styling configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.font_face = config.get('font_face', cv2.FONT_HERSHEY_DUPLEX)
        self.font_scale = config.get('font_scale', 1.0)
        self.color = config.get('color', (255, 255, 255))
        self.thickness = config.get('thickness', 2)
        self.background_color = config.get('background_color')
        self.padding = config.get('padding', 10)
        self.shadow = config.get('shadow', True)
        self.shadow_color = config.get('shadow_color', (0, 0, 0))
        self.animation = config.get('animation', 'fade')

class TextOverlay:
    """Dynamic text overlay system for videos."""
    
    def __init__(self, style: Optional[Dict[str, Any]] = None):
        self.style = TextStyle(style or {})
        self._load_fonts()
        
    def _load_fonts(self):
        """Load custom fonts if available."""
        # TODO: Implement custom font loading
        pass
        
    async def render_text(self,
                         frame: np.ndarray,
                         text: str,
                         position: Tuple[float, float],
                         progress: float = 1.0,
                         style: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Render text overlay on frame.
        
        Args:
            frame: Input video frame
            text: Text to overlay
            position: Position as (x, y) in relative coordinates (0-1)
            progress: Animation progress (0-1)
            style: Optional style override
            
        Returns:
            Frame with text overlay
        """
        text_style = TextStyle(style) if style else self.style
        frame_h, frame_w = frame.shape[:2]
        
        # Convert relative position to absolute pixels
        x = int(position[0] * frame_w)
        y = int(position[1] * frame_h)
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            text_style.font_face,
            text_style.font_scale,
            text_style.thickness
        )
        
        # Apply animation
        if text_style.animation == 'fade':
            alpha = progress
        elif text_style.animation == 'slide':
            x = int(x + (frame_w * (1 - progress)))
        elif text_style.animation == 'pop':
            text_style.font_scale *= progress
            
        # Draw background if specified
        if text_style.background_color:
            bg_x1 = x - text_style.padding
            bg_y1 = y - text_h - text_style.padding
            bg_x2 = x + text_w + text_style.padding
            bg_y2 = y + text_style.padding
            
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (bg_x1, bg_y1),
                (bg_x2, bg_y2),
                text_style.background_color,
                -1
            )
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
            
        # Draw shadow if enabled
        if text_style.shadow:
            shadow_offset = max(1, int(text_style.thickness / 2))
            cv2.putText(
                frame,
                text,
                (x + shadow_offset, y + shadow_offset),
                text_style.font_face,
                text_style.font_scale,
                text_style.shadow_color,
                text_style.thickness
            )
            
        # Draw text
        cv2.putText(
            frame,
            text,
            (x, y),
            text_style.font_face,
            text_style.font_scale,
            text_style.color,
            text_style.thickness
        )
        
        return frame

class TextTemplate:
    """Predefined text overlay templates."""
    
    def __init__(self, template_path: Optional[str] = None):
        self.templates = self._load_templates(template_path)
        
    def _load_templates(self, template_path: Optional[str]) -> Dict[str, Any]:
        """Load text templates from JSON file."""
        if not template_path:
            return self._default_templates()
            
        template_path = Path(template_path)
        if template_path.exists():
            with open(template_path, 'r') as f:
                return json.load(f)
                
        return self._default_templates()
        
    def _default_templates(self) -> Dict[str, Any]:
        """Default text overlay templates."""
        return {
            'price': {
                'style': {
                    'font_scale': 1.5,
                    'color': (0, 255, 0),
                    'background_color': (0, 0, 0),
                    'animation': 'pop'
                },
                'position': (0.1, 0.9)
            },
            'feature': {
                'style': {
                    'font_scale': 1.0,
                    'color': (255, 255, 255),
                    'background_color': None,
                    'animation': 'slide'
                },
                'position': (0.05, 0.8)
            },
            'call_to_action': {
                'style': {
                    'font_scale': 1.2,
                    'color': (255, 255, 0),
                    'background_color': (0, 0, 0),
                    'animation': 'fade'
                },
                'position': (0.5, 0.5)
            }
        }
        
    def get_template(self, name: str) -> Dict[str, Any]:
        """Get template by name."""
        if name not in self.templates:
            raise ValueError(f"Unknown template: {name}")
        return self.templates[name]
        
    def list_templates(self) -> List[str]:
        """List available templates."""
        return list(self.templates.keys())
```

### ./src/director/assets/videodb.py
```python
from typing import Dict, Any, List, Optional
from pathlib import Path
import os

class VideoDBToolWrapper:
    """
    Wrapper for Director's VideoDBTool, providing video processing capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # TODO: Initialize VideoDBTool from Director
        
    async def process_image(self,
                          image_path: str,
                          target_resolution: tuple = (1080, 1920),
                          effects: List[Dict[str, Any]] = None) -> str:
        """
        Process a single image using VideoDBTool.
        Args:
            image_path: Path to input image
            target_resolution: Desired resolution (width, height)
            effects: List of effects to apply
        Returns:
            Path to processed image
        """
        # TODO: Implement image processing with VideoDBTool
        return image_path
        
    async def create_video_clip(self,
                              images: List[str],
                              duration: float,
                              transitions: List[Dict[str, Any]] = None) -> str:
        """
        Create a video clip from images.
        Args:
            images: List of image paths
            duration: Clip duration in seconds
            transitions: List of transition effects
        Returns:
            Path to generated video clip
        """
        # TODO: Implement video clip creation with VideoDBTool
        return ""
        
    async def add_audio_track(self,
                            video_path: str,
                            audio_path: str,
                            volume: float = 1.0) -> str:
        """
        Add audio track to video.
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            volume: Audio volume level
        Returns:
            Path to video with audio
        """
        # TODO: Implement audio addition with VideoDBTool
        return video_path
        
    async def add_text_overlay(self,
                             video_path: str,
                             text: str,
                             style: Dict[str, Any]) -> str:
        """
        Add text overlay to video.
        Args:
            video_path: Path to video file
            text: Text to overlay
            style: Text style parameters
        Returns:
            Path to video with overlay
        """
        # TODO: Implement text overlay with VideoDBTool
        return video_path
        
    async def apply_effects(self,
                          video_path: str,
                          effects: List[Dict[str, Any]]) -> str:
        """
        Apply video effects.
        Args:
            video_path: Path to video file
            effects: List of effects to apply
        Returns:
            Path to processed video
        """
        # TODO: Implement effects application with VideoDBTool
        return video_path
```

### ./src/director/assets/__init__.py
```python
```

### ./src/director/assets/manager.py
```python
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import shutil
import asyncio
from datetime import datetime
import json

class AssetManager:
    """
    Manages assets for video generation, integrating with VideoDBTool.
    Handles: images, audio (music/voiceover), video clips, and temporary files.
    """
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or 'assets')
        self._init_directories()
        
    def _init_directories(self) -> None:
        """Initialize directory structure for assets."""
        directories = [
            'images/original',    # Original listing images
            'images/processed',   # Processed images (resized, effects)
            'audio/music',        # Background music
            'audio/voiceover',    # Generated voiceovers
            'video/clips',        # Individual video clips
            'video/final',        # Final rendered videos
            'temp'               # Temporary processing files
        ]
        
        for dir_path in directories:
            (self.base_path / dir_path).mkdir(parents=True, exist_ok=True)
            
    async def process_listing_images(self, 
                                   session_id: str, 
                                   images: List[str],
                                   target_resolution: tuple = (1080, 1920)) -> List[str]:
        """
        Process listing images for video generation.
        Args:
            session_id: Unique session identifier
            images: List of image paths or URLs
            target_resolution: Target resolution for processed images (width, height)
        Returns:
            List of processed image paths
        """
        # TODO: Implement image processing with VideoDBTool
        processed_paths = []
        return processed_paths
        
    async def store_voiceover(self, 
                             session_id: str, 
                             audio_data: bytes,
                             metadata: Dict[str, Any]) -> str:
        """
        Store generated voiceover audio.
        Args:
            session_id: Unique session identifier
            audio_data: Raw audio data
            metadata: Audio metadata (duration, format, etc.)
        Returns:
            Path to stored audio file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{session_id}_{timestamp}.mp3"
        filepath = self.base_path / 'audio/voiceover' / filename
        
        # Save audio data
        with open(filepath, 'wb') as f:
            f.write(audio_data)
            
        # Save metadata
        meta_path = filepath.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
            
        return str(filepath)
        
    async def get_background_music(self, 
                                 style: str = 'upbeat',
                                 duration: int = 60) -> Optional[str]:
        """
        Get appropriate background music for the video.
        Args:
            style: Music style/mood
            duration: Required duration in seconds
        Returns:
            Path to music file
        """
        # TODO: Implement music selection logic
        return None
        
    async def create_temp_directory(self, session_id: str) -> Path:
        """Create a temporary directory for processing."""
        temp_dir = self.base_path / 'temp' / session_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
        
    async def cleanup_temp_files(self, session_id: str) -> None:
        """Clean up temporary files for a session."""
        temp_dir = self.base_path / 'temp' / session_id
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
    async def store_final_video(self, 
                              session_id: str,
                              video_path: str,
                              metadata: Dict[str, Any]) -> str:
        """
        Store the final generated video.
        Args:
            session_id: Unique session identifier
            video_path: Path to generated video
            metadata: Video metadata
        Returns:
            Path to stored video
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{session_id}_{timestamp}.mp4"
        final_path = self.base_path / 'video/final' / filename
        
        # Copy video to final location
        shutil.copy2(video_path, final_path)
        
        # Save metadata
        meta_path = final_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
            
        return str(final_path)
```

### ./src/__init__.py
```python

```

### ./src/utils/logging.py
```python
import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path("/tmp/video_service/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log")
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    return logger
```

### ./src/api/models.py
```python
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, HttpUrl, field_validator, ConfigDict
import re

class ListingSpecs(BaseModel):
    """Specifications for a real estate listing."""
    bedrooms: Optional[int] = Field(None, description="Number of bedrooms")
    bathrooms: Optional[float] = Field(None, description="Number of bathrooms")
    square_feet: Optional[int] = Field(None, description="Square footage")
    lot_size: Optional[str] = Field(None, description="Lot size")
    year_built: Optional[int] = Field(None, description="Year built")
    property_type: Optional[str] = Field(None, description="Type of property")
    
    model_config = ConfigDict(extra='allow')  # Allow additional fields

class ListingData(BaseModel):
    """Real estate listing data."""
    title: str = Field(..., min_length=1, max_length=200)
    price: float = Field(..., gt=0)
    description: str = Field(..., min_length=1)
    features: List[str] = Field(default_factory=list)
    location: str = Field(..., min_length=1)
    specs: ListingSpecs
    
    @field_validator('price')
    @classmethod
    def validate_price(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Price must be greater than 0")
        return v
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v: List[str]) -> List[str]:
        if not all(isinstance(f, str) and len(f.strip()) > 0 for f in v):
            raise ValueError("All features must be non-empty strings")
        return v

class VideoGenerationRequest(BaseModel):
    """Request model for video generation."""
    listing: ListingData
    images: List[str] = Field(..., min_length=1)
    
    @field_validator('images')
    @classmethod
    def validate_images(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one image is required")
            
        for img in v:
            # Check if it's a URL
            if img.startswith(('http://', 'https://')):
                continue
                
            # Check if it's base64
            try:
                pattern = r'^data:image/[a-zA-Z]+;base64,'
                if not re.match(pattern, img):
                    raise ValueError
            except ValueError:
                raise ValueError("Images must be valid URLs or base64 encoded")
                
        return v

class VideoGenerationResponse(BaseModel):
    """Response model for video generation."""
    task_id: str = Field(..., description="Task ID for tracking generation progress")
    estimated_duration: int = Field(..., description="Estimated duration in seconds")
    template_name: str = Field(..., description="Selected template name")
    status: str = Field(..., description="Current status of the generation")

class GenerationStatus(BaseModel):
    """Status model for video generation task."""
    task_id: str
    status: str = Field(..., description="Status: pending, processing, completed, failed")
    progress: float = Field(..., ge=0, le=100, description="Progress percentage")
    result_url: Optional[str] = None
    error: Optional[str] = None
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid_statuses = {'pending', 'processing', 'completed', 'failed'}
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return v
```

### ./src/api/endpoints.py
```python
from typing import Dict, Any
import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import asyncio
import uuid
from datetime import datetime, UTC
import json
import os
from pathlib import Path

from .models import (
    VideoGenerationRequest,
    VideoGenerationResponse,
    GenerationStatus
)
from ..director.variation.template_engine import TemplateLibrary
from ..director.video_generator import VideoGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
LOGS_DIR = Path('logs')
LOGS_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Real Estate Video Generation API",
    description="Internal API for generating TikTok-style real estate videos",
    version="1.0.0"
)

def get_task_file(task_id: str) -> Path:
    """Get the path to the task's JSON file."""
    return LOGS_DIR / f"task_{task_id}.json"

def save_task_status(task_id: str, status: dict) -> None:
    """Save task status to file."""
    try:
        with open(get_task_file(task_id), 'w') as f:
            json.dump(status, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save task status: {e}")

@app.post("/api/v1/videos/generate",
          summary="Generate a real estate video",
          response_description="Returns task information for the video generation")
async def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks
) -> VideoGenerationResponse:
    """
    Generate a real estate video from listing data and images.
    
    For internal use by developers and cron jobs only.
    The video will be generated asynchronously, and a task ID will be returned
    for checking the generation status.
    
    Args:
        request: Video generation request containing listing data and images
        background_tasks: FastAPI background tasks handler
        
    Returns:
        VideoGenerationResponse with task information
        
    Raises:
        HTTPException: If the request is invalid or generation fails
    """
    try:
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        logger.info(f"Starting video generation task {task_id}")
        
        # Select template
        template_library = TemplateLibrary()
        templates = template_library.list_templates()
        
        # TODO: Implement smart template selection based on listing data
        template_name = templates[0]
        
        # Initialize task status
        task_status = {
            "status": "pending",
            "progress": 0,
            "start_time": datetime.now(UTC),
            "template": template_name,
            "result_url": None,
            "error": None,
            "request": request.model_dump()
        }
        
        # Save initial status
        save_task_status(task_id, task_status)
        
        # Start generation in background
        background_tasks.add_task(
            _generate_video_task,
            task_id,
            request,
            template_name
        )
        
        return VideoGenerationResponse(
            task_id=task_id,
            estimated_duration=60,  # TODO: Calculate based on template
            template_name=template_name,
            status="pending"
        )
        
    except Exception as e:
        logger.error(f"Failed to start video generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start video generation: {str(e)}"
        )

@app.get("/api/v1/videos/{task_id}/status",
        response_model=GenerationStatus,
        summary="Get video generation status",
        response_description="Returns the current status of the video generation task")
async def get_generation_status(task_id: str) -> GenerationStatus:
    """
    Get the status of a video generation task.
    
    Args:
        task_id: Task ID returned from generate_video endpoint
        
    Returns:
        GenerationStatus with current status and progress
        
    Raises:
        HTTPException: If task ID is not found
    """
    task_file = get_task_file(task_id)
    if not task_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    try:
        with open(task_file) as f:
            task = json.load(f)
            
        return GenerationStatus(
            task_id=task_id,
            status=task["status"],
            progress=task["progress"],
            result_url=task.get("result_url"),
            error=task.get("error")
        )
    except Exception as e:
        logger.error(f"Failed to read task status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read task status: {str(e)}"
        )

async def _generate_video_task(
    task_id: str,
    request: VideoGenerationRequest,
    template_name: str
) -> None:
    """
    Background task for video generation.
    
    Args:
        task_id: Task ID for updating status
        request: Video generation request
        template_name: Selected template name
    """
    logger.info(f"Starting video generation for task {task_id}")
    try:
        # Update status to processing
        task_status = {
            "status": "processing",
            "progress": 0,
            "start_time": datetime.now(UTC),
            "template": template_name,
            "result_url": None,
            "error": None,
            "request": request.model_dump()
        }
        save_task_status(task_id, task_status)
        
        # Initialize video generator
        generator = VideoGenerator()
        
        # Generate video
        result_url = await generator.generate(
            template_name=template_name,
            listing_data=request.listing,
            images=request.images,
            progress_callback=lambda p: _update_progress(task_id, p)
        )
        
        # Update task with success
        task_status.update({
            "status": "completed",
            "progress": 100,
            "result_url": result_url,
            "completion_time": datetime.now(UTC)
        })
        save_task_status(task_id, task_status)
        logger.info(f"Video generation completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"Video generation failed for task {task_id}: {e}", exc_info=True)
        # Update task with error
        task_status = {
            "status": "failed",
            "error": str(e),
            "error_time": datetime.now(UTC)
        }
        save_task_status(task_id, task_status)

def _update_progress(task_id: str, progress: float) -> None:
    """Update task progress."""
    try:
        task_file = get_task_file(task_id)
        if task_file.exists():
            with open(task_file) as f:
                task_status = json.load(f)
            
            task_status["progress"] = progress
            save_task_status(task_id, task_status)
            logger.debug(f"Updated progress for task {task_id}: {progress}%")
    except Exception as e:
        logger.error(f"Failed to update progress for task {task_id}: {e}")
```

### ./src/api/routes.py
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Annotated
from src.director.core.reasoning_engine import ReasoningEngine
from src.director.core.session_manager import SessionManager
from src.utils.logging import setup_logger

# Set up logger
logger = setup_logger("api")

app = FastAPI(title="Real Estate Video Generator API")

class PropertySpecs(BaseModel):
    bedrooms: Annotated[int, Field(description="Number of bedrooms")]
    bathrooms: Annotated[float, Field(description="Number of bathrooms")]
    square_feet: Annotated[int, Field(description="Square footage of the property")]
    year_built: Optional[Annotated[int, Field(description="Year the property was built")]] = None
    lot_size: Optional[Annotated[int, Field(description="Lot size in square feet")]] = None

class PropertyListing(BaseModel):
    title: Annotated[str, Field(description="Property title")]
    price: Annotated[int, Field(description="Property price")]
    description: Annotated[str, Field(description="Property description")]
    features: Annotated[List[str], Field(description="List of property features")]
    location: Annotated[str, Field(description="Property location")]
    specs: Annotated[PropertySpecs, Field(description="Property specifications")]

class VideoRequest(BaseModel):
    listing: PropertyListing
    images: Annotated[List[str], Field(description="List of image URLs", min_items=1)]

class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Initialize core components
engine = ReasoningEngine()
session_manager = SessionManager()

@app.post("/api/v1/videos/generate", response_model=TaskStatus)
async def generate_video(request: VideoRequest) -> TaskStatus:
    """
    Generate a real estate video from property listing and images.
    
    Args:
        request: Video generation request containing property listing and image URLs
        
    Returns:
        Task ID and initial status
    """
    try:
        # Create new session with UUID
        import uuid
        session_id = str(uuid.uuid4())
        logger.info(f"Creating new session with ID: {session_id}")
        session = session_manager.create_session(session_id)
        
        # Log request details
        logger.info(f"Processing video generation request for property: {request.listing.title}")
        logger.debug(f"Full request data: {request.model_dump_json()}")
        
        # Transform listing data
        listing_data = {
            "id": str(uuid.uuid4()),  # Generate a unique ID
            "title": request.listing.title,
            "price": request.listing.price,
            "description": request.listing.description,
            "features": request.listing.features,
            "location": request.listing.location,
            "bedrooms": request.listing.specs.bedrooms,
            "bathrooms": request.listing.specs.bathrooms,
            "sqft": request.listing.specs.square_feet,
            "year_built": request.listing.specs.year_built,
            "lot_size": request.listing.specs.lot_size
        }
        
        # Start video generation
        logger.info("Starting video generation process...")
        task = await engine.generate_video(
            session_id=session_id,
            listing_data=listing_data,
            image_urls=request.images,
            style="modern",  # Default style
            duration=30.0  # Default duration
        )
        logger.info(f"Video generation task created with ID: {task.id}")
        
        return TaskStatus(
            task_id=task.id,
            status="processing",
            progress=0.0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/videos/status/{task_id}", response_model=TaskStatus)
async def get_video_status(task_id: str) -> TaskStatus:
    """
    Get the status of a video generation task.
    
    Args:
        task_id: ID of the task to check
        
    Returns:
        Current task status and progress
    """
    try:
        logger.info(f"Checking status for task: {task_id}")
        # Get task status from session manager
        task = session_manager.get_task(task_id)
        if not task:
            logger.warning(f"Task not found: {task_id}")
            raise HTTPException(status_code=404, detail="Task not found")
        
        logger.info(f"Task {task_id} status: {task.status}, progress: {task.progress}")
            
        return TaskStatus(
            task_id=task_id,
            status=task.status,
            progress=task.progress,
            result=task.result if task.status == "completed" else None,
            error=task.error if task.status == "failed" else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### ./src/main.py
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Real Estate TikTok Video Generator",
    description="Director-based service for generating TikTok-style real estate videos",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from api.routes import app as api_app

# Include API routes
app.include_router(api_app.router)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("main:app", host=host, port=port, reload=True)
```

