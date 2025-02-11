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
