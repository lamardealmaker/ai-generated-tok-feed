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
