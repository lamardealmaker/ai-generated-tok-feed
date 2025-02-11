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
