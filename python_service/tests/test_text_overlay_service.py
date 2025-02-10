import pytest
from moviepy.editor import ColorClip
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.services.text_overlay_service import TextOverlayService
from src.models.text_overlay import (
    TextOverlay, TextPosition, TextAlignment,
    AnimationType, TextStyle, Animation, Templates
)


@pytest.fixture
def text_service():
    return TextOverlayService()


@pytest.fixture
def base_clip():
    return ColorClip(size=(1080, 1920), color=(0, 0, 0), duration=5)


def test_basic_text_overlay(text_service, base_clip):
    overlay = TextOverlay(
        text="Test Text",
        position=TextPosition.MIDDLE,
        alignment=TextAlignment.CENTER,
        style=TextStyle(
            font_size=48,
            color=(255, 255, 255)
        )
    )
    
    result = text_service.apply_text_overlays(base_clip, [overlay])
    assert result.duration == base_clip.duration
    assert result.w == base_clip.w
    assert result.h == base_clip.h


def test_emoji_support(text_service, base_clip):
    overlay = TextOverlay(
        text="House: :house: Bath: :bathtub:",
        position=TextPosition.BOTTOM,
        style=TextStyle(font_size=42)
    )
    
    result = text_service.apply_text_overlays(base_clip, [overlay])
    assert result.duration == base_clip.duration


def test_animated_text(text_service, base_clip):
    overlay = TextOverlay(
        text="Animated Text",
        animation=Animation(
            type=AnimationType.FADE,
            duration=1.0
        )
    )
    
    result = text_service.apply_text_overlays(base_clip, [overlay])
    assert result.duration == base_clip.duration


def test_multiple_overlays(text_service, base_clip):
    overlays = [
        TextOverlay(
            text="Title",
            position=TextPosition.TOP,
            style=Templates.PROPERTY_TITLE.style
        ),
        TextOverlay(
            text="$500,000",
            position=TextPosition.MIDDLE,
            style=Templates.PRICE_TAG.style
        ),
        TextOverlay(
            text="Location: :round_pushpin: Downtown",
            position=TextPosition.BOTTOM,
            style=Templates.LOCATION.style
        )
    ]
    
    result = text_service.apply_text_overlays(base_clip, overlays)
    assert result.duration == base_clip.duration


def test_text_timing(text_service, base_clip):
    overlay = TextOverlay(
        text="Timed Text",
        start_time=1.0,
        end_time=3.0
    )
    
    result = text_service.apply_text_overlays(base_clip, [overlay])
    assert result.duration == base_clip.duration


def test_template_usage(text_service, base_clip):
    overlay = TextOverlay(
        text="Luxury Villa",
        style=Templates.PROPERTY_TITLE.style,
        position=Templates.PROPERTY_TITLE.position,
        animation=Templates.PROPERTY_TITLE.animation
    )
    
    result = text_service.apply_text_overlays(base_clip, [overlay])
    assert result.duration == base_clip.duration


def test_custom_position(text_service, base_clip):
    overlay = TextOverlay(
        text="Custom Position",
        position=TextPosition.CUSTOM,
        custom_position=(100, 100)
    )
    
    result = text_service.apply_text_overlays(base_clip, [overlay])
    assert result.duration == base_clip.duration


def test_background_text(text_service, base_clip):
    overlay = TextOverlay(
        text="Text with Background",
        style=TextStyle(
            background_color=(0, 0, 0),
            background_opacity=0.7
        )
    )
    
    result = text_service.apply_text_overlays(base_clip, [overlay])
    assert result.duration == base_clip.duration
