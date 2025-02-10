from moviepy.editor import ColorClip
from src.services.text_overlay_service import TextOverlayService
from src.models.text_overlay import (
    TextOverlay, TextPosition, TextAlignment,
    AnimationType, TextStyle, Animation, Templates
)


def test_text_overlays():
    # Create a test video clip (black background)
    base_clip = ColorClip(size=(1080, 1920), color=(0, 0, 0), duration=10)
    
    # Create text overlays
    overlays = [
        TextOverlay(
            text="Luxury Villa",
            position=TextPosition.TOP,
            style=Templates.PROPERTY_TITLE.style,
            animation=Animation(type=AnimationType.FADE, duration=1.0),
            start_time=0.5
        ),
        TextOverlay(
            text="$1,250,000",
            position=TextPosition.MIDDLE,
            style=Templates.PRICE_TAG.style,
            animation=Animation(type=AnimationType.SCALE, duration=0.8),
            start_time=1.5
        ),
        TextOverlay(
            text="🏠 4 Beds  🛁 3 Baths",
            position=TextPosition.BOTTOM,
            style=Templates.FEATURE_LIST.style,
            animation=Animation(type=AnimationType.SLIDE_LEFT, duration=0.8),
            start_time=2.5
        ),
        TextOverlay(
            text="📍 Beverly Hills",
            position=TextPosition.BOTTOM,
            style=Templates.LOCATION.style,
            animation=Animation(type=AnimationType.SLIDE_UP, duration=0.8),
            start_time=3.5
        )
    ]
    
    # Apply overlays
    text_service = TextOverlayService()
    final_clip = text_service.apply_text_overlays(base_clip, overlays)
    
    # Write output
    final_clip.write_videofile(
        "test_overlays.mp4",
        codec='libx264',
        audio=False,
        fps=24
    )


if __name__ == "__main__":
    test_text_overlays()
