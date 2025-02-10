from typing import Optional, Tuple, List, Literal
from pydantic import BaseModel, Field
from enum import Enum


class TextPosition(str, Enum):
    TOP = "top"
    MIDDLE = "middle"
    BOTTOM = "bottom"
    CUSTOM = "custom"


class TextAlignment(str, Enum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class AnimationType(str, Enum):
    NONE = "none"
    FADE = "fade"
    SLIDE_LEFT = "slide_left"
    SLIDE_RIGHT = "slide_right"
    SLIDE_UP = "slide_up"
    SLIDE_DOWN = "slide_down"
    SCALE = "scale"
    TYPE = "type"


class TextStyle(BaseModel):
    font_family: str = Field(default="Arial")
    font_size: int = Field(default=32)
    color: Tuple[int, int, int] = Field(default=(255, 255, 255))  # RGB
    stroke_width: int = Field(default=2)
    stroke_color: Tuple[int, int, int] = Field(default=(0, 0, 0))  # RGB
    background_color: Optional[Tuple[int, int, int]] = Field(default=None)  # RGB
    background_opacity: float = Field(default=0.0, ge=0.0, le=1.0)
    shadow: bool = Field(default=True)
    shadow_color: Tuple[int, int, int] = Field(default=(0, 0, 0))  # RGB
    shadow_offset: Tuple[int, int] = Field(default=(2, 2))  # x, y offset


class Animation(BaseModel):
    type: AnimationType = Field(default=AnimationType.NONE)
    duration: float = Field(default=1.0)  # seconds
    delay: float = Field(default=0.0)  # seconds
    easing: str = Field(default="linear")


class TextOverlay(BaseModel):
    text: str
    position: TextPosition = Field(default=TextPosition.BOTTOM)
    custom_position: Optional[Tuple[int, int]] = Field(default=None)  # x, y coordinates
    alignment: TextAlignment = Field(default=TextAlignment.CENTER)
    style: TextStyle = Field(default_factory=TextStyle)
    animation: Animation = Field(default_factory=Animation)
    start_time: float = Field(default=0.0)  # seconds
    end_time: Optional[float] = Field(default=None)  # seconds
    emoji_scale: float = Field(default=1.0)  # relative to font size


class TextTemplate(BaseModel):
    name: str
    style: TextStyle
    animation: Animation
    position: TextPosition = Field(default=TextPosition.BOTTOM)
    alignment: TextAlignment = Field(default=TextAlignment.CENTER)


# Pre-defined templates
class Templates:
    PROPERTY_TITLE = TextTemplate(
        name="property_title",
        style=TextStyle(
            font_size=48,
            stroke_width=3,
            shadow=True,
        ),
        animation=Animation(
            type=AnimationType.FADE,
            duration=0.8
        ),
        position=TextPosition.TOP,
    )
    
    PRICE_TAG = TextTemplate(
        name="price_tag",
        style=TextStyle(
            font_size=42,
            color=(50, 205, 50),  # Green
            stroke_width=3,
            background_color=(0, 0, 0),
            background_opacity=0.7,
        ),
        animation=Animation(
            type=AnimationType.SCALE,
            duration=0.5
        ),
        position=TextPosition.MIDDLE,
    )
    
    FEATURE_LIST = TextTemplate(
        name="feature_list",
        style=TextStyle(
            font_size=36,
            stroke_width=2,
            background_color=(0, 0, 0),
            background_opacity=0.5,
        ),
        animation=Animation(
            type=AnimationType.SLIDE_LEFT,
            duration=0.5
        ),
        position=TextPosition.BOTTOM,
    )
    
    LOCATION = TextTemplate(
        name="location",
        style=TextStyle(
            font_size=38,
            stroke_width=2,
            background_color=(0, 0, 0),
            background_opacity=0.6,
        ),
        animation=Animation(
            type=AnimationType.SLIDE_UP,
            duration=0.6
        ),
        position=TextPosition.BOTTOM,
    )
