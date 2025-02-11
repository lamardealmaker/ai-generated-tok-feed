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
