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
