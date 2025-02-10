from fastapi import APIRouter, UploadFile, File, HTTPException
from src.services.video_service import VideoService
from typing import Dict

router = APIRouter()
video_service = VideoService()

@router.post("/videos/generate")
async def generate_video(
    properties: Dict,
    background_music: UploadFile = File(None)
) -> Dict:
    """
    Generate a video from property data and optional background music.
    
    Args:
        properties: Dictionary containing property details
        background_music: Optional background music file
    
    Returns:
        Dict containing the video URL and any additional metadata
    """
    try:
        result = await video_service.generate_video(properties, background_music)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/videos/{video_id}")
async def get_video_status(video_id: str) -> Dict:
    """
    Get the status of a video generation task.
    
    Args:
        video_id: The ID of the video generation task
    
    Returns:
        Dict containing the status and any available results
    """
    try:
        status = await video_service.get_video_status(video_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
