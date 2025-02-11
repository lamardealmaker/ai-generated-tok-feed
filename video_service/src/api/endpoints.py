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
