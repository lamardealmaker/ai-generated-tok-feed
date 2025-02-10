from typing import Dict, Optional, List
from fastapi import UploadFile
import moviepy.editor as mp
from firebase_admin import storage, firestore
import uuid
import tempfile
import os
from .image_processor import ImageProcessor
from .audio_service import AudioService
from .text_overlay_service import TextOverlayService
from ..models.text_overlay import (
    TextOverlay, TextPosition, TextAlignment,
    AnimationType, TextStyle, Animation, Templates
)

class VideoService:
    def __init__(self):
        self.bucket = storage.bucket()
        self.db = firestore.client()
        self.image_processor = ImageProcessor()
        self.audio_service = AudioService()
        self.text_overlay_service = TextOverlayService()
        
    async def generate_video(
        self,
        properties: Dict,
        background_music: Optional[UploadFile] = None
    ) -> Dict:
        """
        Generate a video from property data and optional background music.
        
        Args:
            properties: Dictionary containing property details including:
                - images: List[str] of image URLs
                - title: str
                - price: str
                - location: str
            background_music: Optional background music file
        
        Returns:
            Dict containing the video URL and metadata
        """
        try:
            # Generate a unique ID for this video
            video_id = str(uuid.uuid4())
            
            # Create a document in Firestore to track progress
            self.db.collection('videos').document(video_id).set({
                'status': 'processing',
                'created_at': firestore.SERVER_TIMESTAMP,
                'properties': properties
            })
            
            # Validate required properties
            required_fields = ['images', 'title', 'price', 'location']
            missing_fields = [field for field in required_fields if not properties.get(field)]
            if missing_fields:
                raise ValueError(f"Missing required properties: {', '.join(missing_fields)}")
            
            # Process images into slides with property details
            image_urls = properties['images']
            clips = await self.image_processor.create_slides(image_urls, properties)
            
            # Get video dimensions
            frame_width = clips[0].w
            frame_height = clips[0].h
            
            # Define timing for each section
            section_duration = 2.0  # Each text shows for 2 seconds
            fade_duration = 0.5    # Fade in/out takes 0.5 seconds
            
            text_overlays = [
                # Title appears first at the top
                TextOverlay(
                    text=properties['title'],
                    position=TextPosition.TOP,
                    style=Templates.PROPERTY_TITLE.style,
                    animation=Animation(type=AnimationType.FADE, duration=fade_duration),
                    start_time=0.0,
                    end_time=section_duration  # Fades out as price appears
                ),
                
                # Price appears second in the middle
                TextOverlay(
                    text=f"${properties['price']}",
                    position=TextPosition.MIDDLE,
                    style=Templates.PRICE_TAG.style,
                    animation=Animation(type=AnimationType.SCALE, duration=fade_duration),
                    start_time=section_duration,
                    end_time=section_duration * 2
                ),
                
                # Features appear third in the middle-bottom
                TextOverlay(
                    text=f"🏠 {properties.get('beds', '')} Beds  🛁 {properties.get('baths', '')} Baths",
                    position=TextPosition.BOTTOM,
                    style=Templates.FEATURE_LIST.style,
                    animation=Animation(type=AnimationType.SLIDE_LEFT, duration=fade_duration),
                    start_time=section_duration * 2,
                    end_time=section_duration * 3
                ),
                
                # Location appears last at the bottom
                TextOverlay(
                    text=f"📍 {properties['location']}",
                    position=TextPosition.BOTTOM,
                    style=Templates.LOCATION.style,
                    animation=Animation(type=AnimationType.SLIDE_UP, duration=fade_duration),
                    start_time=section_duration * 3,
                    end_time=section_duration * 4
                )
            ]
            
            # Concatenate clips
            base_video = mp.concatenate_videoclips(clips)
            video_duration = base_video.duration
            
            # Apply text overlays
            final_video = self.text_overlay_service.apply_text_overlays(base_video, text_overlays)
            
            # Handle background music if provided
            audio_clip = None
            temp_audio = None
            
            if background_music:
                # Save uploaded music to temporary file
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                temp_audio.write(await background_music.read())
                temp_audio.close()
                
                # Process the background music
                processed_audio = self.audio_service.prepare_background_music(
                    temp_audio.name,
                    duration=video_duration
                )
                audio_clip = self.audio_service.create_audio_clip(processed_audio)
                
                # Set the audio of the final video
                final_video = final_video.set_audio(audio_clip)
            
            # Save to temporary file
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            final_video.write_videofile(
                temp_output.name,
                codec='libx264',
                audio_codec='aac',  # Explicit audio codec
                audio=bool(audio_clip),  # Only include audio if we have it
                fps=24,
                audio_bitrate='192k'  # Higher quality audio
            )
            
            # Upload to Firebase Storage
            video_path = f"videos/{video_id}/output.mp4"
            video_url = await self._upload_to_firebase(temp_output.name, video_path)
            
            # Update status in Firestore
            self.db.collection('videos').document(video_id).update({
                'status': 'completed',
                'video_url': video_url,
                'completed_at': firestore.SERVER_TIMESTAMP
            })
            
            # Cleanup
            os.unlink(temp_output.name)
            final_video.close()
            
            if audio_clip:
                audio_clip.close()
                self.audio_service.cleanup_temp_files(temp_audio.name)
            
            return {
                "video_id": video_id,
                "status": "completed",
                "video_url": video_url
            }
            
        except Exception as e:
            # Update error status in Firestore
            if video_id:
                self.db.collection('videos').document(video_id).update({
                    'status': 'error',
                    'error': str(e),
                    'completed_at': firestore.SERVER_TIMESTAMP
                })
            raise Exception(f"Failed to generate video: {str(e)}")
    
    async def get_video_status(self, video_id: str) -> Dict:
        """
        Get the status of a video generation task.
        
        Args:
            video_id: The ID of the video generation task
        
        Returns:
            Dict containing the status and any available results
        """
        try:
            # Get video document from Firestore
            doc = self.db.collection('videos').document(video_id).get()
            
            if not doc.exists:
                raise Exception(f"Video with id {video_id} not found")
                
            data = doc.to_dict()
            
            response = {
                "video_id": video_id,
                "status": data.get('status', 'unknown'),
                "created_at": data.get('created_at'),
            }
            
            # Add additional fields based on status
            if data.get('status') == 'completed':
                response["video_url"] = data.get('video_url')
                response["completed_at"] = data.get('completed_at')
            elif data.get('status') == 'error':
                response["error"] = data.get('error')
                response["completed_at"] = data.get('completed_at')
                
            return response
            
        except Exception as e:
            raise Exception(f"Failed to get video status: {str(e)}")
            
    async def _upload_to_firebase(self, file_path: str, destination: str) -> str:
        """
        Upload a file to Firebase Storage.
        
        Args:
            file_path: Local path to the file
            destination: Destination path in Firebase Storage
            
        Returns:
            Public URL of the uploaded file
        """
        try:
            blob = self.bucket.blob(destination)
            blob.upload_from_filename(file_path)
            blob.make_public()
            return blob.public_url
        except Exception as e:
            raise Exception(f"Failed to upload to Firebase: {str(e)}")
