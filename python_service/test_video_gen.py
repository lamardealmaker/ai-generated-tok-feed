import asyncio
import json
import os
import sys
from dotenv import load_dotenv
from firebase_admin import initialize_app, credentials

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Initialize Firebase
cred = credentials.Certificate('config/firebase-service-account.json')
initialize_app(cred, {
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET')
})

from src.services.video_service import VideoService

# Load environment variables
load_dotenv()

async def test_video_generation():
    # Create a sample background music file using moviepy
    from moviepy.editor import AudioClip
    import numpy as np
    
    # Use the downloaded test music file
    temp_music_path = "test_music.mp3"
    
    # Create an UploadFile object from the sample music
    from fastapi import UploadFile
    import aiofiles
    background_music = UploadFile(
        filename="background.mp3",
        file=open(temp_music_path, "rb")
    )
    
    # Sample property data
    property_data = {
        "title": "Luxury Waterfront Villa",
        "price": "2,500,000",
        "location": "Miami Beach, FL",
        "beds": "5",
        "baths": "4.5",
        "sqft": "4,500",
        "images": [
            # Add 3-4 high-quality real estate image URLs here
            "https://images.unsplash.com/photo-1512917774080-9991f1c4c750",  # Modern house
            "https://images.unsplash.com/photo-1613490493576-7fde63acd811",  # Living room
            "https://images.unsplash.com/photo-1560448204-603b3fc33ddc",     # Kitchen
            "https://images.unsplash.com/photo-1560185893-a55cbc8c57e8"      # Pool
        ],
        "features": [
            "Oceanfront Views",
            "Infinity Pool",
            "Smart Home System",
            "Wine Cellar"
        ]
    }

    try:
        # Initialize video service
        video_service = VideoService()
        
        print("Starting video generation...")
        result = await video_service.generate_video(
            property_data,
            background_music=background_music
        )
        
        print("\nVideo generation completed!")
        print(f"Video ID: {result['video_id']}")
        print(f"Status: {result['status']}")
        print(f"Video URL: {result.get('video_url', 'Not available')}")
        
        # Get the status
        status = await video_service.get_video_status(result['video_id'])
        print("\nFinal Status:", json.dumps(status, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Cleanup
        background_music.file.close()
        if os.path.exists(temp_music_path):
            os.remove(temp_music_path)

if __name__ == "__main__":
    asyncio.run(test_video_generation())
