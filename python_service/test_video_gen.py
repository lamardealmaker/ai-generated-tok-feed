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
    
    # Create a simple sine wave audio
    from moviepy.audio.AudioClip import AudioArrayClip
    import numpy as np
    
    fps = 44100
    duration = 5
    t = np.linspace(0, duration, int(fps * duration))
    
    # Generate a simple 440 Hz sine wave
    audio_data = np.sin(2 * np.pi * 440 * t)
    
    # Make it stereo by duplicating the mono channel
    audio_stereo = np.vstack([audio_data, audio_data])
    
    temp_music_path = "temp_music.mp3"
    sample_audio = AudioArrayClip(audio_stereo.T, fps=fps)
    sample_audio.write_audiofile(temp_music_path)
    
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
        "price": "$2,500,000",
        "location": "Miami Beach, FL",
        "images": [
            # Add 3-4 high-quality real estate image URLs here
            "https://images.unsplash.com/photo-1512917774080-9991f1c4c750",  # Modern house
            "https://images.unsplash.com/photo-1613490493576-7fde63acd811",  # Living room
            "https://images.unsplash.com/photo-1560448204-603b3fc33ddc",     # Kitchen
            "https://images.unsplash.com/photo-1560185893-a55cbc8c57e8"      # Pool
        ],
        "details": [
            "5 Bedrooms",
            "4.5 Bathrooms",
            "4,500 sqft",
            "Oceanfront Views",
            "Infinity Pool"
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
