import os
import json
import random
import asyncio
from typing import List, Dict
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
from elevenlabs import Voice, VoiceSettings, generate, voices
import firebase_admin
from firebase_admin import credentials, storage
import subprocess

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize Firebase
cred = credentials.Certificate(os.getenv('FIREBASE_CREDENTIALS_PATH'))
firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET')
})
bucket = storage.bucket()

# Initialize ElevenLabs API key
os.environ['ELEVEN_LABS_API_KEY'] = os.getenv('ELEVEN_LABS_API_KEY')

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

class PropertyData(BaseModel):
    address: str
    price: int
    beds: int
    baths: float
    sqft: int
    description: str
    images: List[str]

def generate_script(property_data: PropertyData) -> str:
    """Generate a TikTok-style script using OpenAI."""
    prompt = f"""Create an engaging, TikTok-style script for a real estate property video. 
    Make it energetic, modern, and attention-grabbing. Keep it between 15-30 seconds when spoken.
    
    Property Details:
    - Address: {property_data.address}
    - Price: ${property_data.price:,}
    - Beds: {property_data.beds}
    - Baths: {property_data.baths}
    - Square Feet: {property_data.sqft}
    - Description: {property_data.description}
    
    Use emojis and modern language. Make it sound natural and conversational."""

    response = openai.ChatCompletion.create(
        model="gpt-4-0125-preview",  # Using the latest model as specified
        messages=[
            {"role": "system", "content": "You are a trendy real estate TikTok creator known for engaging, high-energy content."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

def generate_voiceover(script: str) -> str:
    """Generate voiceover using ElevenLabs with random voice selection."""
    available_voices = voices()
    selected_voice = random.choice(available_voices)
    
    # Generate audio with random stability and similarity boost
    audio = generate(
        text=script,
        voice=selected_voice.voice_id,
        model="eleven_monolingual_v1",
        stability=random.uniform(0.5, 0.8),
        similarity_boost=random.uniform(0.5, 0.8)
    )
    
    # Save audio file
    audio_path = "temp_audio.mp3"
    with open(audio_path, "wb") as f:
        f.write(audio)
    
    return audio_path

def create_video_config(property_data: PropertyData, script: str, audio_path: str) -> Dict:
    """Create configuration for Remotion video."""
    return {
        "propertyData": {
            "address": property_data.address,
            "price": property_data.price,
            "beds": property_data.beds,
            "baths": property_data.baths,
            "sqft": property_data.sqft,
            "images": property_data.images
        },
        "script": script,
        "audioPath": audio_path,
        "style": random.choice(["parallax", "kenBurns", "splitScreen", "3dRotation", "pictureInPicture"])
    }

def render_video(config: Dict) -> str:
    """Render video using Remotion."""
    # Save config to temp file
    with open("video_config.json", "w") as f:
        json.dump(config, f)
    
    # Run Remotion render command
    output_path = "output.mp4"
    subprocess.run([
        "npx",
        "remotion", "render",
        os.getenv('REMOTION_COMPOSITION_ID'),
        output_path,
        "--props", "./video_config.json",
        "--config", "./src/remotion/remotion.config.ts"
    ], check=True)
    
    return output_path

def upload_to_firebase(video_path: str, property_data: PropertyData) -> str:
    """Upload video to Firebase Storage."""
    blob_path = f"videos/{property_data.address.replace(' ', '_')}.mp4"
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(video_path)
    return blob.public_url

def cleanup_temp_files():
    """Clean up temporary files."""
    temp_files = ["temp_audio.mp3", "video_config.json", "output.mp4"]
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)

@app.post("/generate-video")
async def generate_video(property_data: PropertyData):
    try:
        # Generate script
        script = generate_script(property_data)
        
        # Generate voiceover
        audio_path = generate_voiceover(script)
        
        # Create video configuration
        video_config = create_video_config(property_data, script, audio_path)
        
        # Render video
        video_path = render_video(video_config)
        
        # Upload to Firebase
        video_url = upload_to_firebase(video_path, property_data)
        
        # Cleanup
        cleanup_temp_files()
        
        return {"status": "success", "video_url": video_url}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('REMOTION_SERVE_PORT', 3000)))
