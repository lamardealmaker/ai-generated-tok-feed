import os
import time
import json
import uuid
import random
import numpy as np
from pathlib import Path
from typing import List, Dict
import requests

import streamlit as st
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from moviepy.editor import *
from moviepy.video.fx.all import *
import cv2
from PIL import Image
from slugify import slugify

# Initialize environment variables
from dotenv import load_dotenv
load_dotenv()

class ZapCapClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = 'https://api.zapcap.ai'
        self.headers = {
            'x-api-key': api_key
        }
        self.json_headers = {
            'x-api-key': api_key,
            'Content-Type': 'application/json'
        }
        
        # Available template IDs
        self.template_ids = [
            '07ffd4b8-4e1a-4ee3-8921-d58802953bcd',
            '14bcd077-3f98-465b-b788-1b628951c340',
            '1bb3b68b-6a93-453a-afd7-a774b62cdab8',
            '21327a45-df89-46bc-8d56-34b8d29d3a0e',
            '46d20d67-255c-4c6a-b971-31fddcfea7f0',
            '50cdfac1-0a7a-48dd-af14-4d24971e213a',
            '55267be2-9eec-4d06-aff8-edcb401b112e',
            '5de632e7-0b02-4d15-8137-e004871e861b',
            '7b946549-ae16-4085-9dd3-c20c82504daa',
            '982ad276-a76f-4d80-a4e2-b8fae0038464',
            'a104df87-5b1a-4490-8cca-62e504a84615',
            'a51c5222-47a7-4c37-b052-7b9853d66bf6',
            'a6760d82-72c1-4190-bfdb-7d9c908732f1',
            'c88bff11-7f03-4066-94cd-88f71f9ecc68',
            'ca050348-e2d0-49a7-9c75-7a5e8335c67d',
            'cfa6a20f-cacc-4fb6-b1d0-464a81fed6cf',
            'd46bb0da-cce0-4507-909d-fa8904fb8ed7',
            'decf5309-2094-4257-a646-cabe1f1ba89a',
            'dfe027d9-bd9d-4e55-a94f-d57ed368a060',
            'e7e758de-4eb4-460f-aeca-b2801ac7f8cc',
            'eb5de878-2997-41fe-858a-726e9e3712df',
            'cc4b8197-2d49-4cc7-9f77-d9fbd8ef96ab',
            '6255949c-4a52-4255-8a67-39ebccfaa3ef',
            'a5619dcb-199d-4c6d-af05-6e5d5daef601',
            'e659ee0c-53bb-497e-869c-90f8ec0a921f',
            '1c0c9b65-47c4-41bf-a187-25a8305fd0dd',
            '9a2b0ed5-231b-4052-9211-5af9dc2de65e'
        ]
    
    def upload_video(self, video_path):
        """Upload a video file to ZapCap"""
        with open(video_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f'{self.base_url}/videos', headers=self.headers, files=files)
            response.raise_for_status()
            json_response = response.json()
            print(f'Upload response: {json_response}')
            if json_response.get('status') == 'uploaded' and json_response.get('id'):
                return json_response['id']
            else:
                raise Exception(f'Invalid upload response: {json_response}')
    
    def create_task(self, video_id, render_options=None):
        """Create a video task with custom render options"""
        # List of pastel and bright colors
        colors = [
            '#FF2C55',  # TikTok Red
            '#00F2EA',  # TikTok Cyan
            '#FE2C55',  # Vibrant Pink
            '#25F4EE',  # Bright Cyan
            '#FFFFFF',  # White
            '#F5E642',  # Bright Yellow
            '#69C9D0',  # Light Blue
            '#EE1D52',  # Deep Pink
            '#000000',  # Black (for light backgrounds)
        ]
        
        # Default render options with randomization
        default_options = {
            'subsOptions': {
                'emoji': True,
                'emojiAnimation': True,
                'emphasizeKeywords': True,
                'animation': True,
                'punctuation': True,  # Keep this true for readability
                'displayWords': random.randint(1, 4)
            },
            'styleOptions': {
                'top': random.randint(45, 75),  # Random position
                'fontUppercase': random.choice([True, False]),
                # Adjust font size based on number of words displayed
                'fontSize': random.randint(25, 55) if random.randint(1, 4) == 1 else random.randint(30, 44),
                'fontWeight': random.randint(300, 750),  # A50 interpreted as 850
                'fontColor': random.choice(colors)
            },
            'highlightOptions': {
                'randomColourOne': random.choice(colors),
                'randomColourTwo': random.choice(colors),
                'randomColourThree': random.choice(colors)
            }
        }
        
        # Print the randomized options for debugging
        print(f'Using style options:\n{json.dumps(default_options, indent=2)}')
        
        data = {
            'autoApprove': True,
            'language': 'en',
            'templateId': random.choice(self.template_ids),
            'renderOptions': render_options if render_options else default_options
        }
        print(f'Creating task with data: {data}')
        try:
            response = requests.post(
                f'{self.base_url}/videos/{video_id}/task',
                headers=self.json_headers,
                json=data
            )
            response.raise_for_status()
            json_response = response.json()
            print(f'Task response: {json_response}')
            if 'taskId' in json_response:
                return json_response['taskId']
            else:
                raise Exception(f'Invalid task response: {json_response}')
        except requests.exceptions.RequestException as e:
            print(f'Error response: {e.response.text if e.response else e}')
            raise
    
    def get_task_status(self, video_id, task_id):
        """Get the status of a video task"""
        response = requests.get(
            f'{self.base_url}/videos/{video_id}/task/{task_id}',
            headers=self.json_headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_video_url(self, video_id, task_id):
        """Get the URL of the rendered video"""
        response = requests.get(
            f'{self.base_url}/videos/{video_id}/task/{task_id}',
            headers=self.json_headers
        )
        response.raise_for_status()
        json_response = response.json()
        print(f'Task status response: {json_response}')
        if json_response.get('status') == 'completed' and json_response.get('downloadUrl'):
            return json_response['downloadUrl']
        else:
            raise Exception(f'Video not ready yet. Status: {json_response.get("status")}')
    
    def download_video(self, url, output_path):
        """Download the rendered video"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

# Configure APIs
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
zapcap_client = ZapCapClient(api_key=os.getenv("ZAPCAP_API_KEY"))

class RealEstateVideoAgent:
    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Setup font path
        custom_font_path = Path("fonts/THEBOLDFONT-FREEVERSION.ttf")
        self.font_path = custom_font_path if custom_font_path.exists() else None
        if not self.font_path:
            print("Custom font not found. Using system fonts.")
        
        # Font options
        self.system_fonts = [
            'Arial-Bold',
            'Helvetica-Bold',
            'Verdana-Bold',
            'Tahoma-Bold',
            'Trebuchet-Bold',
            'Georgia-Bold',
        ]
        
        # Color options
        self.font_colors = [
            '#FFFFFF',  # White
            '#FFE5E5',  # Light Pink
            '#E5FFE5',  # Light Green
            '#E5E5FF',  # Light Blue
            '#FFFFE5',  # Light Yellow
            '#FFE5FF',  # Light Purple
        ]
        
        self.stroke_colors = [
            '#000000',  # Black
            '#020a26',  # Navy
            '#1a0429',  # Deep Purple
        ]
        
        # Default ElevenLabs voices (pre-selected professional voices)
        self.voices = [
            "21m00Tcm4TlvDq8ikWAM",  # Rachel
            "29vD33N1CtxCmqQRPOHJ",  # Drew
            "2EiwWnXFnvU5JabPnv8n",  # Clyde
            "D38z5RcWu1voky8WS1ja",  # Adam
            "EXAVITQu4vr4xnSDxMaL",  # Antoni
            "ErXwobaYiN019PkySvjV",  # Bella
            "MF3mGyEYCl7XYWbV9V6O",  # Elli
            "TxGEqnHWrfWFTfGW9XjX",  # Josh
            "VR6AewLTigWG4xSOukaG",  # Arnold
            "pNInz6obpgDQGcFmaJgB"   # Sam
        ]
        
    def generate_script(self, property_details: Dict) -> str:
        """Generate an engaging TikTok-style script for the property."""
        prompt = f"""Write a short, factual script (maximum 15 seconds when read at a natural pace) for this property:
        Address: {property_details['address']}, {property_details['city']}, {property_details['state']}
        Price: ${property_details['price']:,}
        Bedrooms: {property_details['beds']}
        Bathrooms: {property_details['baths']}
        Square Feet: {property_details['sqft']}
        Key Features: {', '.join(property_details.get('features', []))}
        Description: {property_details['description']}
        
        take the description and features to create a BRIEF factual script. highlighting the 
        selling points and key features. do not add hyperbole or calls to action if not in property description.
        script needs to be able to be read in 10-15 seconds at a natural pace. If too long you will
        cause errors. 
        Every word in script will be read so do not
        add any information that is not to be read aloud. Spell out all abreviations and acronyms.
        
        IMPORTANT: Keep the script very concise. It should take no more than 10 seconds to read at a natural pace.
        Focus on the most important details and unique features only."""
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content

    def generate_voiceover(self, script: str, voice_id: str = None):
        """Generate AI voiceover from the script using ElevenLabs and get word timings."""
        # Generate audio file
        audio_path = self.temp_dir / f"{uuid.uuid4()}.mp3"
        
        # Select random voice if none specified
        if voice_id is None:
            voice_id = random.choice(self.voices)
        
        # Generate audio with word-level timestamps
        response = elevenlabs_client.generate(
            text=script,
            voice=voice_id,
            model="eleven_multilingual_v2",
            output_format="mp3_44100_128",
            voice_settings={
                "stability": 0.71,
                "similarity_boost": 0.5,
                "style": 0.9,
                "use_speaker_boost": True,
                "speaking_rate": 1.5
            },
            stream=False
        )

        # Save the audio file
        with open(audio_path, 'wb') as f:
            # Handle the generator response
            for chunk in response:
                f.write(chunk)
            
        # Create default word timings based on word count and audio duration
        from mutagen.mp3 import MP3
        
        # Get audio duration
        audio = MP3(str(audio_path))
        duration = audio.info.length  # Duration in seconds
        
        # Create evenly spaced word timings
        words = script.split()
        word_duration = duration / len(words)
        word_timings = [
            {
                'word': word,
                'start': i * word_duration * 1000,  # Convert to milliseconds
                'end': (i + 1) * word_duration * 1000
            }
            for i, word in enumerate(words)
        ]

        # Return both the audio path and word timings

        return str(audio_path), word_timings

    def create_video_clip(self, images: List[str], audio_path: str) -> VideoFileClip:
        """Create video with TikTok-style effects."""
        # Video dimensions (9:16 aspect ratio, 1080p)
        width, height = 1080, 1920
        
        # Load audio and calculate clip durations
        audio = AudioFileClip(audio_path)
        clip_duration = audio.duration / len(images)
        
        # Load and process images
        clips = []
        for i, img_path in enumerate(images):
            # Load and process image
            img = Image.open(img_path)
            
            # Calculate dimensions to crop to 9:16 while maintaining center
            target_ratio = height / width
            img_ratio = img.height / img.width
            
            if img_ratio > target_ratio:
                # Image is too tall, crop height
                new_height = int(img.width * target_ratio)
                top = (img.height - new_height) // 2
                img = img.crop((0, top, img.width, top + new_height))
            else:
                # Image is too wide, crop width
                new_width = int(img.height / target_ratio)
                left = (img.width - new_width) // 2
                img = img.crop((left, 0, left + new_width, img.height))
            
            # Resize to target dimensions
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            # Convert to video clip
            clip = ImageClip(np.array(img)).set_duration(clip_duration)
            
            # Add random transition effect
            if i > 0:
                effect = random.choice(['fade', 'slide', 'zoom'])
                if effect == 'fade':
                    clip = clip.crossfadein(0.5)
                elif effect == 'slide':
                    clip = clip.set_position(lambda t: ('center', -height + (t * height)))
                elif effect == 'zoom':
                    clip = clip.resize(lambda t: 1 + 0.1 * t)
            
            # No text overlay on first image
            
            clips.append(clip)
        
        # Concatenate clips
        video = concatenate_videoclips(clips, method="compose")
        video = video.set_audio(audio)
        
        return video

    def add_captions(self, video: VideoFileClip, script: str, word_timings=None) -> VideoFileClip:
        """Add animated captions to the video with word-by-word timing using ElevenLabs timestamps."""
        # Get audio for timing
        audio = video.audio
        text_clips = []
        
        # Choose random styling once for consistency
        font_size = random.randint(50, 80)
        stroke_width = random.randint(4, 7)
        font = str(self.font_path) if self.font_path else random.choice(self.system_fonts)
        font_color = random.choice(self.font_colors)
        stroke_color = random.choice(self.stroke_colors)
        
        # Random vertical position (25% from top, center, or 25% from bottom)
        y_positions = [
            int(video.h * 0.25),  # 25% from top
            int(video.h * 0.5),   # center
            int(video.h * 0.75)   # 25% from bottom
        ]
        y_pos = random.choice(y_positions)
        
        if word_timings:
            # Use ElevenLabs word timings for precise synchronization
            current_words = []
            current_start = None
            current_end = None
            
            for word_timing in word_timings:
                word = word_timing.get('word', '')
                start = word_timing.get('start', 0) / 1000  # Convert ms to seconds
                end = word_timing.get('end', 0) / 1000
                
                if current_start is None:
                    current_start = start
                
                current_words.append(word)
                current_end = end
                
                # Break on punctuation, 3 words, or long pause (>0.5s)
                next_timing = word_timings[word_timings.index(word_timing) + 1] if word_timings.index(word_timing) < len(word_timings) - 1 else None
                pause_duration = (next_timing.get('start', end) / 1000 - end) if next_timing else 0
                
                if len(current_words) == 3 or any(p in word for p in '.!?,') or pause_duration > 0.5:
                    formatted_text = ' '.join(current_words)
                    
                    # Use consistent styling for all clips
                    
                    # Create text clip with styling
                    txt_clip = TextClip(
                        formatted_text,
                        fontsize=font_size,
                        color=font_color,
                        stroke_color=stroke_color,
                        stroke_width=stroke_width,
                        font=font,
                        method='caption',
                        size=(1000, None),
                        bg_color='transparent',
                        align='center'
                    )
                    
                    # Position and timing
                    txt_clip = txt_clip.set_position(('center', y_pos))
                    txt_clip = txt_clip.set_start(current_start)
                    txt_clip = txt_clip.set_duration(current_end - current_start)
                    txt_clip = txt_clip.crossfadein(0.1).crossfadeout(0.1)
                    
                    text_clips.append(txt_clip)
                    
                    # Reset for next group
                    current_words = []
                    current_start = None
        else:
            # Fallback to evenly spaced timing if no word timings available
            words = script.replace(".", "").split()
            word_groups = []
            current_group = []
            
            for word in words:
                current_group.append(word)
                if len(current_group) == 3 or word.endswith(('.', '!', '?', ',')):
                    word_groups.append(current_group)
                    current_group = []
            if current_group: 
                word_groups.append(current_group)
            
            # Calculate timing
            total_groups = len(word_groups)
            duration_per_group = audio.duration / total_groups
        
            # Create clips for evenly-spaced timing fallback
            for i, word_group in enumerate(word_groups):
                formatted_text = ' '.join(word_group)
                
                # Create text clip with styling
                txt_clip = TextClip(
                    formatted_text,
                    fontsize=80,
                    color='#FFFFFF',
                    stroke_color='#000000',
                    stroke_width=4,
                    font=str(self.font_path),
                    method='caption',
                    size=(1000, None),
                    bg_color='transparent',
                    align='center'
                )
                
                # Position and timing
                txt_clip = txt_clip.set_position(('center', 1440))
                txt_clip = txt_clip.set_start(i * duration_per_group)
                txt_clip = txt_clip.set_duration(duration_per_group)
                txt_clip = txt_clip.crossfadein(0.1).crossfadeout(0.1)
                
                text_clips.append(txt_clip)
            
        # Combine video with text clips
        final_video = CompositeVideoClip([video] + text_clips)
        return final_video









    def generate_features(self, property_details: Dict) -> List[str]:
        """Generate 3-5 key features for the property using OpenAI."""
        try:
            # Create a prompt for OpenAI
            prompt = f"Given a ${property_details['price']:,} property in {property_details['city']}, {property_details['state']} "
            prompt += f"with {property_details['beds']} beds, {property_details['baths']} baths, and {property_details['sqft']} square feet, "
            prompt += f"and this description: {property_details['description']}, "
            prompt += "list 3-5 key features that would be most appealing to potential buyers. Each feature should be short and less than 3 words. "
            prompt += "Return only the feature names, separated by commas, no explanations or additional text."

            # Call OpenAI
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": "You are a real estate expert. List only the features, no explanations."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.7,
                max_tokens=100
            )

            # Parse the response
            features_text = response.choices[0].message.content.strip()
            features = [feature.strip() for feature in features_text.split(',')]
            return features

        except Exception as e:
            print(f"Error generating features: {str(e)}")
            return ["Spacious Layout", "Modern Amenities", "Prime Location"]  # Default fallback

    def create_property_video(self, property_details: Dict, images: List[str]) -> str:
        """Create complete TikTok-style real estate video."""
        try:
            # Generate features first
            features = self.generate_features(property_details)
            property_details['features'] = features
            print(f"Generated features: {features}")

            # Generate script
            script = self.generate_script(property_details)
            
            # Generate voiceover with word timings
            audio_path, word_timings = self.generate_voiceover(script)
            
            # Create base video with effects
            video = self.create_video_clip(images, audio_path)
            
            # Save temporary video without captions
            temp_path = self.temp_dir / f"temp_{uuid.uuid4()}.mp4"
            video.write_videofile(
                str(temp_path),
                fps=30,
                codec='libx264',
                audio_codec='aac',
                preset='medium',
                threads=2
            )
            video.close()
        
            # Upload video to ZapCap
            print("Uploading video to ZapCap...")
            video_id = zapcap_client.upload_video(str(temp_path))
            
            # Configure render options
            render_options = {
                'subsOptions': {
                    'emoji': True,
                    'emojiAnimation': True,
                    'emphasizeKeywords': True,
                    'animation': True,
                    'punctuation': False,
                    'displayWords': random.randint(1, 5)
                },
                'styleOptions': {
                    'top': random.choice([45, 50, 75]),
                    'fontUppercase': random.choice([True, False]),
                    'fontSize': random.randint(20, 50),
                    'fontWeight': random.randint(300, 750)
                }
            }
            
            # Create ZapCap task
            print("Creating ZapCap task...")
            task_id = zapcap_client.create_task(video_id, render_options)
            
            # Wait for task completion
            print("Processing video with ZapCap...")
            while True:
                status = zapcap_client.get_task_status(video_id, task_id)
                if status['status'] == 'completed':
                    break
                elif status['status'] == 'failed':
                    raise Exception(f"ZapCap task failed: {status.get('error')}")
                time.sleep(5)
            
            # Get rendered video URL
            print("Downloading final video...")
            video_url = zapcap_client.get_video_url(video_id, task_id)
            
            # Download final video
            output_path = self.output_dir / f"{slugify(property_details['address'])}.mp4"
            zapcap_client.download_video(video_url, str(output_path))
            
            return str(output_path)
            
        except Exception as e:
            print(f"Error creating property video: {str(e)}")
            raise
        finally:
            # Cleanup
            try:
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink()
                if 'audio_path' in locals():
                    os.remove(audio_path)
            except Exception as e:
                print(f"Error during cleanup: {str(e)}")

def main():
    from firebase_utils import init_firebase, save_to_videos_collection
    
    # Initialize Firebase
    db, bucket = init_firebase()
    
    st.title("🏠 Real Estate TikTok Video Generator")
    
    # Property details input methods
    input_method = st.radio("Choose input method:", ["Form", "JSON"])
    
    property_details = {}
    image_paths = []
    
    if input_method == "Form":
        # Property details form
        st.header("Property Details")
        default_description = """Luxury Living at Zilkr On The Park Condos! Rare, sun drenched, corner One Bedroom Plus Den floor plan with coveted southwest facing views overlooking Zilker Park, Barton Creek and rooftop Pool. Live your best life with the Hike & Bike Trail only steps from the building, and enjoy easy access to downtown, SoCo, Mopac, and I35."""
        
        # Basic Property Information
        st.subheader("Basic Information")
        col1, col2 = st.columns(2)
        with col1:
            address = st.text_input("Street Address", value="1900 Barton Springs Rd")
            city = st.text_input("City", value="Austin")
            state = st.text_input("State", value="TX")
            zip_code = st.text_input("ZIP Code", value="78704")
        
        with col2:
            price = st.number_input("Price", min_value=0, value=599000, step=1000)
            beds = st.number_input("Bedrooms", min_value=0, value=3, step=1)
            baths = st.number_input("Bathrooms", min_value=0.0, value=2.0, step=0.5)
            sqft = st.number_input("Square Feet", min_value=0, value=2000, step=100)
        
        # Property Description
        st.subheader("Description")
        description = st.text_area("Property Description", value=default_description, height=100)
        
        # Features will be generated automatically
        
        # Agent Information
        st.subheader("Agent Information")
        col3, col4 = st.columns(2)
        with col3:
            agent_name = st.text_input("Agent Name")
        with col4:
            agent_company = st.text_input("Agency Name")
        
        # Image upload
        st.header("Property Images")
        st.write("Upload 6-8 high-quality images (They will be shown in the uploaded order)")
        uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
        
        if st.button("Generate Video") and uploaded_files:
            if len(uploaded_files) < 6 or len(uploaded_files) > 8:
                st.error("Please upload between 6 and 8 images.")
                return
                
            # Save uploaded images temporarily
            temp_dir = Path("temp")
            temp_dir.mkdir(exist_ok=True)
            
            # Clear any existing image files
            for f in temp_dir.glob("*.jpg"):
                f.unlink()
            for f in temp_dir.glob("*.jpeg"):
                f.unlink()
            for f in temp_dir.glob("*.png"):
                f.unlink()
            
            # Save new images
            for idx, file in enumerate(uploaded_files):
                file_ext = Path(file.name).suffix
                temp_path = temp_dir / f"image_{idx}{file_ext}"
                with open(temp_path, "wb") as f:
                    f.write(file.getvalue())
                image_paths.append(str(temp_path))
                
            property_details = {
                "address": address,
                "city": city,
                "state": state,
                "zip_code": zip_code,
                "price": price,
                "beds": beds,
                "baths": baths,
                "sqft": sqft,
                "description": description,
                "agent_name": agent_name,
                "agent_company": agent_company,
                "userId": "test_user"  # Default user ID
            }
    
    else:  # JSON input
        st.header("JSON Input")
        json_input = st.text_area("Enter property details in JSON format", """
        {
            "address": "123 Main St, Austin, TX",
            "price": 750000,
            "beds": 4,
            "baths": 3,
            "sqft": 2500,
            "description": "Beautiful modern home with updated kitchen...",
            "images": [
                "https://example.com/image1.jpg",
                "https://example.com/image2.jpg",
                "https://example.com/image3.jpg",
                "https://example.com/image4.jpg",
                "https://example.com/image5.jpg",
                "https://example.com/image6.jpg"
            ]
        }
        """)
        
        if st.button("Generate Video"):
            try:
                data = json.loads(json_input)
                property_details = data
                
                # Download images from URLs
                temp_dir = Path("temp")
                temp_dir.mkdir(exist_ok=True)
                
                for i, url in enumerate(data['images']):
                    response = requests.get(url)
                    if response.status_code == 200:
                        ext = url.split('.')[-1]
                        temp_path = temp_dir / f"image_{i}.{ext}"
                        with open(temp_path, 'wb') as f:
                            f.write(response.content)
                        image_paths.append(str(temp_path))
                
                if len(image_paths) < 6:
                    st.error("Could not download enough valid images (minimum 6 required)")
                    return
                    
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
                return
            except Exception as e:
                st.error(f"Error processing input: {str(e)}")
                return
        
    # Only proceed if we have property details and images
    if property_details and image_paths:
        with st.spinner("🎬 Creating your TikTok video..."):
            try:
                agent = RealEstateVideoAgent()
                video_path = agent.create_property_video(property_details, image_paths)
                
                # Save to Firebase
                video_id = save_to_videos_collection(db, bucket, property_details, video_path, image_paths)
                
                st.success("✨ Video created and saved successfully!")
                st.video(video_path)
                
                # Show Firebase details
                st.info(f"Video saved to Firebase with ID: {video_id}")
                
                # Download button
                with open(video_path, "rb") as file:
                    st.download_button(
                        label="Download Video",
                        data=file,
                        file_name=os.path.basename(video_path),
                        mime="video/mp4"
                    )
                    
                # Cleanup temporary images after everything is done
                for path in image_paths:
                    try:
                        os.remove(path)
                    except:
                        pass
            except Exception as e:
                st.error(f"Error creating video: {str(e)}")
                # Cleanup on error
                for path in image_paths:
                    try:
                        os.remove(path)
                    except:
                        pass

if __name__ == "__main__":
    main()
