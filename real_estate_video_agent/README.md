# Real Estate TikTok Video Generator 🏠

This agent automatically creates engaging, TikTok-style videos for real estate listings. It combines property images with AI-generated voiceovers and modern video effects to create professional, social media-ready content.

## Features

- 📝 AI-Generated Scripts: Creates engaging, TikTok-style property descriptions
- 🗣️ Professional Voiceovers: Uses ElevenLabs' ultra-realistic AI voices for natural-sounding narration
- 🎬 Dynamic Video Effects: Includes zoom effects, transitions, and modern animations
- 📱 TikTok-Optimized: Videos are formatted in 9:16 ratio with trending-style captions
- 🎵 Background Music: Automatically adds suitable background tracks
- ⚡ Fast Processing: Creates videos in minutes

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

3. Run the Streamlit app:
```bash
streamlit run real_estate_video_agent.py
```

## Usage

1. Enter property details:
   - Address
   - Price
   - Number of bedrooms and bathrooms
   - Property description

2. Upload 6-8 high-quality property images

3. Click "Generate Video" and wait for processing

4. Download your TikTok-ready video!

## Technical Details

- Uses OpenAI GPT-4 for script generation
- Play.ht for realistic AI voiceovers
- MoviePy and OpenCV for video effects
- Streamlit for the user interface
- Videos are exported in MP4 format, optimized for TikTok

## Requirements

- Python 3.9+
- See requirements.txt for full list of dependencies

## Notes

- Images should be high quality and in landscape orientation
- Processing time depends on the number of images and video length
- Videos are automatically cropped to TikTok's 9:16 aspect ratio
