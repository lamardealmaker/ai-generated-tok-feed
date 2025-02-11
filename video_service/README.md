# Real Estate TikTok Video Generator

A Director-based video generation service that creates engaging TikTok-style real estate videos with AI-generated voiceovers, music, and dynamic effects.

## Overview
This service uses the Director framework to create unique and engaging real estate videos. Each video is automatically generated with:
- AI-driven voiceovers
- Background music
- Dynamic text overlays
- TikTok-style effects
- Automatic style variations

## Setup

1. Install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure Director:
```bash
# Clone Director repository
git clone https://github.com/video-db/Director.git
cd Director/backend
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

## Architecture
- Built on Director's video agent framework
- FastAPI backend for API endpoints
- Custom video generation agents
- Dynamic template system
- Asset management system

## Documentation
See the `docs/` directory for detailed documentation:
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [API Documentation](docs/API.md)
- [Agent Documentation](docs/AGENTS.md)

## Development
```bash
# Start development server
python src/main.py
```

## Testing
```bash
pytest tests/
```
