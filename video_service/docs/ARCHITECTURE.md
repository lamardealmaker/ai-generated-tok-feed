# Video Service Architecture Documentation

## Overview
The video service is a sophisticated backend system designed to generate engaging real estate video content. It uses a modular architecture with multiple components working together to create professional-quality videos with dynamic scripts, effects, and music.

## Directory Structure
```
video_service/
├── src/                    # Main source code
│   ├── api/               # API endpoints and models
│   ├── director/          # Core video generation logic
│   │   ├── agents/       # Specialized AI agents
│   │   ├── ai/           # AI/LLM integration
│   │   ├── core/         # Core engine components
│   │   ├── effects/      # Video effects and transitions
│   │   ├── tools/        # Utility tools
│   │   └── variation/    # Style variation engines
├── assets/                # Static assets
│   ├── audio/            # Music and voiceover files
│   │   ├── music/        # Background music tracks
│   │   └── voiceover/    # Generated voiceovers
├── config/                # Configuration files
├── output/                # Generated video files
└── tests/                 # Test suite
```

## Core Components

### 1. Director Module (`src/director/`)

#### Core Engine (`core/`)
- **ReasoningEngine** (`reasoning_engine.py`): The central orchestrator that coordinates all components:
  - Manages the video generation workflow
  - Coordinates between different agents
  - Handles task scheduling and progress tracking
  - Implements hook system for extensibility

#### Agents (`agents/`)
- **ScriptAgent** (`script_agent.py`): Generates dynamic video scripts
  - Uses templates for different styles (modern, luxury, minimal)
  - Analyzes property data to identify key features
  - Generates engaging hooks and closings
  - Supports script pacing optimization

- **PacingAgent** (`pacing_agent.py`): Manages video timing and flow
  - Optimizes segment durations
  - Ensures smooth transitions
  - Adapts to content type

- **EffectAgent** (`effect_agent.py`): Handles visual effects
  - Generates effect combinations
  - Matches effects to content type
  - Supports different visual styles

#### Tools (`tools/`)
- **VideoDBTool** (`video_db_tool.py`): Manages video processing and transitions
  - Provides transition library
  - Handles effect application
  - Processes video segments

- **VideoRenderer** (`video_renderer.py`): Handles video rendering
  - Applies effects and transitions
  - Creates text overlays
  - Manages final video composition

- **VoiceoverTool** (`voiceover_tool.py`): Manages audio generation
  - Integrates with ElevenLabs API
  - Generates voiceovers for scripts
  - Handles audio synchronization

- **MusicTool** (`music_tool.py`): Manages background music
  - Selects appropriate tracks
  - Handles music timing
  - Adjusts volume levels

### 2. API Layer (`src/api/`)
- **endpoints.py**: REST API endpoints
- **models.py**: Data models and schemas
- **routes.py**: URL routing and handlers

## Key Features

### 1. Video Generation Pipeline
1. **Script Generation**
   - Analyzes property data
   - Generates engaging script using templates
   - Optimizes for platform (e.g., TikTok)

2. **Visual Processing**
   - Applies style-specific effects
   - Manages transitions
   - Handles text overlays

3. **Audio Processing**
   - Generates voiceovers
   - Selects and applies background music
   - Synchronizes audio elements

### 2. Style System
- Supports multiple video styles:
  - Modern: Dynamic and energetic
  - Luxury: Elegant and sophisticated
  - Minimal: Clean and simple

### 3. Effect System
- **Transitions**:
  - Fade, slide, zoom effects
  - Style-specific transitions
  - Customizable durations

- **Visual Effects**:
  - Color grading
  - Filters
  - Overlays

## Configuration

### 1. Environment Variables
Required in `.env`:
- `ELEVENLABS_API_KEY`: For voiceover generation
- Other API keys as needed

### 2. Configuration Files
- `config/director/llm.yaml`: LLM settings
- `config/director/styles.yaml`: Video style definitions

## Asset Management

### 1. Audio Assets
- **Music**: Organized by style in `assets/audio/music/`
- **Voiceovers**: Generated and stored in `assets/audio/voiceover/`

### 2. Output
- Generated videos stored in `output/`
- Unique IDs for each video
- Maintains video metadata

## Testing
Comprehensive test suite in `tests/`:
- Unit tests for each component
- Integration tests
- Style variation tests
- AI integration tests

## Development Workflow

### 1. Adding New Features
1. Implement feature in appropriate module
2. Add corresponding tests
3. Update configuration if needed
4. Document changes

### 2. Modifying Styles
1. Update style templates in `script_agent.py`
2. Modify effects in `video_db_tool.py`
3. Update corresponding music selection

### 3. Testing
```bash
# Run all tests
pytest tests/

# Run specific test category
pytest tests/test_agents.py
```

## Performance Considerations

### 1. Video Processing
- Optimized for TikTok format
- Efficient effect application
- Smart caching of generated assets

### 2. Resource Management
- Proper cleanup of temporary files
- Efficient memory usage during rendering
- Parallel processing where possible

## Security
- API key management
- Input validation
- Secure file handling
- Rate limiting

## Future Improvements
1. Add more video styles
2. Implement advanced transitions
3. Enhance AI-driven content generation
4. Add analytics integration
5. Implement caching system

## Troubleshooting
Common issues and solutions:
1. Video generation fails
   - Check available disk space
   - Verify API keys
   - Check input data format

2. Audio sync issues
   - Verify segment durations
   - Check voiceover generation
   - Validate music file format

## API Documentation

### 1. Generate Video
```python
POST /api/video/generate
{
    "property_data": {
        "id": "string",
        "location": "string",
        "price": float,
        "features": [string],
        ...
    },
    "style": "modern" | "luxury" | "minimal",
    "duration": float
}
```

### 2. Get Video Status
```python
GET /api/video/{video_id}/status
```

## Dependencies
Major dependencies:
- MoviePy: Video processing
- ElevenLabs: Voice generation
- Firebase: Asset storage
- FastAPI: API framework
