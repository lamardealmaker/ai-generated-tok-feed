# Real Estate Tok - Python Backend Service

This service handles video generation and processing for the Real Estate Tok application.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Firebase credentials:
   - Create a service account in Firebase Console
   - Download the service account key JSON file
   - Place it in `config/firebase-service-account.json`
   - Or set the environment variable `FIREBASE_CREDENTIALS_PATH`

4. Set environment variables:
```bash
export FIREBASE_STORAGE_BUCKET="your-bucket-name.appspot.com"
```

## Development

Run the development server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## Testing

Run tests:
```bash
pytest
```

## API Endpoints

### POST /api/v1/videos/generate
Generate a new video from property data and optional background music.

### GET /api/v1/videos/{video_id}
Get the status of a video generation task.

## Project Structure

```
python_service/
├── src/
│   ├── api/          # FastAPI routes and endpoints
│   ├── services/     # Business logic for video generation
│   └── firebase/     # Firebase integration
├── tests/            # Test files
├── config/           # Configuration files (not in repo)
├── requirements.txt  # Python dependencies
├── main.py          # Application entry point
└── README.md        # This file
```
