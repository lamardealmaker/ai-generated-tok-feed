from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from src.api.routes import router as api_router
from src.firebase.admin import initialize_firebase

app = FastAPI(
    title="Real Estate Tok Backend",
    description="Backend service for Real Estate Tok video generation and processing",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Firebase Admin SDK
initialize_firebase()

# Include API routes
app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
