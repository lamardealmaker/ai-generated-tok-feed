import firebase_admin
from firebase_admin import credentials, firestore, storage
import os
from pathlib import Path

def initialize_firebase():
    """Initialize Firebase Admin SDK with service account credentials."""
    try:
        # Look for service account key in environment or config directory
        cred_path = os.getenv('FIREBASE_CREDENTIALS_PATH', 
                             Path(__file__).parent.parent.parent / 'config' / 'firebase-service-account.json')
        
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET')
        })
        
        # Initialize Firestore client
        db = firestore.client()
        # Initialize Storage bucket
        bucket = storage.bucket()
        
        return db, bucket
    except Exception as e:
        print(f"Failed to initialize Firebase: {str(e)}")
        raise
