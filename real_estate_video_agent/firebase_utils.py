import firebase_admin
from firebase_admin import initialize_app, credentials, firestore, storage
import datetime

def init_firebase():
    """Initialize Firebase Admin SDK or get existing app"""
    try:
        # Try to get the existing app
        app = firebase_admin.get_app()
    except ValueError:
        # If no app exists, initialize a new one
        cred = credentials.Certificate('config/firebase-service-account.json')
        app = initialize_app(cred, {
            'storageBucket': 'project3-ziltok.firebasestorage.app'
        })
    
    return firestore.client(), storage.bucket()

def save_to_videos_collection(db, bucket, property_data, video_path, image_paths):
    """Save property data and media to Firebase"""
    try:
        # Upload video to Firebase Storage
        video_blob = bucket.blob(f'videos/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
        video_blob.upload_from_filename(video_path)
        video_blob.make_public()
        video_url = video_blob.public_url
        
        # Upload images to Firebase Storage
        image_urls = []
        for idx, image_path in enumerate(image_paths):
            image_blob = bucket.blob(f'property_images/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{idx}.jpg')
            image_blob.upload_from_filename(image_path)
            image_blob.make_public()
            image_urls.append(image_blob.public_url)
        
        # Create document in videos collection
        video_data = {
            'id': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            'userId': property_data.get('userId', 'test_user'),
            'title': f"Property at {property_data['address']}",
            'description': property_data['description'],
            'thumbnailUrl': image_urls[0] if image_urls else None,
            'videoUrl': video_url,
            'likes': 0,
            'comments': 0,
            'shares': 0,
            'isFavorite': False,
            'isLiked': False,
            'createdAt': datetime.datetime.now(),
            'propertyDetails': {
                'address': property_data['address'],
                'city': property_data.get('city', ''),
                'state': property_data.get('state', ''),
                'zipCode': property_data.get('zip_code', ''),
                'price': float(property_data['price']),
                'beds': int(property_data['beds']),
                'baths': int(property_data['baths']),
                'squareFeet': int(property_data['sqft']),
                'description': property_data['description'],
                'features': property_data.get('features', []),
                'agentName': property_data.get('agent_name', ''),
                'agencyName': property_data.get('agent_company', '')
            }
        }
        
        # Add to Firestore
        db.collection('videos').document(video_data['id']).set(video_data)
        return video_data['id']
        
    except Exception as e:
        print(f"Error saving to Firebase: {str(e)}")
        raise
