from firebase_admin import initialize_app, credentials, firestore, storage
import datetime
import requests
import os
import random
from faker import Faker

# Initialize Firebase Admin
cred = credentials.Certificate('/Users/Learn/Desktop/gauntlet-projects/real_estate_tok/python_service/config/firebase-service-account.json')
app = initialize_app(cred, {
    'storageBucket': 'project3-ziltok.firebasestorage.app'
})
db = firestore.client()
bucket = storage.bucket()

# Initialize Faker for generating fake data
fake = Faker()

# Sample house data with real images from a public real estate dataset
SAMPLE_HOUSES = [
    {
        'image_urls': [
            'https://images.pexels.com/photos/1396122/pexels-photo-1396122.jpeg',
            'https://images.pexels.com/photos/1396132/pexels-photo-1396132.jpeg',
            'https://images.pexels.com/photos/1396122/pexels-photo-1396122.jpeg'
        ],
        'bedrooms': 4,
        'bathrooms': 3,
        'sqft': 2800,
        'price': 750000,
        'style': 'Modern Contemporary'
    },
    {
        'image_urls': [
            'https://images.pexels.com/photos/323780/pexels-photo-323780.jpeg',
            'https://images.pexels.com/photos/1643383/pexels-photo-1643383.jpeg',
            'https://images.pexels.com/photos/1643384/pexels-photo-1643384.jpeg'
        ],
        'bedrooms': 3,
        'bathrooms': 2,
        'sqft': 2200,
        'price': 550000,
        'style': 'Ranch Style'
    },
    {
        'image_urls': [
            'https://images.pexels.com/photos/1029599/pexels-photo-1029599.jpeg',
            'https://images.pexels.com/photos/1643385/pexels-photo-1643385.jpeg',
            'https://images.pexels.com/photos/1643386/pexels-photo-1643386.jpeg'
        ],
        'bedrooms': 5,
        'bathrooms': 4,
        'sqft': 3500,
        'price': 925000,
        'style': 'Mediterranean'
    }
]

def download_and_upload_image(image_url, house_id, index):
    """Download image from URL and upload to Firebase Storage"""
    try:
        # Download image
        response = requests.get(image_url)
        response.raise_for_status()
        
        # Create a temporary file
        tmp_path = f'/tmp/house_{house_id}_image_{index}.jpg'
        with open(tmp_path, 'wb') as f:
            f.write(response.content)
        
        # Upload to Firebase Storage
        blob_path = f'property_listings/{house_id}/image_{index}.jpg'
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(tmp_path)
        
        # Clean up temp file
        os.remove(tmp_path)
        
        # Get the public URL
        blob.make_public()
        return blob.public_url
    
    except Exception as e:
        print(f"Error processing image {image_url}: {str(e)}")
        return None

def create_test_listing(house_data, house_id):
    """Create a test listing with anonymized data but real images"""
    
    # Download and upload images
    image_urls = []
    for idx, image_url in enumerate(house_data['image_urls']):
        firebase_url = download_and_upload_image(image_url, house_id, idx)
        if firebase_url:
            image_urls.append(firebase_url)
    
    # Create anonymized listing data
    listing_data = {
        'id': house_id,
        'userId': f'test_user_{random.randint(1000, 9999)}',
        'title': f'{house_data["style"]} {house_data["bedrooms"]}BR Home',
        'description': fake.paragraph(nb_sentences=3),
        'propertyDetails': {
            'address': fake.street_address(),
            'city': fake.city(),
            'state': fake.state(),
            'zipCode': fake.zipcode(),
            'price': house_data['price'],
            'bedrooms': house_data['bedrooms'],
            'bathrooms': house_data['bathrooms'],
            'squareFootage': house_data['sqft']
        },
        'images': image_urls,
        'status': 'pending',
        'createdAt': datetime.datetime.now(),
        'updatedAt': datetime.datetime.now(),
        'processingStartedAt': None,
        'processingCompletedAt': None,
        'resultingVideoId': None,
        'agent': {
            'name': fake.name(),
            'phone': fake.phone_number(),
            'email': fake.email(),
            'company': fake.company()
        }
    }
    
    # Add to Firestore
    db.collection('property_listings').document(house_id).set(listing_data)
    print(f"Created listing {house_id}")
    return listing_data

def main():
    """Import test listings"""
    try:
        for idx, house_data in enumerate(SAMPLE_HOUSES):
            house_id = f'test_house_{idx + 1}'
            listing = create_test_listing(house_data, house_id)
            print(f"Successfully created listing for {listing['title']}")
    
    except Exception as e:
        print(f"Error importing test data: {str(e)}")

if __name__ == '__main__':
    main()
