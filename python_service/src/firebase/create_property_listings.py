from firebase_admin import initialize_app, credentials, firestore
import datetime
import json

# Initialize Firebase Admin
cred = credentials.Certificate('/Users/Learn/Desktop/gauntlet-projects/real_estate_tok/python_service/config/firebase-service-account.json')
app = initialize_app(cred)
db = firestore.client()

# Create a test property listing document
test_property = {
    "id": "test_property_1",
    "userId": "test_user_1",
    "title": "Beautiful 3BR House in Austin",
    "description": "Spacious family home with modern amenities",
    "propertyDetails": {
        "address": "123 Test Street, Austin, TX",
        "price": 450000,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "squareFootage": 2200
    },
    "status": "pending",
    "createdAt": datetime.datetime.now(),
    "updatedAt": datetime.datetime.now(),
    "processingStartedAt": None,
    "processingCompletedAt": None,
    "resultingVideoId": None
}

# Add the document to the property_listings collection
try:
    db.collection('property_listings').document(test_property['id']).set(test_property)
    print("Successfully created property_listings collection and added test document")
    
    # Verify the document was created
    doc = db.collection('property_listings').document(test_property['id']).get()
    if doc.exists:
        print("Verified document exists in collection")
        print(f"Document data: {doc.to_dict()}")
    else:
        print("Error: Document was not created")
        
except Exception as e:
    print(f"Error creating collection/document: {str(e)}")
