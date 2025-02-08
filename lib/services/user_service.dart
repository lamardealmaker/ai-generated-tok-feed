import 'dart:io';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import '../models/user_model.dart';
import '../utils/avatar_generator.dart';

class UserService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseStorage _storage = FirebaseStorage.instance;

  // Collection reference
  CollectionReference get _users => _firestore.collection('users');

  // Create new user document
  Future<void> createUser(UserModel user) async {
    // Generate avatar color based on username/email
    final color = AvatarGenerator.generateColor(user.username ?? user.email);
    
    final userData = user.toJson();
    userData['avatarColor'] = color.value; // Store color for consistency
    
    await _users.doc(user.uid).set(userData);
  }

  // Get user by ID
  Future<UserModel?> getUserById(String uid) async {
    DocumentSnapshot doc = await _users.doc(uid).get();
    if (doc.exists) {
      return UserModel.fromFirestore(doc);
    }
    return null;
  }

  // Update user data
  Future<void> updateUser(String uid, Map<String, dynamic> data) async {
    // If username is being updated, generate new avatar color
    if (data.containsKey('username')) {
      final color = AvatarGenerator.generateColor(data['username']);
      data['avatarColor'] = color.value;
    }
    await _users.doc(uid).update(data);
  }

  // Upload profile image
  Future<String> uploadProfileImage(String uid, File imageFile) async {
    // Create a reference to the location you want to upload to in firebase
    Reference ref = _storage.ref().child('profile_images').child(uid).child('profile.jpg');
    
    // Upload the file to firebase
    await ref.putFile(imageFile);
    
    // Get download URL
    String downloadURL = await ref.getDownloadURL();
    
    // Update user profile with new image URL
    await updateUser(uid, {'profileImageUrl': downloadURL});
    
    return downloadURL;
  }

  // Delete profile image
  Future<void> deleteProfileImage(String uid) async {
    try {
      // Delete from storage
      await _storage.ref().child('profile_images').child(uid).child('profile.jpg').delete();
      
      // Update user profile
      await updateUser(uid, {'profileImageUrl': null});
    } catch (e) {
      print('Error deleting profile image: $e');
    }
  }

  // Update user type
  Future<void> updateUserType(String uid, String userType) async {
    await updateUser(uid, {'userType': userType});
  }

  // Update user preferences
  Future<void> updatePreferences(String uid, Map<String, dynamic> preferences) async {
    await updateUser(uid, {'preferences': preferences});
  }

  // Get all users of a specific type
  Future<List<UserModel>> getUsersByType(String userType) async {
    QuerySnapshot snapshot = await _users
        .where('userType', isEqualTo: userType)
        .get();
    
    return snapshot.docs
        .map((doc) => UserModel.fromFirestore(doc))
        .toList();
  }

  // Search users by username or fullName
  Future<List<UserModel>> searchUsers(String query) async {
    QuerySnapshot snapshot = await _users
        .where('username', isGreaterThanOrEqualTo: query)
        .where('username', isLessThan: query + 'z')
        .get();

    return snapshot.docs
        .map((doc) => UserModel.fromFirestore(doc))
        .toList();
  }

  // Update last login
  Future<void> updateLastLogin(String uid) async {
    await updateUser(uid, {
      'lastLogin': DateTime.now().toIso8601String(),
    });
  }

  // Delete user data (when account is deleted)
  Future<void> deleteUserData(String uid) async {
    try {
      // Delete profile image
      await deleteProfileImage(uid);
      // Delete user document
      await _users.doc(uid).delete();
    } catch (e) {
      print('Error deleting user data: $e');
    }
  }

  // Check if username is available
  Future<bool> isUsernameAvailable(String username) async {
    final normalizedUsername = username.toLowerCase();
    final QuerySnapshot result = await _users
        .where('username', isEqualTo: normalizedUsername)
        .limit(1)
        .get();
    return result.docs.isEmpty;
  }

  // Set default avatar for user
  Future<void> setDefaultAvatar(String uid, int avatarNumber) async {
    final String avatarUrl = 'assets/avatars/avatar$avatarNumber.png';
    await updateUser(uid, {'profileImageUrl': avatarUrl});
  }

  // Update username
  Future<void> updateUsername(String uid, String username) async {
    await updateUser(uid, {'username': username});
  }
}
