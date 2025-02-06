import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:flutter/material.dart';

class UserModel {
  final String uid;
  final String email;
  final String? username;
  final String? fullName;
  final String? phoneNumber;
  final String? profileImageUrl;
  final bool isEmailVerified;
  final DateTime createdAt;
  final DateTime? lastLogin;
  final String userType;
  final Map<String, dynamic>? preferences;
  final Color avatarColor;

  UserModel({
    required this.uid,
    required this.email,
    this.username,
    this.fullName,
    this.phoneNumber,
    this.profileImageUrl,
    required this.isEmailVerified,
    required this.createdAt,
    this.lastLogin,
    required this.userType,
    this.preferences,
    Color? avatarColor,
  }) : avatarColor = avatarColor ?? Colors.blue;

  Map<String, dynamic> toJson() {
    return {
      'uid': uid,
      'email': email,
      'username': username,
      'fullName': fullName,
      'phoneNumber': phoneNumber,
      'profileImageUrl': profileImageUrl,
      'isEmailVerified': isEmailVerified,
      'createdAt': createdAt.toIso8601String(),
      'lastLogin': lastLogin?.toIso8601String(),
      'userType': userType,
      'preferences': preferences,
      'avatarColor': avatarColor.value,
    };
  }

  factory UserModel.fromJson(Map<String, dynamic> json) {
    return UserModel(
      uid: json['uid'] ?? '',
      email: json['email'] ?? '',
      username: json['username'],
      fullName: json['fullName'],
      phoneNumber: json['phoneNumber'],
      profileImageUrl: json['profileImageUrl'],
      isEmailVerified: json['isEmailVerified'] ?? false,
      createdAt: json['createdAt'] != null 
          ? DateTime.parse(json['createdAt'])
          : DateTime.now(),
      lastLogin: json['lastLogin'] != null 
          ? DateTime.parse(json['lastLogin'])
          : null,
      userType: json['userType'] ?? 'buyer',
      preferences: json['preferences'],
      avatarColor: Color(json['avatarColor'] ?? Colors.blue.value),
    );
  }

  factory UserModel.fromFirestore(DocumentSnapshot doc) {
    Map<String, dynamic> data = doc.data() as Map<String, dynamic>;
    return UserModel.fromJson(data);
  }

  UserModel copyWith({
    String? username,
    String? fullName,
    String? phoneNumber,
    String? profileImageUrl,
    bool? isEmailVerified,
    DateTime? lastLogin,
    String? userType,
    Map<String, dynamic>? preferences,
    Color? avatarColor,
  }) {
    return UserModel(
      uid: uid,
      email: email,
      username: username ?? this.username,
      fullName: fullName ?? this.fullName,
      phoneNumber: phoneNumber ?? this.phoneNumber,
      profileImageUrl: profileImageUrl ?? this.profileImageUrl,
      isEmailVerified: isEmailVerified ?? this.isEmailVerified,
      createdAt: createdAt,
      lastLogin: lastLogin ?? this.lastLogin,
      userType: userType ?? this.userType,
      preferences: preferences ?? this.preferences,
      avatarColor: avatarColor ?? this.avatarColor,
    );
  }
}
