import 'package:cloud_firestore/cloud_firestore.dart';

class VideoModel {
  final String id;
  final String userId;
  final String title;
  final String description;
  final String thumbnailUrl;
  final String videoUrl;
  final int likes;
  final int comments;
  final int shares;
  final bool isFavorite;
  final bool isLiked;
  final DateTime createdAt;
  final Map<String, dynamic> propertyDetails;

  VideoModel({
    required this.id,
    required this.userId,
    required this.title,
    required this.description,
    required this.thumbnailUrl,
    required this.videoUrl,
    this.likes = 0,
    this.comments = 0,
    this.shares = 0,
    this.isFavorite = false,
    this.isLiked = false,
    required this.createdAt,
    required this.propertyDetails,
  }) {
    // Validate video URL
    if (videoUrl.isEmpty) {
      throw ArgumentError('Video URL cannot be empty');
    }
    if (!Uri.parse(videoUrl).isAbsolute) {
      throw ArgumentError('Invalid video URL format');
    }
  }

  VideoModel copyWith({
    String? id,
    String? userId,
    String? title,
    String? description,
    String? thumbnailUrl,
    String? videoUrl,
    int? likes,
    int? comments,
    int? shares,
    bool? isFavorite,
    bool? isLiked,
    DateTime? createdAt,
    Map<String, dynamic>? propertyDetails,
  }) {
    return VideoModel(
      id: id ?? this.id,
      userId: userId ?? this.userId,
      title: title ?? this.title,
      description: description ?? this.description,
      thumbnailUrl: thumbnailUrl ?? this.thumbnailUrl,
      videoUrl: videoUrl ?? this.videoUrl,
      likes: likes ?? this.likes,
      comments: comments ?? this.comments,
      shares: shares ?? this.shares,
      isFavorite: isFavorite ?? this.isFavorite,
      isLiked: isLiked ?? this.isLiked,
      createdAt: createdAt ?? this.createdAt,
      propertyDetails: propertyDetails ?? this.propertyDetails,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'userId': userId,
      'title': title,
      'description': description,
      'thumbnailUrl': thumbnailUrl,
      'videoUrl': videoUrl,
      'likes': likes,
      'comments': comments,
      'shares': shares,
      'isFavorite': isFavorite,
      'isLiked': isLiked,
      'createdAt': createdAt.toIso8601String(),
      'propertyDetails': propertyDetails,
      'status': 'active', // Add status field
    };
  }

  factory VideoModel.fromJson(Map<String, dynamic> json) {
    if (json['videoUrl'] == null || json['videoUrl'].toString().isEmpty) {
      throw ArgumentError('Video URL is required');
    }
    
    return VideoModel(
      id: json['id'],
      userId: json['userId'],
      title: json['title'],
      description: json['description'],
      thumbnailUrl: json['thumbnailUrl'],
      videoUrl: json['videoUrl'],
      likes: json['likes'],
      comments: json['comments'],
      shares: json['shares'],
      isFavorite: json['isFavorite'],
      isLiked: json['isLiked'] ?? false,
      createdAt: DateTime.parse(json['createdAt']),
      propertyDetails: json['propertyDetails'],
    );
  }

  factory VideoModel.fromFirestore(DocumentSnapshot doc) {
    Map<String, dynamic> data = doc.data() as Map<String, dynamic>;
    
    // Validate required fields
    if (data['videoUrl'] == null || data['videoUrl'].toString().isEmpty) {
      throw ArgumentError('Video URL is required for video ${doc.id}');
    }

    try {
      // Handle createdAt that could be either Timestamp or String
      DateTime createdAt;
      final createdAtField = data['createdAt'];
      if (createdAtField is Timestamp) {
        createdAt = createdAtField.toDate();
      } else if (createdAtField is String) {
        createdAt = DateTime.parse(createdAtField);
      } else {
        createdAt = DateTime.now(); // Fallback if field is missing or invalid
        print('Warning: Invalid createdAt format for video ${doc.id}, using current time');
      }

      return VideoModel(
        id: doc.id,
        userId: data['userId'] ?? '',
        title: data['title'] ?? '',
        description: data['description'] ?? '',
        thumbnailUrl: data['thumbnailUrl'] ?? '',
        videoUrl: data['videoUrl'],  // No default value, will throw if missing
        likes: (data['likes'] ?? 0).toInt(),
        comments: (data['comments'] ?? 0).toInt(),
        shares: (data['shares'] ?? 0).toInt(),
        isFavorite: data['isFavorite'] ?? false,
        isLiked: data['isLiked'] ?? false,
        createdAt: createdAt,
        propertyDetails: Map<String, dynamic>.from(data['propertyDetails'] ?? {}),
      );
    } catch (e) {
      print('Error creating VideoModel from document ${doc.id}: $e');
      rethrow;
    }
  }
}
