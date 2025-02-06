import 'package:cloud_firestore/cloud_firestore.dart';

class VideoModel {
  final String id;
  final String userId;
  final String title;
  final String description;
  final String thumbnailUrl;
  final String videoUrl;  // This will be placeholder for now
  final int likes;
  final int comments;
  final int shares;
  final DateTime createdAt;
  final Map<String, dynamic> propertyDetails; // Real estate specific data

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
    required this.createdAt,
    required this.propertyDetails,
  });

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
      'createdAt': createdAt.toIso8601String(),
      'propertyDetails': propertyDetails,
    };
  }

  factory VideoModel.fromJson(Map<String, dynamic> json) {
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
      createdAt: DateTime.parse(json['createdAt']),
      propertyDetails: json['propertyDetails'],
    );
  }

  factory VideoModel.fromFirestore(DocumentSnapshot doc) {
    Map<String, dynamic> data = doc.data() as Map<String, dynamic>;
    return VideoModel(
      id: doc.id,
      userId: data['userId'] ?? '',
      title: data['title'] ?? '',
      description: data['description'] ?? '',
      thumbnailUrl: data['thumbnailUrl'] ?? '',
      videoUrl: data['videoUrl'] ?? '',
      likes: data['likes'] ?? 0,
      comments: data['comments'] ?? 0,
      shares: data['shares'] ?? 0,
      createdAt: (data['createdAt'] as Timestamp).toDate(),
      propertyDetails: data['propertyDetails'] ?? {},
    );
  }
}
