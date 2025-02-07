import 'package:cloud_firestore/cloud_firestore.dart';

class CommentModel {
  final String id;
  final String videoId;
  final String userId;
  final String username;
  final String text;
  final dynamic createdAt;  // Keep as dynamic to handle both DateTime and Timestamp
  final int likes;
  final String? parentId;

  CommentModel({
    required this.id,
    required this.videoId,
    required this.userId,
    required this.username,
    required this.text,
    required this.createdAt,
    required this.likes,
    this.parentId,
  });

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'videoId': videoId,
      'userId': userId,
      'username': username,
      'text': text,
      'createdAt': createdAt is DateTime ? Timestamp.fromDate(createdAt as DateTime) : createdAt,
      'likes': likes,
      'parentId': parentId,
    };
  }

  factory CommentModel.fromJson(Map<String, dynamic> json) {
    return CommentModel(
      id: json['id'],
      videoId: json['videoId'],
      userId: json['userId'],
      username: json['username'],
      text: json['text'],
      createdAt: json['createdAt'],
      likes: json['likes'],
      parentId: json['parentId'],
    );
  }

  factory CommentModel.fromFirestore(DocumentSnapshot doc) {
    Map<String, dynamic> data = doc.data() as Map<String, dynamic>;
    var createdAtData = data['createdAt'];
    
    // Handle different timestamp formats
    DateTime timestamp;
    if (createdAtData is Timestamp) {
      timestamp = createdAtData.toDate();
    } else if (createdAtData is String) {
      timestamp = DateTime.parse(createdAtData);
    } else {
      timestamp = DateTime.now(); // Fallback
    }

    return CommentModel(
      id: doc.id,
      videoId: data['videoId'] ?? '',
      userId: data['userId'] ?? '',
      username: data['username'] ?? '',
      text: data['text'] ?? '',
      createdAt: timestamp,
      likes: data['likes'] ?? 0,
      parentId: data['parentId'],
    );
  }

  DateTime get createdAtDate {
    if (createdAt is DateTime) return createdAt as DateTime;
    if (createdAt is Timestamp) return createdAt.toDate();
    if (createdAt is String) return DateTime.parse(createdAt);
    return DateTime.now(); // Fallback
  }

  CommentModel copyWith({
    String? id,
    String? videoId,
    String? userId,
    String? username,
    String? text,
    dynamic createdAt,
    int? likes,
    String? parentId,
  }) {
    return CommentModel(
      id: id ?? this.id,
      videoId: videoId ?? this.videoId,
      userId: userId ?? this.userId,
      username: username ?? this.username,
      text: text ?? this.text,
      createdAt: createdAt ?? this.createdAt,
      likes: likes ?? this.likes,
      parentId: parentId ?? this.parentId,
    );
  }
}
