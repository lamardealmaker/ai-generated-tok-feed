import 'package:cloud_firestore/cloud_firestore.dart';

class CommentModel {
  final String id;
  final String videoId;
  final String userId;
  final String username;
  final String text;
  final DateTime createdAt;
  final int likes;
  final String? parentId; // For replies to comments

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

  CommentModel copyWith({
    String? id,
    String? videoId,
    String? userId,
    String? username,
    String? text,
    DateTime? createdAt,
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

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'videoId': videoId,
      'userId': userId,
      'username': username,
      'text': text,
      'createdAt': createdAt.toIso8601String(),
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
      createdAt: DateTime.parse(json['createdAt']),
      likes: json['likes'],
      parentId: json['parentId'],
    );
  }

  factory CommentModel.fromFirestore(DocumentSnapshot doc) {
    Map<String, dynamic> data = doc.data() as Map<String, dynamic>;
    return CommentModel(
      id: doc.id,
      videoId: data['videoId'] ?? '',
      userId: data['userId'] ?? '',
      username: data['username'] ?? '',
      text: data['text'] ?? '',
      createdAt: (data['createdAt'] as Timestamp).toDate(),
      likes: data['likes'] ?? 0,
      parentId: data['parentId'],
    );
  }
}
