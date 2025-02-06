import 'package:cloud_firestore/cloud_firestore.dart';

class CommentModel {
  final String id;
  final String videoId;
  final String userId;
  final String text;
  final int likes;
  final DateTime createdAt;
  final String? parentId; // For replies to comments

  CommentModel({
    required this.id,
    required this.videoId,
    required this.userId,
    required this.text,
    this.likes = 0,
    required this.createdAt,
    this.parentId,
  });

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'videoId': videoId,
      'userId': userId,
      'text': text,
      'likes': likes,
      'createdAt': createdAt.toIso8601String(),
      'parentId': parentId,
    };
  }

  factory CommentModel.fromJson(Map<String, dynamic> json) {
    return CommentModel(
      id: json['id'],
      videoId: json['videoId'],
      userId: json['userId'],
      text: json['text'],
      likes: json['likes'],
      createdAt: DateTime.parse(json['createdAt']),
      parentId: json['parentId'],
    );
  }

  factory CommentModel.fromFirestore(DocumentSnapshot doc) {
    Map<String, dynamic> data = doc.data() as Map<String, dynamic>;
    return CommentModel(
      id: doc.id,
      videoId: data['videoId'] ?? '',
      userId: data['userId'] ?? '',
      text: data['text'] ?? '',
      likes: data['likes'] ?? 0,
      createdAt: (data['createdAt'] as Timestamp).toDate(),
      parentId: data['parentId'],
    );
  }
}
