import 'package:get/get.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import '../models/video_model.dart';
import '../models/comment_model.dart';

class VideoController extends GetxController {
  static VideoController get instance => Get.find();

  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final RxList<VideoModel> videos = RxList<VideoModel>([]);
  final RxInt currentVideoIndex = 0.obs;
  final RxBool isLoading = false.obs;

  // Placeholder video URLs for testing (using Pexels free stock videos)
  final List<String> placeholderVideos = [
    'https://flutter.github.io/assets-for-api-docs/assets/videos/butterfly.mp4', // Flutter's test video
    'https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4', // Google test video
    'https://storage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4', // Another Google test video
  ];

  @override
  void onInit() {
    super.onInit();
    loadVideos();
  }

  // Load videos (currently with placeholder data)
  Future<void> loadVideos() async {
    try {
      isLoading.value = true;
      // For testing, create placeholder videos
      videos.value = List.generate(
        10,
        (index) => VideoModel(
          id: 'video_$index',
          userId: 'user_$index',
          title: 'Beautiful House $index',
          description: 'Check out this amazing property! 🏠✨',
          thumbnailUrl: 'https://picsum.photos/seed/$index/1080/1920',
          videoUrl: placeholderVideos[index % placeholderVideos.length],
          createdAt: DateTime.now().subtract(Duration(days: index)),
          propertyDetails: {
            'price': '\$${(500000 + (index * 100000))}',
            'location': 'City $index',
            'bedrooms': (2 + (index % 4)).toString(),
            'bathrooms': (2 + (index % 3)).toString(),
            'sqft': '${1500 + (index * 500)}',
          },
        ),
      );
    } catch (e) {
      print('Error loading videos: $e');
    } finally {
      isLoading.value = false;
    }
  }

  // Like video
  Future<void> likeVideo(String videoId) async {
    try {
      final videoIndex = videos.indexWhere((v) => v.id == videoId);
      if (videoIndex != -1) {
        final video = videos[videoIndex];
        videos[videoIndex] = VideoModel(
          id: video.id,
          userId: video.userId,
          title: video.title,
          description: video.description,
          thumbnailUrl: video.thumbnailUrl,
          videoUrl: video.videoUrl,
          likes: video.likes + 1,
          comments: video.comments,
          shares: video.shares,
          createdAt: video.createdAt,
          propertyDetails: video.propertyDetails,
        );
      }
    } catch (e) {
      print('Error liking video: $e');
    }
  }

  // Add comment
  Future<void> addComment(String videoId, String text) async {
    try {
      final comment = CommentModel(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        videoId: videoId,
        userId: 'current_user_id', // Replace with actual user ID
        text: text,
        createdAt: DateTime.now(),
      );

      final videoIndex = videos.indexWhere((v) => v.id == videoId);
      if (videoIndex != -1) {
        final video = videos[videoIndex];
        videos[videoIndex] = VideoModel(
          id: video.id,
          userId: video.userId,
          title: video.title,
          description: video.description,
          thumbnailUrl: video.thumbnailUrl,
          videoUrl: video.videoUrl,
          likes: video.likes,
          comments: video.comments + 1,
          shares: video.shares,
          createdAt: video.createdAt,
          propertyDetails: video.propertyDetails,
        );
      }
    } catch (e) {
      print('Error adding comment: $e');
    }
  }

  // Share video
  Future<void> shareVideo(String videoId) async {
    try {
      final videoIndex = videos.indexWhere((v) => v.id == videoId);
      if (videoIndex != -1) {
        final video = videos[videoIndex];
        videos[videoIndex] = VideoModel(
          id: video.id,
          userId: video.userId,
          title: video.title,
          description: video.description,
          thumbnailUrl: video.thumbnailUrl,
          videoUrl: video.videoUrl,
          likes: video.likes,
          comments: video.comments,
          shares: video.shares + 1,
          createdAt: video.createdAt,
          propertyDetails: video.propertyDetails,
        );
      }
    } catch (e) {
      print('Error sharing video: $e');
    }
  }
}
