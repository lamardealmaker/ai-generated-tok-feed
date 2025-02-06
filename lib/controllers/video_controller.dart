import 'package:get/get.dart';
import 'package:video_player/video_player.dart';
import '../models/video_model.dart';
import '../models/comment_model.dart';

class VideoController extends GetxController {
  final RxList<VideoModel> videos = RxList<VideoModel>([]);
  final RxInt currentVideoIndex = 0.obs;
  final RxBool isLoading = false.obs;
  
  // Keep track of preloaded controllers
  final Map<String, VideoPlayerController> _videoControllers = {};

  @override
  void onInit() {
    super.onInit();
    loadVideos();
  }

  @override
  void onClose() {
    // Dispose all controllers
    for (var controller in _videoControllers.values) {
      controller.dispose();
    }
    _videoControllers.clear();
    super.onClose();
  }

  Future<void> loadVideos() async {
    try {
      isLoading.value = true;
      
      // Sample video URLs that are known to work with autoplay
      final List<String> videoUrls = [
        'https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4',
        'https://storage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4',
        'https://storage.googleapis.com/gtv-videos-bucket/sample/WeAreGoingOnBullrun.mp4',
      ];

      videos.value = videoUrls.asMap().entries.map((entry) {
        return VideoModel(
          id: 'video_${entry.key}',
          userId: 'user_1',
          title: 'Sample Property ${entry.key + 1}',
          description: 'Beautiful property with amazing views',
          thumbnailUrl: 'https://picsum.photos/seed/${entry.key}/300/400',
          videoUrl: entry.value,
          likes: 0,
          comments: 0,
          shares: 0,
          createdAt: DateTime.now(),
          propertyDetails: {
            'price': '\$${(500000 + entry.key * 100000).toString()}',
            'location': 'San Francisco, CA',
            'bedrooms': '${3 + entry.key}',
            'bathrooms': '${2 + entry.key}',
            'sqft': '${2000 + entry.key * 500}',
          },
        );
      }).toList();

      // Initialize first video
      if (videos.isNotEmpty) {
        await _initializeVideo(0);
      }
    } catch (e) {
      print('Error loading videos: $e');
    } finally {
      isLoading.value = false;
    }
  }

  Future<void> _initializeVideo(int index) async {
    if (index < 0 || index >= videos.length) return;

    final video = videos[index];
    try {
      // Dispose previous controller if it exists
      if (_videoControllers.containsKey(video.id)) {
        await _videoControllers[video.id]!.dispose();
        _videoControllers.remove(video.id);
      }

      // Create and initialize new controller
      final controller = VideoPlayerController.network(
        video.videoUrl,
        videoPlayerOptions: VideoPlayerOptions(
          mixWithOthers: true,
          allowBackgroundPlayback: false,
        ),
      );
      _videoControllers[video.id] = controller;
      
      await controller.initialize();
      controller.setLooping(true);
      
      // Only play if it's the current video
      if (index == currentVideoIndex.value) {
        controller.play();
      }
    } catch (e) {
      print('Error initializing video $index: $e');
    }
  }

  Future<void> onVideoIndexChanged(int index) async {
    // Pause previous video
    if (currentVideoIndex.value != index && currentVideoIndex.value < videos.length) {
      final previousVideo = videos[currentVideoIndex.value];
      final previousController = _videoControllers[previousVideo.id];
      await previousController?.pause();
    }

    // Update current index
    currentVideoIndex.value = index;

    // Initialize next video if not already initialized
    if (!_videoControllers.containsKey(videos[index].id)) {
      await _initializeVideo(index);
    } else {
      // Play current video
      final currentVideo = videos[index];
      final currentController = _videoControllers[currentVideo.id];
      if (currentController != null) {
        currentController.play();
      }
    }

    // Preload next video
    if (index + 1 < videos.length) {
      _initializeVideo(index + 1);
    }
  }

  VideoPlayerController? getController(String videoId) {
    return _videoControllers[videoId];
  }

  void likeVideo(String videoId) {
    final index = videos.indexWhere((video) => video.id == videoId);
    if (index != -1) {
      final video = videos[index];
      videos[index] = video.copyWith(likes: video.likes + 1);
    }
  }

  void shareVideo(String videoId) {
    final index = videos.indexWhere((video) => video.id == videoId);
    if (index != -1) {
      final video = videos[index];
      videos[index] = video.copyWith(shares: video.shares + 1);
    }
  }

  void addComment(String videoId, String text) {
    final index = videos.indexWhere((video) => video.id == videoId);
    if (index != -1) {
      final video = videos[index];
      videos[index] = video.copyWith(comments: video.comments + 1);
    }
  }
}
