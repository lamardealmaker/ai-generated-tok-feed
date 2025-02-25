import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:video_player/video_player.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:share_plus/share_plus.dart';
import '../models/video_model.dart';
import '../models/comment_model.dart';
import '../constants.dart';
import '../views/widgets/loading_overlay.dart';

class VideoController extends GetxController {
  final RxList<VideoModel> videos = RxList<VideoModel>([]);
  final RxList<VideoModel> trendingVideos = RxList<VideoModel>([]);
  final RxList<VideoModel> forYouVideos = RxList<VideoModel>([]);
  final RxInt currentVideoIndex = 0.obs;
  final RxInt trendingIndex = 0.obs;
  final RxInt forYouIndex = 0.obs;
  final RxBool isLoading = false.obs;
  final RxBool isTrendingSelected = false.obs;
  final RxBool _isTabChanging = false.obs;
  
  // Store PageController reference
  PageController? _pageController;
  
  // Map to store comments for each video
  final RxMap<String, List<CommentModel>> commentsMap = RxMap<String, List<CommentModel>>({});
  
  // Firebase instances
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;
  
  // Separate controller maps for each tab
  final Map<String, VideoPlayerController> _trendingControllers = {};
  final Map<String, VideoPlayerController> _forYouControllers = {};
  
  // Getter for active controller map
  Map<String, VideoPlayerController> get _videoControllers => 
      isTrendingSelected.value ? _trendingControllers : _forYouControllers;
      
  // Track the currently playing video
  String? _currentlyPlayingId;
  
  // Preload flag
  bool _isPreloading = false;

  @override
  void onInit() {
    super.onInit();
    loadVideos();
  }

  @override
  void onClose() {
    // Dispose all controllers in both maps
    for (var controller in _trendingControllers.values) {
      controller.dispose();
    }
    for (var controller in _forYouControllers.values) {
      controller.dispose();
    }
    _trendingControllers.clear();
    _forYouControllers.clear();
    super.onClose();
  }

  Future<void> loadVideos() async {
    try {
      isLoading.value = true;
      
      // Get videos collection reference
      final videosRef = _firestore.collection('videos');
      
      // Query all videos ordered by creation date
      final videosSnapshot = await videosRef
          .orderBy('createdAt', descending: true)
          .get();

      final List<VideoModel> loadedVideos = [];

      for (var doc in videosSnapshot.docs) {
        try {
          final video = VideoModel.fromFirestore(doc);
          
          // Validate video URL format
          try {
            final uri = Uri.parse(video.videoUrl);
            if (!uri.isAbsolute || !uri.scheme.startsWith('http')) {
              continue;
            }
          } catch (e) {
            continue;
          }
          
          // Get favorite and like status for current user
          if (_auth.currentUser != null) {
            final favoriteDoc = await _firestore
                .collection('users')
                .doc(_auth.currentUser!.uid)
                .collection('favorites')
                .doc(video.id)
                .get();
            video.isFavorite.value = favoriteDoc.exists;

            final likeDoc = await _firestore
                .collection('videos')
                .doc(video.id)
                .collection('likes')
                .doc(_auth.currentUser!.uid)
                .get();
            video.isLiked.value = likeDoc.exists;
          }
          
          loadedVideos.add(video);
        } catch (e) {
          print('Error loading video: $e');
        }
      }

      // Sort videos by likes for trending
      trendingVideos.value = List.from(loadedVideos)
        ..sort((a, b) => b.likes.compareTo(a.likes));

      // For now, randomize the For You feed
      forYouVideos.value = List.from(loadedVideos)..shuffle();
      
      // Set initial videos list based on selected tab
      videos.value = isTrendingSelected.value ? trendingVideos : forYouVideos;

      if (videos.isNotEmpty) {
        // Initialize first video immediately
        await _initializeVideoController(videos[0], 0);
        
        // Pre-initialize second video if available
        if (videos.length > 1) {
          _initializeVideoController(videos[1], 1);
        }
      }
      
      isLoading.value = false;
    } catch (e) {
      print('Error loading videos: $e');
      isLoading.value = false;
    }
  }

  Future<void> switchTab(bool toTrending) async {
    _isTabChanging.value = true;
    
    // Pause and cleanup current video before switching
    final currentVideoId = videos[currentVideoIndex.value].id;
    final currentController = _videoControllers[currentVideoId];
    if (currentController != null) {
      if (currentController.value.isPlaying) {
        await currentController.pause();
      }
      await currentController.dispose();
      _videoControllers.remove(currentVideoId);
    }

    isTrendingSelected.value = toTrending;
    videos.value = toTrending ? trendingVideos : forYouVideos;
    
    // Restore previous position for the selected tab
    currentVideoIndex.value = toTrending ? trendingIndex.value : forYouIndex.value;
    
    // Initialize video at remembered position
    await _initializeVideoController(videos[currentVideoIndex.value], currentVideoIndex.value);
    
    // Pre-initialize next video if available
    if (currentVideoIndex.value < videos.length - 1) {
      _initializeVideoController(videos[currentVideoIndex.value + 1], currentVideoIndex.value + 1);
    }
    _isTabChanging.value = false;
  }

  void setPageController(PageController controller) {
    _pageController = controller;
  }

  void onVideoFinished() {
    if (_pageController == null || _isTabChanging.value) return;

    final currentList = isTrendingSelected.value ? trendingVideos : forYouVideos;
    final currentIndex = isTrendingSelected.value ? trendingIndex.value : forYouIndex.value;
    
    // Always use nextPage, even for the last video
    _pageController!.nextPage(
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeInOut,
    );
    
    // If we're on the last video, pre-initialize first video
    if (currentIndex >= currentList.length - 1) {
      _initializeVideoController(videos[0], 0);
    }
  }

  void onVideoIndexChanged(int index) async {
    if (index < 0 || index >= videos.length) return;
    
    // Pause current video before switching
    final currentVideo = videos[currentVideoIndex.value];
    if (_videoControllers.containsKey(currentVideo.id)) {
      await _videoControllers[currentVideo.id]?.pause();
    }
    
    // Update indices
    if (isTrendingSelected.value) {
      trendingIndex.value = index;
    } else {
      forYouIndex.value = index;
    }
    currentVideoIndex.value = index;
    
    // Initialize current video
    await _initializeVideoController(videos[index], index);
    
    // Pre-initialize adjacent videos
    if (index > 0) {
      _initializeVideoController(videos[index - 1], index - 1);
    }
    if (index < videos.length - 1) {
      _initializeVideoController(videos[index + 1], index + 1);
    }
    
    // Clean up videos that are no longer needed
    await _cleanupUnusedControllers();
  }

  Future<void> _cleanupUnusedControllers() async {
    final keysToKeep = <String>{};
    final currentIndex = currentVideoIndex.value;
    
    // Keep only adjacent videos
    keysToKeep.add(videos[currentIndex].id);
    if (currentIndex > 0) {
      keysToKeep.add(videos[currentIndex - 1].id);
    }
    if (currentIndex < videos.length - 1) {
      keysToKeep.add(videos[currentIndex + 1].id);
    }
    
    // Remove and dispose controllers that are no longer needed
    final keysToRemove = _videoControllers.keys.where((key) => !keysToKeep.contains(key)).toList();
    for (final key in keysToRemove) {
      final controller = _videoControllers[key];
      if (controller != null) {
        await controller.pause();
        await controller.dispose();
        _videoControllers.remove(key);
      }
    }
  }

  Future<void> _initializeVideoController(VideoModel video, int index) async {
    // If controller exists and is initialized, handle playback state
    if (_videoControllers.containsKey(video.id)) {
      final controller = _videoControllers[video.id]!;
      if (controller.value.isInitialized) {
        if (index == currentVideoIndex.value && !_isTabChanging.value) {
          // Only play if this is the current video and we're not changing tabs
          if (controller.value.isPlaying) {
            await controller.pause();
          }
          await controller.seekTo(Duration.zero);
          await controller.play();
        } else {
          // Ensure non-current videos are paused
          if (controller.value.isPlaying) {
            await controller.pause();
          }
        }
        return;
      }
    }

    try {
      final controller = VideoPlayerController.network(
        video.videoUrl,
        videoPlayerOptions: VideoPlayerOptions(mixWithOthers: false),
      );
      _videoControllers[video.id] = controller;
      
      await controller.initialize();
      controller.setLooping(false);
      await controller.setVolume(1.0);
      
      // Only play if this is the current video and we're not changing tabs
      if (index == currentVideoIndex.value && !_isTabChanging.value) {
        await controller.play();
      }
    } catch (e) {
      print('Error initializing video ${video.id}: $e');
      if (_videoControllers.containsKey(video.id)) {
        await _videoControllers[video.id]?.dispose();
        _videoControllers.remove(video.id);
      }
    }
  }

  VideoPlayerController? getController(String videoId) {
    final controller = _videoControllers[videoId];
    print('Getting controller for video $videoId: ${controller != null ? 'found' : 'not found'}'); // Debug print
    return controller;
  }

  Future<void> likeVideo(String videoId) async {
    try {
      if (_auth.currentUser == null) {
        Get.snackbar('Error', 'Please sign in to like videos', backgroundColor: AppColors.error.withOpacity(0.8), colorText: AppColors.textPrimary);
        return;
      }

      final videoIndex = videos.indexWhere((v) => v.id == videoId);
      if (videoIndex == -1) return;

      final video = videos[videoIndex];
      final likeRef = _firestore
          .collection('videos')
          .doc(videoId)
          .collection('likes')
          .doc(_auth.currentUser!.uid);

      if (!video.isLiked.value) {
        // Add like
        await likeRef.set({
          'timestamp': FieldValue.serverTimestamp(),
        });
        video.isLiked.value = true;
      } else {
        // Remove like
        await likeRef.delete();
        video.isLiked.value = false;
      }
    } catch (e) {
      print('Error toggling like: $e');
    }
  }

  Future<void> toggleFavorite(String videoId) async {
    try {
      if (_auth.currentUser == null) {
        Get.snackbar('Error', 'Please sign in to favorite videos', backgroundColor: AppColors.error.withOpacity(0.8), colorText: AppColors.textPrimary);
        return;
      }

      final videoIndex = videos.indexWhere((v) => v.id == videoId);
      if (videoIndex == -1) return;

      final video = videos[videoIndex];
      final favoriteRef = _firestore
          .collection('users')
          .doc(_auth.currentUser!.uid)
          .collection('favorites')
          .doc(videoId);

      if (!video.isFavorite.value) {
        // Add to favorites
        await favoriteRef.set({
          'timestamp': FieldValue.serverTimestamp(),
        });
        video.isFavorite.value = true;
        Get.snackbar('Success', 'Added to favorites', backgroundColor: AppColors.success.withOpacity(0.8), colorText: AppColors.textPrimary);
      } else {
        // Remove from favorites
        await favoriteRef.delete();
        video.isFavorite.value = false;
      }
    } catch (e) {
      print('Error toggling favorite: $e');
      Get.snackbar('Error', 'Failed to update favorites', backgroundColor: AppColors.error.withOpacity(0.8), colorText: AppColors.textPrimary);
    }
  }

  Future<void> shareVideo(String videoId) async {
    try {
      final video = videos.firstWhere((v) => v.id == videoId);
      final property = video.propertyDetails;
      
      if (property == null) return;

      final shareText = '''
Check out this property!
${property.fullAddress}
Price: ${property.formattedPrice}
${property.beds} beds, ${property.baths} baths
${property.squareFeet} sq ft
''';

      Share.share(shareText);
      
      // Update share count in Firestore
      await _firestore.collection('videos').doc(videoId).update({
        'shares': FieldValue.increment(1)
      });

      // Update local state
      final index = videos.indexWhere((v) => v.id == videoId);
      if (index != -1) {
        videos[index] = videos[index].copyWith(
          shares: videos[index].shares + 1
        );
      }
    } catch (e) {
      print('Error sharing video: $e');
      Get.snackbar('Error', 'Failed to share video', backgroundColor: AppColors.error.withOpacity(0.8), colorText: AppColors.textPrimary);
    }
  }

  Stream<List<CommentModel>> getCommentsStream(String videoId) {
    return _firestore
        .collection('videos')
        .doc(videoId)
        .collection('comments')
        .orderBy('createdAt', descending: true)
        .snapshots()
        .map((snapshot) {
          print('Received comment snapshot for video $videoId with ${snapshot.docs.length} comments'); // Debug
          return snapshot.docs.map((doc) => CommentModel.fromFirestore(doc)).toList();
        });
  }

  Future<void> loadComments(String videoId) async {
    try {
      print('Loading comments for video $videoId'); // Debug
      if (!commentsMap.containsKey(videoId)) {
        commentsMap[videoId] = [];
      }
    } catch (e) {
      print('Error loading comments: $e');
      commentsMap[videoId] = [];
    }
  }

  Future<void> addComment(String videoId, String text) async {
    if (_auth.currentUser == null) {
      Get.snackbar('Error', 'Please login to comment', backgroundColor: AppColors.error.withOpacity(0.8), colorText: AppColors.textPrimary);
      return;
    }

    try {
      final user = _auth.currentUser!;
      
      // Get the current user's data from Firestore
      final userDoc = await _firestore.collection('users').doc(user.uid).get();
      final userData = userDoc.data();
      final username = userData != null ? userData['username'] as String? ?? user.displayName : user.displayName;
      
      final commentRef = _firestore
          .collection('videos')
          .doc(videoId)
          .collection('comments')
          .doc();

      final timestamp = Timestamp.now();

      final comment = CommentModel(
        id: commentRef.id,
        videoId: videoId,
        userId: user.uid,
        username: username ?? 'Anonymous',
        text: text,
        createdAt: timestamp,
        parentId: null,
      );

      // Use a batch to update both the comment and the count
      final batch = _firestore.batch();
      batch.set(commentRef, comment.toJson());
      batch.update(_firestore.collection('videos').doc(videoId), {
        'comments': FieldValue.increment(1),
      });
      
      await batch.commit();
    } catch (e) {
      print('Error adding comment: $e');
      Get.snackbar('Error', 'Failed to add comment', backgroundColor: AppColors.error.withOpacity(0.8), colorText: AppColors.textPrimary);
    }
  }

  Future<List<VideoModel>> loadFavoriteVideos() async {
    if (_auth.currentUser == null) {
      return [];
    }

    try {
      final favoritesSnapshot = await _firestore
          .collection('users')
          .doc(_auth.currentUser!.uid)
          .collection('favorites')
          .get();

      final List<VideoModel> favoriteVideos = [];
      
      for (var doc in favoritesSnapshot.docs) {
        final videoDoc = await _firestore
            .collection('videos')
            .doc(doc.id)
            .get();
            
        if (videoDoc.exists) {
          final video = VideoModel.fromFirestore(videoDoc);
          favoriteVideos.add(video.copyWith(isFavorite: true));
        }
      }

      return favoriteVideos;
    } catch (e) {
      print('Error loading favorite videos: $e');
      Get.snackbar('Error', 'Failed to load favorites', backgroundColor: AppColors.error.withOpacity(0.8), colorText: AppColors.textPrimary);
      return [];
    }
  }
}
