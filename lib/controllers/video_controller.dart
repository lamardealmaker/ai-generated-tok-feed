import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:video_player/video_player.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:share_plus/share_plus.dart';
import '../models/video_model.dart';
import '../models/comment_model.dart';
import '../constants.dart';

class VideoController extends GetxController {
  final RxList<VideoModel> videos = RxList<VideoModel>([]);
  final RxList<VideoModel> trendingVideos = RxList<VideoModel>([]);
  final RxList<VideoModel> forYouVideos = RxList<VideoModel>([]);
  final RxInt currentVideoIndex = 0.obs;
  final RxBool isLoading = false.obs;
  final RxBool isTrendingSelected = false.obs; // Changed from true to false
  
  // Map to store comments for each video
  final RxMap<String, List<CommentModel>> commentsMap = RxMap<String, List<CommentModel>>({});
  
  // Firebase instances
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;
  
  // Keep track of preloaded controllers
  final Map<String, VideoPlayerController> _videoControllers = {};

  @override
  void onInit() {
    super.onInit();
    loadVideos();
  }

  @override
  void onClose() {
    for (var controller in _videoControllers.values) {
      controller.dispose();
    }
    _videoControllers.clear();
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
        await _initializeVideo(0);
        
        // Pre-initialize second video if available
        if (videos.length > 1) {
          _initializeVideo(1); // Don't await this one
        }
      }
      
      isLoading.value = false;
    } catch (e) {
      print('Error loading videos: $e');
      isLoading.value = false;
    }
  }

  void switchTab(bool toTrending) {
    isTrendingSelected.value = toTrending;
    videos.value = toTrending ? trendingVideos : forYouVideos;
    currentVideoIndex.value = 0;  // Reset to first video when switching tabs
    
    // Initialize first video of the new tab immediately
    if (videos.isNotEmpty) {
      _initializeVideo(0);
      // Pre-initialize second video if available
      if (videos.length > 1) {
        _initializeVideo(1);
      }
    }
  }

  Future<void> _initializeVideo(int index) async {
    if (index < 0 || index >= videos.length) return;

    final video = videos[index];
    try {
      print('Initializing video ${video.id} at index $index'); // Debug print
      print('Video URL: ${video.videoUrl}'); // Debug print

      // Check if we already have an initialized controller
      if (_videoControllers.containsKey(video.id) && 
          _videoControllers[video.id]!.value.isInitialized) {
        print('Using existing controller for ${video.id}'); // Debug print
        return;
      }

      // Only dispose if we're creating a new controller
      if (_videoControllers.containsKey(video.id)) {
        print('Disposing existing controller for ${video.id}'); // Debug print
        await _videoControllers[video.id]!.dispose();
        _videoControllers.remove(video.id);
      }

      // Validate URL format and accessibility
      try {
        final uri = Uri.parse(video.videoUrl);
        if (!uri.isAbsolute) {
          throw ArgumentError('Invalid video URL format');
        }
      } catch (e) {
        print('Error parsing video URL for ${video.id}: $e');
        return;
      }

      print('Creating new controller for ${video.id}'); // Debug print
      final controller = VideoPlayerController.network(
        video.videoUrl,
        videoPlayerOptions: VideoPlayerOptions(
          mixWithOthers: true,
          allowBackgroundPlayback: false,
        ),
        httpHeaders: {
          'User-Agent': 'Mozilla/5.0',
          'Accept': '*/*',
        },
      );
      
      _videoControllers[video.id] = controller;
      
      print('Initializing controller for ${video.id}'); // Debug print
      await controller.initialize();
      print('Controller initialized successfully for ${video.id}'); // Debug print
      
      // Check if video loaded successfully
      if (!controller.value.isInitialized) {
        print('Error: Video controller failed to initialize for ${video.id}');
        _videoControllers.remove(video.id);
        return;
      }

      controller.setLooping(true);
      
      if (index == currentVideoIndex.value) {
        print('Playing video ${video.id} as it is the current video'); // Debug print
        await controller.play();
      }

      // Preload next video if available and not already loaded
      if (index + 1 < videos.length && !_videoControllers.containsKey(videos[index + 1].id)) {
        print('Preloading next video at index ${index + 1}'); // Debug print
        _initializeVideo(index + 1);
      }
    } catch (e) {
      print('Error initializing video ${video.id}: $e');
      // Remove failed controller
      if (_videoControllers.containsKey(video.id)) {
        await _videoControllers[video.id]!.dispose();
        _videoControllers.remove(video.id);
      }
    }
  }

  void onVideoIndexChanged(int index) {
    currentVideoIndex.value = index;
    
    // Initialize current video if not already initialized
    _initializeVideo(index);
    
    // Pre-initialize next video if available
    if (index < videos.length - 1) {
      _initializeVideo(index + 1);
    }
    
    // Clean up videos that are no longer needed
    _cleanupUnusedControllers(index);
  }

  void _cleanupUnusedControllers(int currentIndex) {
    // Keep only controllers for current, previous, and next videos
    final keysToKeep = <String>{};
    
    // Add current video
    if (currentIndex >= 0 && currentIndex < videos.length) {
      keysToKeep.add(videos[currentIndex].id);
    }
    
    // Add previous video
    if (currentIndex > 0) {
      keysToKeep.add(videos[currentIndex - 1].id);
    }
    
    // Add next video
    if (currentIndex < videos.length - 1) {
      keysToKeep.add(videos[currentIndex + 1].id);
    }
    
    // Remove controllers that are no longer needed
    _videoControllers.removeWhere((key, controller) {
      if (!keysToKeep.contains(key)) {
        controller.dispose();
        return true;
      }
      return false;
    });
  }

  VideoPlayerController? getController(String videoId) {
    final controller = _videoControllers[videoId];
    print('Getting controller for video $videoId: ${controller != null ? 'found' : 'not found'}'); // Debug print
    return controller;
  }

  Future<void> likeVideo(String videoId) async {
    try {
      if (_auth.currentUser == null) {
        Get.snackbar('Error', 'Please sign in to like videos', backgroundColor: AppColors.error.withOpacity(0.8), colorText: AppColors.white);
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
        Get.snackbar('Error', 'Please sign in to favorite videos', backgroundColor: AppColors.error.withOpacity(0.8), colorText: AppColors.white);
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
        Get.snackbar('Success', 'Added to favorites', backgroundColor: AppColors.save.withOpacity(0.8), colorText: AppColors.white);
      } else {
        // Remove from favorites
        await favoriteRef.delete();
        video.isFavorite.value = false;
      }
    } catch (e) {
      print('Error toggling favorite: $e');
      Get.snackbar('Error', 'Failed to update favorites', backgroundColor: AppColors.error.withOpacity(0.8), colorText: AppColors.white);
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
      Get.snackbar('Error', 'Failed to share video', backgroundColor: AppColors.error.withOpacity(0.8), colorText: AppColors.white);
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
      Get.snackbar('Error', 'Please login to comment', backgroundColor: AppColors.error.withOpacity(0.8), colorText: AppColors.white);
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
      Get.snackbar('Error', 'Failed to add comment', backgroundColor: AppColors.error.withOpacity(0.8), colorText: AppColors.white);
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
      Get.snackbar('Error', 'Failed to load favorites', backgroundColor: AppColors.error.withOpacity(0.8), colorText: AppColors.white);
      return [];
    }
  }
}
