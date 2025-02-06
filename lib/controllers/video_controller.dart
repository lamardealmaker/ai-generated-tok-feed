import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:video_player/video_player.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../models/video_model.dart';
import '../models/comment_model.dart';
import '../constants.dart';

class VideoController extends GetxController {
  final RxList<VideoModel> videos = RxList<VideoModel>([]);
  final RxInt currentVideoIndex = 0.obs;
  final RxBool isLoading = false.obs;
  final RxList<CommentModel> comments = <CommentModel>[].obs;
  
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
      
      // Query videos ordered by creation date
      final videosSnapshot = await videosRef
          .orderBy('createdAt', descending: true)
          .get();

      print('Found ${videosSnapshot.docs.length} videos in Firestore'); // Debug print
      
      // Debug: Print all document data
      for (var doc in videosSnapshot.docs) {
        final data = doc.data();
        print('Video document ${doc.id}:');
        print('  videoUrl: ${data['videoUrl']}');
        print('  thumbnailUrl: ${data['thumbnailUrl']}');
        print('  createdAt: ${data['createdAt']}');
        print('  title: ${data['title']}');
      }

      final List<VideoModel> loadedVideos = [];

      for (var doc in videosSnapshot.docs) {
        try {
          final video = VideoModel.fromFirestore(doc);
          print('Successfully loaded video: ${video.id}, URL: ${video.videoUrl}'); // Debug print
          
          // Validate video URL format
          try {
            final uri = Uri.parse(video.videoUrl);
            if (!uri.isAbsolute) {
              print('Invalid video URL format for ${video.id}: ${video.videoUrl}');
              continue;
            }
            
            // Check if URL starts with https:// or http://
            if (!uri.scheme.startsWith('http')) {
              print('Video URL must start with http:// or https:// for ${video.id}: ${video.videoUrl}');
              continue;
            }
          } catch (e) {
            print('Error parsing video URL for ${video.id}: $e');
            continue;
          }
          
          // Get favorite status for current user
          if (_auth.currentUser != null) {
            final favoriteDoc = await _firestore
                .collection('users')
                .doc(_auth.currentUser!.uid)
                .collection('favorites')
                .doc(video.id)
                .get();
            
            if (favoriteDoc.exists) {
              loadedVideos.add(video.copyWith(isFavorite: true));
            } else {
              loadedVideos.add(video);
            }
          } else {
            loadedVideos.add(video);
          }
        } catch (e) {
          print('Error loading video from doc ${doc.id}: $e'); // Debug print
          continue; // Skip this video if there's an error
        }
      }
      
      videos.value = loadedVideos;

      if (videos.isEmpty) {
        print('No videos found in the database'); // Debug print
        Get.snackbar(
          'No Videos',
          'No real estate videos are currently available.',
          backgroundColor: Colors.orange,
          colorText: Colors.white,
          duration: const Duration(seconds: 3),
        );
      } else {
        print('Found ${videos.length} valid videos, initializing first video...'); // Debug print
        // Initialize first video
        await _initializeVideo(0);
      }
    } catch (e) {
      print('Error loading videos: $e');
      Get.snackbar(
        'Error',
        'Failed to load videos. Please check your connection.',
        backgroundColor: Colors.red,
        colorText: Colors.white,
        duration: const Duration(seconds: 3),
      );
    } finally {
      isLoading.value = false;
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

  Future<void> onVideoIndexChanged(int index) async {
    if (index >= videos.length) return;

    if (currentVideoIndex.value != index && currentVideoIndex.value < videos.length) {
      final previousVideo = videos[currentVideoIndex.value];
      final previousController = _videoControllers[previousVideo.id];
      if (previousController != null) {
        print('Pausing previous video ${previousVideo.id}'); // Debug print
        await previousController.pause();
      }
    }

    currentVideoIndex.value = index;
    print('Video index changed to $index'); // Debug print

    if (!_videoControllers.containsKey(videos[index].id)) {
      print('Initializing new video at index $index'); // Debug print
      await _initializeVideo(index);
    } else {
      final currentVideo = videos[index];
      final currentController = _videoControllers[currentVideo.id];
      if (currentController != null) {
        print('Playing existing video ${currentVideo.id}'); // Debug print
        await currentController.play();
      }
    }

    // Preload next video if available
    if (index + 1 < videos.length) {
      print('Preloading next video at index ${index + 1}'); // Debug print
      _initializeVideo(index + 1);
    }
  }

  VideoPlayerController? getController(String videoId) {
    final controller = _videoControllers[videoId];
    print('Getting controller for video $videoId: ${controller != null ? 'found' : 'not found'}'); // Debug print
    return controller;
  }

  Future<void> likeVideo(String videoId) async {
    if (_auth.currentUser == null) {
      Get.snackbar(
        'Error',
        'Please login to like videos',
        backgroundColor: Colors.red,
        colorText: Colors.white,
      );
      return;
    }

    try {
      final videoRef = _firestore.collection('videos').doc(videoId);
      final userLikesRef = _firestore
          .collection('users')
          .doc(_auth.currentUser!.uid)
          .collection('likes')
          .doc(videoId);

      final likeDoc = await userLikesRef.get();
      
      await _firestore.runTransaction((transaction) async {
        final videoSnapshot = await transaction.get(videoRef);
        
        if (!likeDoc.exists) {
          // Add like
          transaction.set(userLikesRef, {
            'timestamp': FieldValue.serverTimestamp(),
          });
          
          transaction.update(videoRef, {
            'likes': (videoSnapshot.data()?['likes'] ?? 0) + 1,
          });
          
          // Update local state
          final index = videos.indexWhere((v) => v.id == videoId);
          if (index != -1) {
            videos[index] = videos[index].copyWith(likes: videos[index].likes + 1);
          }
        } else {
          // Remove like
          transaction.delete(userLikesRef);
          
          transaction.update(videoRef, {
            'likes': (videoSnapshot.data()?['likes'] ?? 1) - 1,
          });
          
          // Update local state
          final index = videos.indexWhere((v) => v.id == videoId);
          if (index != -1) {
            videos[index] = videos[index].copyWith(likes: videos[index].likes - 1);
          }
        }
      });
    } catch (e) {
      print('Error toggling like: $e');
      Get.snackbar(
        'Error',
        'Failed to like video',
        backgroundColor: Colors.red,
        colorText: Colors.white,
      );
    }
  }

  Future<void> toggleFavorite(String videoId) async {
    if (_auth.currentUser == null) {
      Get.snackbar(
        'Error',
        'Please login to favorite videos',
        backgroundColor: Colors.red,
        colorText: Colors.white,
      );
      return;
    }

    try {
      final userFavoritesRef = _firestore
          .collection('users')
          .doc(_auth.currentUser!.uid)
          .collection('favorites')
          .doc(videoId);

      final favoriteDoc = await userFavoritesRef.get();
      
      if (!favoriteDoc.exists) {
        // Add to favorites
        await userFavoritesRef.set({
          'timestamp': FieldValue.serverTimestamp(),
        });
        
        // Update local state
        final index = videos.indexWhere((v) => v.id == videoId);
        if (index != -1) {
          videos[index] = videos[index].copyWith(isFavorite: true);
        }
      } else {
        // Remove from favorites
        await userFavoritesRef.delete();
        
        // Update local state
        final index = videos.indexWhere((v) => v.id == videoId);
        if (index != -1) {
          videos[index] = videos[index].copyWith(isFavorite: false);
        }
      }
    } catch (e) {
      print('Error toggling favorite: $e');
      Get.snackbar(
        'Error',
        'Failed to update favorites',
        backgroundColor: Colors.red,
        colorText: Colors.white,
      );
    }
  }

  Future<void> shareVideo(String videoId) async {
    try {
      final videoRef = _firestore.collection('videos').doc(videoId);
      
      await videoRef.update({
        'shares': FieldValue.increment(1),
      });

      final index = videos.indexWhere((v) => v.id == videoId);
      if (index != -1) {
        videos[index] = videos[index].copyWith(shares: videos[index].shares + 1);
      }

      Get.snackbar(
        'Share',
        'Sharing ${videos[index].title}...',
        backgroundColor: Colors.blue.withOpacity(0.8),
        colorText: Colors.white,
      );
    } catch (e) {
      print('Error sharing video: $e');
      Get.snackbar(
        'Error',
        'Failed to share video',
        backgroundColor: Colors.red,
        colorText: Colors.white,
      );
    }
  }

  Future<void> loadComments(String videoId) async {
    try {
      final commentsSnapshot = await _firestore
          .collection('videos')
          .doc(videoId)
          .collection('comments')
          .orderBy('createdAt', descending: true)
          .get();

      comments.value = commentsSnapshot.docs
          .map((doc) => CommentModel.fromFirestore(doc))
          .toList();
    } catch (e) {
      print('Error loading comments: $e');
    }
  }

  Future<void> addComment(String videoId, String text) async {
    if (_auth.currentUser == null) {
      Get.snackbar(
        'Error',
        'Please login to comment',
        backgroundColor: Colors.red,
        colorText: Colors.white,
      );
      return;
    }

    try {
      final user = _auth.currentUser!;
      final commentRef = _firestore
          .collection('videos')
          .doc(videoId)
          .collection('comments')
          .doc();

      final comment = CommentModel(
        id: commentRef.id,
        videoId: videoId,
        userId: user.uid,
        username: user.displayName ?? 'Anonymous',
        text: text,
        createdAt: DateTime.now(),
        likes: 0,
      );

      await commentRef.set(comment.toJson());

      // Update comment count
      await _firestore.collection('videos').doc(videoId).update({
        'comments': FieldValue.increment(1),
      });

      // Update local state
      comments.insert(0, comment);
      final videoIndex = videos.indexWhere((v) => v.id == videoId);
      if (videoIndex != -1) {
        videos[videoIndex] = videos[videoIndex].copyWith(
          comments: videos[videoIndex].comments + 1,
        );
      }
    } catch (e) {
      print('Error adding comment: $e');
      Get.snackbar(
        'Error',
        'Failed to add comment',
        backgroundColor: Colors.red,
        colorText: Colors.white,
      );
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
      Get.snackbar(
        'Error',
        'Failed to load favorites',
        backgroundColor: Colors.red,
        colorText: Colors.white,
      );
      return [];
    }
  }
}
