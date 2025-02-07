import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../../constants.dart';
import '../../controllers/video_controller.dart';
import '../../models/video_model.dart';
import '../widgets/video_grid_item.dart';

class FavoritesScreen extends StatefulWidget {
  const FavoritesScreen({super.key});

  @override
  State<FavoritesScreen> createState() => _FavoritesScreenState();
}

class _FavoritesScreenState extends State<FavoritesScreen> {
  final VideoController videoController = Get.find<VideoController>();
  final RxList<VideoModel> favoriteVideos = <VideoModel>[].obs;
  final RxBool isLoading = true.obs;

  @override
  void initState() {
    super.initState();
    _loadFavorites();
    // Ensure main videos are loaded
    if (videoController.videos.isEmpty) {
      videoController.loadVideos();
    }
  }

  Future<void> _loadFavorites() async {
    try {
      isLoading.value = true;
      favoriteVideos.value = await videoController.loadFavoriteVideos();
    } catch (e) {
      print('Error loading favorites: $e');
    } finally {
      isLoading.value = false;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: AppColors.background,
        title: const Text(
          'Favorites',
          style: TextStyle(
            color: AppColors.white,
            fontSize: AppTheme.fontSize_xl,
            fontWeight: FontWeight.bold,
          ),
        ),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: AppColors.white),
          onPressed: () => Get.back(),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: AppColors.accent),
            onPressed: _loadFavorites,
          ),
        ],
        elevation: 0,
      ),
      body: Obx(
        () {
          if (isLoading.value) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(
                    color: AppColors.accent,
                    strokeWidth: 3,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Loading favorites...',
                    style: TextStyle(
                      color: AppColors.accent,
                      fontSize: AppTheme.fontSize_md,
                    ),
                  ),
                ],
              ),
            );
          }

          if (favoriteVideos.isEmpty) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    Icons.bookmark_border,
                    color: AppColors.accent.withOpacity(0.5),
                    size: 80,
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'No favorite properties yet',
                    style: TextStyle(
                      color: AppColors.white,
                      fontSize: AppTheme.fontSize_lg,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Save properties to view them later',
                    style: TextStyle(
                      color: AppColors.white.withOpacity(0.7),
                      fontSize: AppTheme.fontSize_md,
                    ),
                  ),
                  const SizedBox(height: 24),
                  ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.accent,
                      foregroundColor: AppColors.buttonText,
                      padding: const EdgeInsets.symmetric(
                        horizontal: AppTheme.spacing_xl,
                        vertical: AppTheme.spacing_md,
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    onPressed: () => Get.back(),
                    child: const Text(
                      'Browse Properties',
                      style: TextStyle(
                        fontSize: AppTheme.fontSize_md,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),
            );
          }

          return RefreshIndicator(
            onRefresh: _loadFavorites,
            color: AppColors.accent,
            backgroundColor: AppColors.darkGrey,
            child: GridView.builder(
              padding: const EdgeInsets.all(12),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                childAspectRatio: 0.75,
                crossAxisSpacing: 12,
                mainAxisSpacing: 12,
              ),
              itemCount: favoriteVideos.length,
              itemBuilder: (context, index) {
                final video = favoriteVideos[index];
                return VideoGridItem(
                  video: video,
                  onTap: () async {
                    // If main videos aren't loaded yet, load them
                    if (videoController.videos.isEmpty) {
                      await videoController.loadVideos();
                    }
                    
                    final mainIndex = videoController.videos
                        .indexWhere((v) => v.id == video.id);
                    
                    if (mainIndex != -1) {
                      videoController.currentVideoIndex.value = mainIndex;
                      // Navigate to main screen
                      Get.offAllNamed('/');
                    } else {
                      Get.snackbar(
                        'Error',
                        'Could not find video in main feed',
                        backgroundColor: AppColors.error.withOpacity(0.8),
                        colorText: AppColors.white,
                      );
                    }
                  },
                );
              },
            ),
          );
        },
      ),
    );
  }
}
