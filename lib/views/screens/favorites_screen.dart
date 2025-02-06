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
            icon: const Icon(Icons.refresh, color: AppColors.white),
            onPressed: _loadFavorites,
          ),
        ],
        elevation: 0,
      ),
      body: Obx(
        () {
          if (isLoading.value) {
            return const Center(
              child: CircularProgressIndicator(
                color: AppColors.accent,
              ),
            );
          }

          if (favoriteVideos.isEmpty) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(
                    Icons.bookmark_border,
                    color: AppColors.grey,
                    size: 64,
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'No favorite videos yet',
                    style: TextStyle(
                      color: AppColors.white,
                      fontSize: AppTheme.fontSize_lg,
                    ),
                  ),
                  const SizedBox(height: 8),
                  TextButton(
                    onPressed: () => Get.back(),
                    child: const Text(
                      'Browse videos',
                      style: TextStyle(
                        color: AppColors.accent,
                        fontSize: AppTheme.fontSize_md,
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
              padding: const EdgeInsets.all(8),
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 3,
                childAspectRatio: 0.8,
                crossAxisSpacing: 2,
                mainAxisSpacing: 2,
              ),
              itemCount: favoriteVideos.length,
              itemBuilder: (context, index) {
                final video = favoriteVideos[index];
                return VideoGridItem(
                  video: video,
                  onTap: () {
                    // Find the index of this video in the main video list
                    final mainIndex = videoController.videos
                        .indexWhere((v) => v.id == video.id);
                    if (mainIndex != -1) {
                      // Update current index and navigate back to main screen
                      videoController.currentVideoIndex.value = mainIndex;
                      Get.back();
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
