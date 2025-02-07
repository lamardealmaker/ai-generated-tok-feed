import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../../constants.dart';
import '../../controllers/video_controller.dart';
import '../widgets/video_player_item.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final VideoController videoController = Get.put(VideoController());
  late PageController _pageController;
  bool _isPageChanging = false;

  @override
  void initState() {
    super.initState();
    _pageController = PageController();
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Obx(() => Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.3),
            borderRadius: BorderRadius.circular(25),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              _buildTabButton(
                title: 'Trending',
                isSelected: videoController.isTrendingSelected.value,
                onTap: () => videoController.switchTab(true),
              ),
              const SizedBox(width: 16),
              _buildTabButton(
                title: 'For You',
                isSelected: !videoController.isTrendingSelected.value,
                onTap: () => videoController.switchTab(false),
              ),
            ],
          ),
        )),
        centerTitle: true,
      ),
      body: Obx(
        () => videoController.isLoading.value
            ? const Center(child: CircularProgressIndicator())
            : NotificationListener<ScrollNotification>(
                onNotification: (notification) {
                  if (notification is ScrollStartNotification) {
                    setState(() => _isPageChanging = true);
                  } else if (notification is ScrollEndNotification) {
                    setState(() => _isPageChanging = false);
                  }
                  return true;
                },
                child: PageView.builder(
                  scrollDirection: Axis.vertical,
                  controller: _pageController,
                  onPageChanged: (index) {
                    videoController.onVideoIndexChanged(index);
                  },
                  itemCount: videoController.videos.length,
                  itemBuilder: (context, index) {
                    final video = videoController.videos[index];
                    return VideoPlayerItem(
                      key: Key(video.id),
                      video: video,
                      isPlaying: videoController.currentVideoIndex.value == index && !_isPageChanging,
                    );
                  },
                ),
              ),
      ),
    );
  }

  Widget _buildTabButton({
    required String title,
    required bool isSelected,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color: isSelected ? AppColors.accent : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Text(
          title,
          style: TextStyle(
            color: isSelected ? AppColors.buttonText : AppColors.white,
            fontSize: isSelected ? 16 : 15,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }
}
