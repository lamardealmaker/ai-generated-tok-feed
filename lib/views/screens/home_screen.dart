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

class _HomeScreenState extends State<HomeScreen> with SingleTickerProviderStateMixin {
  final VideoController videoController = Get.put(VideoController());
  late PageController _pageController;
  late AnimationController _fadeController;
  late Animation<double> _fadeAnimation;
  bool _isPageChanging = false;
  bool _isTabChanging = false;

  @override
  void initState() {
    super.initState();
    _initPageController();
    _initAnimationController();
  }

  void _initPageController() {
    _pageController = PageController(
      initialPage: videoController.currentVideoIndex.value,
    );
  }

  void _initAnimationController() {
    _fadeController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 150),  
    );
    _fadeAnimation = CurvedAnimation(
      parent: _fadeController,
      curve: Curves.easeOut,  
      reverseCurve: Curves.easeOut,  
    );
  }

  @override
  void dispose() {
    _pageController.dispose();
    _fadeController.dispose();
    super.dispose();
  }

  Future<void> _handleTabChange(bool toTrending) async {
    if (_isTabChanging) return;
    
    setState(() => _isTabChanging = true);
    
    // Quick fade out
    await _fadeController.forward();
    
    // Switch tab content
    await videoController.switchTab(toTrending);
    
    // Reset page controller to the saved position
    _pageController.dispose();
    _initPageController();
    
    // Quick fade in
    await _fadeController.reverse();
    
    setState(() => _isTabChanging = false);
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
                onTap: () => _handleTabChange(true),
                isEnabled: !_isTabChanging,
              ),
              const SizedBox(width: 16),
              _buildTabButton(
                title: 'For You',
                isSelected: !videoController.isTrendingSelected.value,
                onTap: () => _handleTabChange(false),
                isEnabled: !_isTabChanging,
              ),
            ],
          ),
        )),
        centerTitle: true,
      ),
      body: Obx(
        () => videoController.isLoading.value
            ? const Center(child: CircularProgressIndicator())
            : Stack(
                children: [
                  // Video Content
                  NotificationListener<ScrollNotification>(
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
                          isPlaying: videoController.currentVideoIndex.value == index && 
                                   !_isPageChanging && 
                                   !_isTabChanging,
                        );
                      },
                    ),
                  ),
                  
                  // Fade Overlay
                  if (_isTabChanging)
                    FadeTransition(
                      opacity: _fadeAnimation,
                      child: Container(
                        color: AppColors.background.withOpacity(0.8),
                        child: Center(
                          child: Container(
                            width: 60,
                            height: 60,
                            decoration: BoxDecoration(
                              color: AppColors.background.withOpacity(0.8),
                              borderRadius: BorderRadius.circular(15),
                            ),
                            child: const Padding(
                              padding: EdgeInsets.all(12),
                              child: CircularProgressIndicator(
                                strokeWidth: 3,
                                valueColor: AlwaysStoppedAnimation<Color>(AppColors.accent),
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                ],
              ),
      ),
    );
  }

  Widget _buildTabButton({
    required String title,
    required bool isSelected,
    required VoidCallback onTap,
    required bool isEnabled,
  }) {
    return GestureDetector(
      onTap: isEnabled ? onTap : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),  // Match fade duration
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        decoration: BoxDecoration(
          color: isSelected ? AppColors.accent : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Text(
          title,
          style: TextStyle(
            color: isEnabled 
                ? (isSelected ? AppColors.buttonText : AppColors.white)
                : AppColors.white.withOpacity(0.5),
            fontSize: isSelected ? 16 : 15,
            fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ),
    );
  }
}
