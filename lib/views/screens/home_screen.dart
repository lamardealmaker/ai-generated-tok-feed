import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'dart:ui';
import '../../constants.dart';
import '../../controllers/video_controller.dart';
import '../widgets/video_player_item.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with SingleTickerProviderStateMixin, WidgetsBindingObserver {
  final VideoController videoController = Get.put(VideoController());
  late PageController _pageController;
  late AnimationController _fadeController;
  late Animation<double> _fadeAnimation;
  bool _isPageChanging = false;
  bool _isTabChanging = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initPageController();
    _initAnimationController();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.paused || state == AppLifecycleState.inactive) {
      // Pause video when app goes to background
      final currentVideoId = videoController.videos[videoController.currentVideoIndex.value].id;
      final currentController = videoController.getController(currentVideoId);
      currentController?.pause();
    }
  }

  void _initPageController() {
    _pageController = PageController(
      initialPage: videoController.currentVideoIndex.value,
    );
    // Give VideoController access to PageController
    videoController.setPageController(_pageController);
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
    WidgetsBinding.instance.removeObserver(this);
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
        toolbarHeight: 44, 
        title: Obx(() => Padding(
          padding: const EdgeInsets.only(top: 8), 
          child: Container(
            height: 36, 
            padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.2),
              borderRadius: BorderRadius.circular(25),
              border: Border.all(
                color: Colors.white.withOpacity(0.1),
                width: 0.5,
              ),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(25),
              child: BackdropFilter(
                filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    _buildTabButton(
                      title: 'Trending',
                      isSelected: videoController.isTrendingSelected.value,
                      onTap: () => _handleTabChange(true),
                      isEnabled: !_isTabChanging,
                    ),
                    Container(
                      height: 12,
                      width: 0.5,
                      color: Colors.white.withOpacity(0.2),
                      margin: const EdgeInsets.symmetric(horizontal: 8),
                    ),
                    _buildTabButton(
                      title: 'For You',
                      isSelected: !videoController.isTrendingSelected.value,
                      onTap: () => _handleTabChange(false),
                      isEnabled: !_isTabChanging,
                    ),
                  ],
                ),
              ),
            ),
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
                        // Convert actual page index to video index
                        final videoIndex = index % videoController.videos.length;
                        videoController.onVideoIndexChanged(videoIndex);
                      },
                      itemBuilder: (context, index) {
                        // Map infinite scroll index to actual video index
                        final videoIndex = index % videoController.videos.length;
                        final video = videoController.videos[videoIndex];
                        
                        return VideoPlayerItem(
                          key: ValueKey('video_${video.id}_$index'),
                          video: video,
                          isPlaying: !_isPageChanging && index == _pageController.page?.round(),
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
        duration: const Duration(milliseconds: 150),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
        decoration: BoxDecoration(
          color: isSelected ? Colors.white.withOpacity(0.2) : Colors.transparent,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Text(
          title,
          style: TextStyle(
            color: isEnabled 
                ? (isSelected ? Colors.white : Colors.white.withOpacity(0.8))
                : Colors.white.withOpacity(0.5),
            fontSize: 14,
            fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
            letterSpacing: 0.2,
          ),
        ),
      ),
    );
  }
}
