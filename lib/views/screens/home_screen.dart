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
        title: Obx(() => Row(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            GestureDetector(
              onTap: () => videoController.switchTab(true),
              child: Container(
                padding: const EdgeInsets.only(bottom: 5),
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(
                      color: videoController.isTrendingSelected.value ? AppColors.white : Colors.transparent,
                      width: 2,
                    ),
                  ),
                ),
                child: Text(
                  'Trending',
                  style: TextStyle(
                    color: videoController.isTrendingSelected.value ? AppColors.white : AppColors.grey,
                    fontSize: videoController.isTrendingSelected.value ? 17 : 15,
                    fontWeight: videoController.isTrendingSelected.value ? FontWeight.bold : FontWeight.normal,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 20),
            GestureDetector(
              onTap: () => videoController.switchTab(false),
              child: Container(
                padding: const EdgeInsets.only(bottom: 5),
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(
                      color: !videoController.isTrendingSelected.value ? AppColors.white : Colors.transparent,
                      width: 2,
                    ),
                  ),
                ),
                child: Text(
                  'For You',
                  style: TextStyle(
                    color: !videoController.isTrendingSelected.value ? AppColors.white : AppColors.grey,
                    fontSize: !videoController.isTrendingSelected.value ? 17 : 15,
                    fontWeight: !videoController.isTrendingSelected.value ? FontWeight.bold : FontWeight.normal,
                  ),
                ),
              ),
            ),
          ],
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
}
