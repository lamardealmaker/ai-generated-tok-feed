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
  bool _isTrendingSelected = true;

  void _switchTab(bool isTrending) {
    setState(() {
      _isTrendingSelected = isTrending;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            GestureDetector(
              onTap: () => _switchTab(true),
              child: Container(
                padding: const EdgeInsets.only(bottom: 5),
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(
                      color: _isTrendingSelected ? AppColors.white : Colors.transparent,
                      width: 2,
                    ),
                  ),
                ),
                child: Text(
                  'Trending',
                  style: TextStyle(
                    color: _isTrendingSelected ? AppColors.white : AppColors.grey,
                    fontSize: _isTrendingSelected ? 17 : 15,
                    fontWeight: _isTrendingSelected ? FontWeight.bold : FontWeight.normal,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 20),
            GestureDetector(
              onTap: () => _switchTab(false),
              child: Container(
                padding: const EdgeInsets.only(bottom: 5),
                decoration: BoxDecoration(
                  border: Border(
                    bottom: BorderSide(
                      color: !_isTrendingSelected ? AppColors.white : Colors.transparent,
                      width: 2,
                    ),
                  ),
                ),
                child: Text(
                  'For You',
                  style: TextStyle(
                    color: !_isTrendingSelected ? AppColors.white : AppColors.grey,
                    fontSize: !_isTrendingSelected ? 17 : 15,
                    fontWeight: !_isTrendingSelected ? FontWeight.bold : FontWeight.normal,
                  ),
                ),
              ),
            ),
          ],
        ),
        centerTitle: true,
      ),
      body: Obx(
        () => videoController.isLoading.value
            ? const Center(child: CircularProgressIndicator())
            : PageView.builder(
                scrollDirection: Axis.vertical,
                controller: PageController(initialPage: 0, viewportFraction: 1),
                itemCount: videoController.videos.length,
                onPageChanged: (index) {
                  videoController.currentVideoIndex.value = index;
                },
                itemBuilder: (context, index) {
                  final video = videoController.videos[index];
                  return VideoPlayerItem(
                    video: video,
                    isPlaying: videoController.currentVideoIndex.value == index,
                  );
                },
              ),
      ),
    );
  }
}
