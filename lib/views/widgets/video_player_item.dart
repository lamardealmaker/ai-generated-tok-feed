import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'package:get/get.dart';
import '../../models/video_model.dart';
import '../../controllers/video_controller.dart';
import '../../constants.dart';

class VideoPlayerItem extends StatefulWidget {
  final VideoModel video;
  final bool isPlaying;

  const VideoPlayerItem({
    super.key,
    required this.video,
    required this.isPlaying,
  });

  @override
  State<VideoPlayerItem> createState() => _VideoPlayerItemState();
}

class _VideoPlayerItemState extends State<VideoPlayerItem> {
  late VideoPlayerController _videoPlayerController;
  final VideoController videoController = Get.find<VideoController>();

  @override
  void initState() {
    super.initState();
    _initializeVideo();
  }

  void _initializeVideo() {
    _videoPlayerController = VideoPlayerController.network(widget.video.videoUrl)
      ..initialize().then((_) {
        if (mounted) {
          setState(() {});
        }
        if (widget.isPlaying) {
          _videoPlayerController.play();
          _videoPlayerController.setLooping(true);
        }
      }).catchError((error) {
        print('Error initializing video: $error');
        if (mounted) {
          setState(() {});
        }
      });
  }

  @override
  void didUpdateWidget(VideoPlayerItem oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isPlaying != oldWidget.isPlaying) {
      if (widget.isPlaying) {
        _videoPlayerController.play();
      } else {
        _videoPlayerController.pause();
      }
    }
  }

  @override
  void dispose() {
    _videoPlayerController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        // Video Player
        _videoPlayerController.value.isInitialized
            ? VideoPlayer(_videoPlayerController)
            : Container(
                color: AppColors.darkGrey,
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const CircularProgressIndicator(),
                    const SizedBox(height: 16),
                    Text(
                      'Loading video...',
                      style: TextStyle(
                        color: AppColors.white,
                        fontSize: AppTheme.fontSize_md,
                      ),
                    ),
                  ],
                ),
              ),

        // Property Details Overlay
        Positioned(
          left: 0,
          bottom: 0,
          right: 0,
          child: Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.bottomCenter,
                end: Alignment.topCenter,
                colors: [
                  Colors.black.withOpacity(0.7),
                  Colors.transparent,
                ],
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  widget.video.title,
                  style: const TextStyle(
                    color: AppColors.white,
                    fontSize: AppTheme.fontSize_lg,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  '${widget.video.propertyDetails['price']} • ${widget.video.propertyDetails['location']}',
                  style: const TextStyle(
                    color: AppColors.white,
                    fontSize: AppTheme.fontSize_md,
                  ),
                ),
                Text(
                  '${widget.video.propertyDetails['bedrooms']} beds • ${widget.video.propertyDetails['bathrooms']} baths • ${widget.video.propertyDetails['sqft']} sqft',
                  style: const TextStyle(
                    color: AppColors.white,
                    fontSize: AppTheme.fontSize_sm,
                  ),
                ),
              ],
            ),
          ),
        ),

        // Interaction Buttons
        Positioned(
          right: 10,
          bottom: 100,
          child: Column(
            children: [
              // Like Button
              IconButton(
                onPressed: () => videoController.likeVideo(widget.video.id),
                icon: const Icon(Icons.favorite, color: AppColors.white),
              ),
              Text(
                widget.video.likes.toString(),
                style: const TextStyle(color: AppColors.white),
              ),
              const SizedBox(height: 20),

              // Comment Button
              IconButton(
                onPressed: () {
                  // Show comments sheet
                  _showCommentsSheet(context);
                },
                icon: const Icon(Icons.comment, color: AppColors.white),
              ),
              Text(
                widget.video.comments.toString(),
                style: const TextStyle(color: AppColors.white),
              ),
              const SizedBox(height: 20),

              // Share Button
              IconButton(
                onPressed: () => videoController.shareVideo(widget.video.id),
                icon: const Icon(Icons.share, color: AppColors.white),
              ),
              Text(
                widget.video.shares.toString(),
                style: const TextStyle(color: AppColors.white),
              ),
            ],
          ),
        ),
      ],
    );
  }

  void _showCommentsSheet(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: AppColors.background,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.6,
        minChildSize: 0.2,
        maxChildSize: 0.8,
        builder: (_, controller) => Column(
          children: [
            Container(
              height: 4,
              width: 40,
              margin: const EdgeInsets.symmetric(vertical: 8),
              decoration: BoxDecoration(
                color: AppColors.grey,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            Expanded(
              child: ListView.builder(
                controller: controller,
                itemCount: 10, // Replace with actual comments
                itemBuilder: (context, index) => ListTile(
                  leading: const CircleAvatar(
                    backgroundColor: AppColors.accent,
                    child: Icon(Icons.person, color: AppColors.white),
                  ),
                  title: Text(
                    'User $index',
                    style: const TextStyle(color: AppColors.white),
                  ),
                  subtitle: Text(
                    'This is a sample comment $index',
                    style: const TextStyle(color: AppColors.grey),
                  ),
                ),
              ),
            ),
            Padding(
              padding: EdgeInsets.only(
                bottom: MediaQuery.of(context).viewInsets.bottom,
                left: 16,
                right: 16,
                top: 8,
              ),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      style: const TextStyle(color: AppColors.white),
                      decoration: InputDecoration(
                        hintText: 'Add a comment...',
                        hintStyle: TextStyle(color: AppColors.grey),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(20),
                        ),
                        filled: true,
                        fillColor: AppColors.darkGrey,
                      ),
                    ),
                  ),
                  IconButton(
                    onPressed: () {
                      // Add comment logic
                    },
                    icon: const Icon(Icons.send, color: AppColors.accent),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
