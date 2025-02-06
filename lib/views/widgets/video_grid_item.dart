import 'package:flutter/material.dart';
import '../../constants.dart';
import '../../models/video_model.dart';

class VideoGridItem extends StatelessWidget {
  final VideoModel video;
  final VoidCallback onTap;

  const VideoGridItem({
    super.key,
    required this.video,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Stack(
        fit: StackFit.expand,
        children: [
          // Thumbnail
          Image.network(
            video.thumbnailUrl,
            fit: BoxFit.cover,
            errorBuilder: (context, error, stackTrace) {
              return Container(
                color: AppColors.darkGrey,
                child: const Icon(
                  Icons.error_outline,
                  color: AppColors.white,
                ),
              );
            },
          ),

          // Gradient overlay
          Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: [
                  Colors.transparent,
                  Colors.black.withOpacity(0.7),
                ],
              ),
            ),
          ),

          // Video info
          Positioned(
            left: 8,
            right: 8,
            bottom: 8,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  video.propertyDetails['price'] ?? '',
                  style: const TextStyle(
                    color: AppColors.white,
                    fontSize: AppTheme.fontSize_sm,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                Text(
                  video.propertyDetails['location'] ?? '',
                  style: const TextStyle(
                    color: AppColors.white,
                    fontSize: AppTheme.fontSize_xs,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),

          // Play icon overlay
          Center(
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.5),
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.play_arrow,
                color: AppColors.white,
                size: 24,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
