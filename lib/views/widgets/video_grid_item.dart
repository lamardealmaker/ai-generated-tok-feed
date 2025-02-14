import 'package:flutter/material.dart';
import '../../models/video_model.dart';
import '../../models/property_details.dart';
import '../../constants.dart';

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
      child: Container(
        decoration: BoxDecoration(
          color: AppColors.darkGrey,
          borderRadius: BorderRadius.circular(12),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.2),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        clipBehavior: Clip.antiAlias,
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
                  child: Icon(
                    Icons.image_not_supported,
                    color: AppColors.grey,
                    size: 32,
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

            // Property details overlay
            if (video.propertyDetails != null) Positioned(
              left: 8,
              right: 8,
              bottom: 8,
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Price
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: AppColors.darkGrey.withOpacity(0.85),
                      borderRadius: BorderRadius.circular(6),
                      border: Border.all(
                        color: AppColors.accent.withOpacity(0.3),
                        width: 1,
                      ),
                    ),
                    child: Text(
                      video.propertyDetails!.formattedPrice,
                      style: const TextStyle(
                        color: AppColors.textPrimary,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        shadows: [
                          Shadow(
                            offset: Offset(0, 1),
                            blurRadius: 1,
                            color: Color(0x40000000),
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 4),
                  
                  // Location
                  Text(
                    video.propertyDetails!.fullAddress,
                    style: const TextStyle(
                      color: AppColors.white,
                      fontSize: 12,
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  
                  // Stats row
                  const SizedBox(height: 4),
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(
                        Icons.bed,
                        color: AppColors.save,
                        size: 14,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        '${video.propertyDetails!.beds}',
                        style: const TextStyle(
                          color: AppColors.white,
                          fontSize: 12,
                        ),
                      ),
                      const SizedBox(width: 8),
                      const Icon(
                        Icons.bathroom,
                        color: AppColors.save,
                        size: 14,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        '${video.propertyDetails!.baths}',
                        style: const TextStyle(
                          color: AppColors.white,
                          fontSize: 12,
                        ),
                      ),
                      const SizedBox(width: 8),
                      const Icon(
                        Icons.square_foot,
                        color: AppColors.save,
                        size: 14,
                      ),
                      const SizedBox(width: 4),
                      Text(
                        '${video.propertyDetails!.squareFeet}',
                        style: const TextStyle(
                          color: AppColors.white,
                          fontSize: 12,
                        ),
                      ),
                    ],
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
