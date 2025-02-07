import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'package:get/get.dart';
import '../../models/video_model.dart';
import '../../controllers/video_controller.dart';
import '../../models/comment_model.dart';
import '../../models/property_details.dart';
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
  final VideoController videoController = Get.find<VideoController>();
  VideoPlayerController? _videoPlayerController;
  bool _isInitialized = false;

  @override
  void initState() {
    super.initState();
    _initializeVideo();
  }

  void _initializeVideo() {
    _videoPlayerController = videoController.getController(widget.video.id);
    if (_videoPlayerController != null) {
      setState(() {
        _isInitialized = true;
      });
    }
  }

  @override
  void didUpdateWidget(VideoPlayerItem oldWidget) {
    super.didUpdateWidget(oldWidget);
    
    // Check if we need to update the controller
    final newController = videoController.getController(widget.video.id);
    if (newController != _videoPlayerController) {
      setState(() {
        _videoPlayerController = newController;
        _isInitialized = newController != null;
      });
    }

    // Handle play/pause state changes
    if (widget.isPlaying != oldWidget.isPlaying) {
      if (widget.isPlaying) {
        _videoPlayerController?.play();
      } else {
        _videoPlayerController?.pause();
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        // Video Player
        _isInitialized && _videoPlayerController != null
            ? AspectRatio(
                aspectRatio: _videoPlayerController!.value.aspectRatio,
                child: VideoPlayer(_videoPlayerController!),
              )
            : Container(
                color: AppColors.darkGrey,
                child: const Center(
                  child: CircularProgressIndicator(
                    color: AppColors.accent,
                  ),
                ),
              ),

        // Video Controls
        GestureDetector(
          onTap: () {
            if (_videoPlayerController?.value.isPlaying ?? false) {
              _videoPlayerController?.pause();
            } else {
              _videoPlayerController?.play();
            }
            setState(() {});
          },
          child: Container(
            color: Colors.transparent,
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
                  '${widget.video.propertyDetails?.formattedPrice} • ${widget.video.propertyDetails?.city}, ${widget.video.propertyDetails?.state}',
                  style: const TextStyle(
                    color: AppColors.white,
                    fontSize: AppTheme.fontSize_md,
                  ),
                ),
                Text(
                  '${widget.video.propertyDetails?.beds} beds • ${widget.video.propertyDetails?.baths} baths • ${widget.video.propertyDetails?.squareFeet} sqft',
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
          child: Padding(
            padding: const EdgeInsets.only(right: 8.0),
            child: Column(
              children: [
                // Like Button and Count
                Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    IconButton(
                      onPressed: () => videoController.likeVideo(widget.video.id),
                      icon: Obx(() => Icon(
                        widget.video.isLiked.value ? Icons.favorite : Icons.favorite_border,
                        color: widget.video.isLiked.value ? Colors.red : AppColors.white,
                        size: 30,
                      )),
                    ),
                    Text(
                      widget.video.likes.toString(),
                      style: const TextStyle(
                        color: AppColors.white,
                        fontSize: AppTheme.fontSize_sm,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),

                // Comment Button and Count
                Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    IconButton(
                      onPressed: () {
                        _showCommentsSheet(context);
                      },
                      icon: const Icon(
                        Icons.comment,
                        color: AppColors.white,
                        size: 30,
                      ),
                    ),
                    Text(
                      widget.video.comments.toString(),
                      style: const TextStyle(
                        color: AppColors.white,
                        fontSize: AppTheme.fontSize_sm,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),

                // Share Button and Count
                Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    IconButton(
                      onPressed: () => videoController.shareVideo(widget.video.id),
                      icon: const Icon(
                        Icons.share,
                        color: AppColors.white,
                        size: 30,
                      ),
                    ),
                    Text(
                      widget.video.shares.toString(),
                      style: const TextStyle(
                        color: AppColors.white,
                        fontSize: AppTheme.fontSize_sm,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),

                // Property Info Button
                Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    IconButton(
                      onPressed: () {
                        _showPropertySheet(context);
                      },
                      icon: const Icon(
                        Icons.info_outline,
                        color: AppColors.white,
                        size: 30,
                      ),
                    ),
                    const Text(
                      'Info',
                      style: TextStyle(
                        color: AppColors.white,
                        fontSize: AppTheme.fontSize_sm,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),

                // Favorite Button
                Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    IconButton(
                      onPressed: () => videoController.toggleFavorite(widget.video.id),
                      icon: Obx(() => Icon(
                        widget.video.isFavorite.value ? Icons.bookmark : Icons.bookmark_border,
                        color: widget.video.isFavorite.value ? Colors.blue : AppColors.white,
                        size: 30,
                      )),
                    ),
                    const Text(
                      'Save',
                      style: TextStyle(
                        color: AppColors.white,
                        fontSize: AppTheme.fontSize_sm,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  void _showCommentsSheet(BuildContext context) {
    final TextEditingController commentController = TextEditingController();
    
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.black,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => Container(
        height: MediaQuery.of(context).size.height * 0.75,
        padding: EdgeInsets.only(
          bottom: MediaQuery.of(context).viewInsets.bottom,
        ),
        child: Column(
          children: [
            // Handle and title
            Container(
              padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 16),
              decoration: BoxDecoration(
                border: Border(
                  bottom: BorderSide(
                    color: Colors.grey[900]!,
                    width: 0.5,
                  ),
                ),
              ),
              child: Column(
                children: [
                  Container(
                    height: 4,
                    width: 40,
                    decoration: BoxDecoration(
                      color: Colors.grey[600],
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'Comments',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),
            ),
            
            // Comments list
            Expanded(
              child: StreamBuilder<List<CommentModel>>(
                stream: videoController.getCommentsStream(widget.video.id),
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.waiting) {
                    return const Center(
                      child: SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: AppColors.accent,
                        ),
                      ),
                    );
                  }
                  
                  if (snapshot.hasError) {
                    return Center(
                      child: Text(
                        'Error loading comments',
                        style: TextStyle(color: Colors.grey[600]),
                      ),
                    );
                  }
                  
                  final comments = snapshot.data ?? [];
                  
                  if (comments.isEmpty) {
                    return Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.chat_bubble_outline,
                            color: Colors.grey[600],
                            size: 40,
                          ),
                          const SizedBox(height: 8),
                          Text(
                            'No comments yet\nBe the first to comment!',
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              color: Colors.grey[600],
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    );
                  }
                  
                  return ListView.builder(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    itemCount: comments.length,
                    itemBuilder: (context, index) {
                      final comment = comments[index];
                      return Padding(
                        padding: const EdgeInsets.symmetric(
                          vertical: 8,
                          horizontal: 16,
                        ),
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            // Avatar
                            CircleAvatar(
                              radius: 16,
                              backgroundColor: AppColors.accent,
                              child: Text(
                                comment.username[0].toUpperCase(),
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 14,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                            const SizedBox(width: 12),
                            
                            // Comment content
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    comment.username,
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 13,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    comment.text,
                                    style: const TextStyle(
                                      color: Colors.white,
                                      fontSize: 14,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    _getTimeAgo(comment.createdAtDate),
                                    style: TextStyle(
                                      color: Colors.grey[600],
                                      fontSize: 12,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  );
                },
              ),
            ),
            
            // Comment input
            Container(
              padding: const EdgeInsets.symmetric(
                horizontal: 16,
                vertical: 8,
              ),
              decoration: BoxDecoration(
                color: Colors.grey[900],
                border: Border(
                  top: BorderSide(
                    color: Colors.grey[800]!,
                    width: 0.5,
                  ),
                ),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: commentController,
                      style: const TextStyle(color: Colors.white),
                      decoration: InputDecoration(
                        hintText: 'Add a comment...',
                        hintStyle: TextStyle(color: Colors.grey[600]),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(20),
                          borderSide: BorderSide.none,
                        ),
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 8,
                        ),
                        filled: true,
                        fillColor: Colors.grey[850],
                      ),
                      maxLines: null,
                      textInputAction: TextInputAction.send,
                      onSubmitted: (text) {
                        if (text.isNotEmpty) {
                          videoController.addComment(
                            widget.video.id,
                            text,
                          );
                          commentController.clear();
                          FocusScope.of(context).unfocus();
                        }
                      },
                    ),
                  ),
                  const SizedBox(width: 8),
                  IconButton(
                    onPressed: () {
                      if (commentController.text.isNotEmpty) {
                        videoController.addComment(
                          widget.video.id,
                          commentController.text,
                        );
                        commentController.clear();
                        FocusScope.of(context).unfocus();
                      }
                    },
                    icon: const Icon(
                      Icons.send_rounded,
                      color: AppColors.accent,
                    ),
                    constraints: const BoxConstraints(
                      minWidth: 40,
                      minHeight: 40,
                    ),
                    padding: EdgeInsets.zero,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _showPropertySheet(BuildContext context) {
    if (widget.video.propertyDetails == null) return;
    
    final property = widget.video.propertyDetails!;  // Safe to use ! after null check
    
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.black,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => Container(
        height: MediaQuery.of(context).size.height * 0.75,
        padding: const EdgeInsets.symmetric(horizontal: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Handle and title
            Container(
              padding: const EdgeInsets.symmetric(vertical: 10),
              decoration: BoxDecoration(
                border: Border(
                  bottom: BorderSide(
                    color: Colors.grey[900]!,
                    width: 0.5,
                  ),
                ),
              ),
              child: Column(
                children: [
                  // Drag handle
                  Center(
                    child: Container(
                      height: 4,
                      width: 40,
                      decoration: BoxDecoration(
                        color: Colors.grey[600],
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  // Title
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        'Property Details',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      IconButton(
                        onPressed: () => Navigator.pop(context),
                        icon: const Icon(
                          Icons.close,
                          color: Colors.white,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            
            // Property content
            Expanded(
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 20),
                    // Address and Price
                    Text(
                      property.fullAddress,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      property.formattedPrice,
                      style: TextStyle(
                        color: AppColors.accent,
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 24),
                    
                    // Property Stats
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _buildStat(
                          property.beds.toString(),
                          'Beds',
                          Icons.bed,
                        ),
                        _buildStat(
                          property.baths.toString(),
                          'Baths',
                          Icons.bathtub_outlined,
                        ),
                        _buildStat(
                          '${property.squareFeet}',
                          'Sq Ft',
                          Icons.square_foot,
                        ),
                      ],
                    ),
                    const SizedBox(height: 24),
                    
                    // Description
                    const Text(
                      'Description',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      property.description,
                      style: TextStyle(
                        color: Colors.grey[300],
                        fontSize: 16,
                      ),
                    ),
                    const SizedBox(height: 24),
                    
                    // Features
                    if (property.features.isNotEmpty) ...[
                      const Text(
                        'Features',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 8),
                      ...property.features.map((feature) => 
                        Padding(
                          padding: const EdgeInsets.only(bottom: 8),
                          child: Row(
                            children: [
                              Icon(
                                Icons.check_circle,
                                color: AppColors.accent,
                                size: 20,
                              ),
                              const SizedBox(width: 8),
                              Text(
                                feature,
                                style: TextStyle(
                                  color: Colors.grey[300],
                                  fontSize: 16,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 24),
                    ],
                    
                    // Listed By
                    const Text(
                      'Listed By',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      property.agentName,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    Text(
                      property.agencyName,
                      style: TextStyle(
                        color: Colors.grey[300],
                        fontSize: 14,
                      ),
                    ),
                    const SizedBox(height: 40),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildStat(String value, String label, IconData icon) {
    return Column(
      children: [
        Icon(
          icon,
          color: AppColors.white,
          size: 24,
        ),
        const SizedBox(height: 4),
        Text(
          value,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),
        Text(
          label,
          style: TextStyle(
            color: Colors.grey[400],
            fontSize: 14,
          ),
        ),
      ],
    );
  }

  String _getTimeAgo(DateTime dateTime) {
    final difference = DateTime.now().difference(dateTime);
    if (difference.inSeconds < 60) {
      return '${difference.inSeconds}s ago';
    } else if (difference.inMinutes < 60) {
      return '${difference.inMinutes}m ago';
    } else if (difference.inHours < 24) {
      return '${difference.inHours}h ago';
    } else if (difference.inDays < 7) {
      return '${difference.inDays}d ago';
    } else {
      return '${difference.inDays ~/ 7}w ago';
    }
  }
}
