import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'package:get/get.dart';
import '../../models/video_model.dart';
import '../../controllers/video_controller.dart';
import '../../models/comment_model.dart';
import '../../models/property_details.dart';
import '../../constants.dart';
import 'package:url_launcher/url_launcher_string.dart';
import 'video_interaction_button.dart';
import 'video_progress_bar.dart';

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
  final ValueNotifier<Duration> _position = ValueNotifier(Duration.zero);
  bool _hasTriggeredCompletion = false;

  @override
  void initState() {
    super.initState();
    _initializeVideo();
  }

  void _setupVideoController() {
    if (_videoPlayerController != null) {
      // Listen for position updates
      _videoPlayerController!.addListener(() {
        if (!mounted) return;
        
        // Update progress bar
        _position.value = _videoPlayerController!.value.position;
        
        // Check for video completion - use a threshold approach
        if (_videoPlayerController!.value.duration.inMilliseconds > 0 &&
            _videoPlayerController!.value.position.inMilliseconds >= 
            _videoPlayerController!.value.duration.inMilliseconds - 50) {  // Trigger slightly earlier
          // Only trigger once near the end
          if (!_hasTriggeredCompletion) {
            _hasTriggeredCompletion = true;
            videoController.onVideoFinished();
          }
        } else {
          _hasTriggeredCompletion = false;
        }
      });
      
      // Ensure video doesn't loop
      _videoPlayerController!.setLooping(false);
    }
  }

  void _initializeVideo() {
    _videoPlayerController = videoController.getController(widget.video.id);
    if (_videoPlayerController != null) {
      setState(() {
        _isInitialized = true;
      });
      _setupVideoController();
    }
  }

  @override
  void didUpdateWidget(VideoPlayerItem oldWidget) {
    super.didUpdateWidget(oldWidget);
    
    final newController = videoController.getController(widget.video.id);
    if (newController != _videoPlayerController) {
      setState(() {
        _videoPlayerController = newController;
        _isInitialized = newController != null;
      });
      _setupVideoController();
    }

    if (widget.isPlaying != oldWidget.isPlaying) {
      if (widget.isPlaying) {
        _videoPlayerController?.play();
      } else {
        _videoPlayerController?.pause();
      }
    }
  }

  @override
  void dispose() {
    _position.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        // Video Player
        Container(
          width: MediaQuery.of(context).size.width,
          height: MediaQuery.of(context).size.height,
          decoration: const BoxDecoration(
            color: AppColors.background,
          ),
          child: _isInitialized && _videoPlayerController != null
              ? AspectRatio(
                  aspectRatio: _videoPlayerController!.value.aspectRatio,
                  child: VideoPlayer(_videoPlayerController!),
                )
              : const Center(
                  child: CircularProgressIndicator(
                    color: AppColors.accent,
                  ),
                ),
        ),

        // Progress Bar
        Positioned(
          top: 0,
          left: 0,
          right: 0,
          child: _isInitialized && _videoPlayerController != null
              ? ValueListenableBuilder<Duration>(
                  valueListenable: _position,
                  builder: (context, position, child) {
                    return VideoProgressBar(
                      position: position,
                      duration: _videoPlayerController!.value.duration,
                      height: 3.5,
                      color: Colors.white,
                    );
                  },
                )
              : const SizedBox.shrink(),
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
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.bottomCenter,
                end: Alignment.topCenter,
                colors: [
                  Colors.black.withOpacity(0.8),
                  Colors.transparent,
                ],
                stops: const [0.0, 0.8],
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                if (widget.video.propertyDetails != null) ...[
                  // Price
                  Text(
                    widget.video.propertyDetails!.formattedPrice,
                    style: const TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.bold,
                      color: AppColors.accent,
                    ),
                  ),
                  const SizedBox(height: 4),

                  // Location
                  Text(
                    widget.video.propertyDetails!.fullAddress,
                    style: const TextStyle(
                      fontSize: 14,
                      color: AppColors.white,
                    ),
                  ),
                  const SizedBox(height: 8),

                  // Key Features Row
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      _buildCompactFeatureItem(
                        Icons.bed,
                        '${widget.video.propertyDetails!.beds}',
                        AppColors.save,
                      ),
                      const SizedBox(width: 12),
                      _buildCompactFeatureItem(
                        Icons.bathroom,
                        '${widget.video.propertyDetails!.baths}',
                        AppColors.save,
                      ),
                      const SizedBox(width: 12),
                      _buildCompactFeatureItem(
                        Icons.square_foot,
                        '${widget.video.propertyDetails!.squareFeet}',
                        AppColors.save,
                      ),
                    ],
                  ),
                ],
              ],
            ),
          ),
        ),

        // Interaction Buttons
        Positioned(
          right: 8,
          bottom: 100,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Like Button
              Obx(() => VideoInteractionButton(
                icon: widget.video.isLiked.value ? Icons.favorite : Icons.favorite_border,
                count: widget.video.likes.toString(),
                onPressed: () => videoController.likeVideo(widget.video.id),
                isSelected: widget.video.isLiked.value,
              )),
              const SizedBox(height: 4),

              // Comment Button
              VideoInteractionButton(
                icon: Icons.comment,
                count: widget.video.comments.toString(),
                onPressed: () => _showCommentsSheet(context),
              ),
              const SizedBox(height: 4),

              // Share Button
              VideoInteractionButton(
                icon: Icons.share,
                count: widget.video.shares.toString(),
                onPressed: () => videoController.shareVideo(widget.video.id),
              ),
              const SizedBox(height: 4),

              // Property Info Button
              VideoInteractionButton(
                icon: Icons.info_outline,
                count: 'Info',
                onPressed: () => _showPropertySheet(context),
              ),
              const SizedBox(height: 4),

              // Favorite Button
              Obx(() => VideoInteractionButton(
                icon: widget.video.isFavorite.value ? Icons.bookmark : Icons.bookmark_border,
                count: 'Save',
                onPressed: () => videoController.toggleFavorite(widget.video.id),
                isSelected: widget.video.isFavorite.value,
              )),
            ],
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
    final property = widget.video.propertyDetails;
    if (property == null) return;

    showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.background,
      isScrollControlled: true,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.9,
        maxChildSize: 0.9,
        minChildSize: 0.5,
        expand: false,
        builder: (context, scrollController) => SingleChildScrollView(
          controller: scrollController,
          child: Container(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Price Section
                Text(
                  property.formattedPrice,
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: AppColors.accent,
                  ),
                ),
                const SizedBox(height: 8),

                // Address
                Text(
                  property.fullAddress,
                  style: const TextStyle(
                    fontSize: 18,
                    color: AppColors.white,
                  ),
                ),
                const SizedBox(height: 20),

                // Key Features Row
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _buildFeatureItem(
                      Icons.bed,
                      '${property.beds} beds',
                      AppColors.save,
                    ),
                    _buildFeatureItem(
                      Icons.bathroom,
                      '${property.baths} baths',
                      AppColors.save,
                    ),
                    _buildFeatureItem(
                      Icons.square_foot,
                      '${property.squareFeet} sq ft',
                      AppColors.save,
                    ),
                  ],
                ),
                const SizedBox(height: 24),

                // Description Section
                _buildSectionHeader('Description'),
                const SizedBox(height: 8),
                Text(
                  property.description,
                  style: const TextStyle(
                    color: AppColors.white,
                    fontSize: 16,
                    height: 1.5,
                  ),
                ),
                const SizedBox(height: 24),

                // Features Section
                _buildSectionHeader('Features'),
                const SizedBox(height: 12),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: property.features.map((feature) => Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 6,
                    ),
                    decoration: BoxDecoration(
                      color: AppColors.accent.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(
                        color: AppColors.accent.withOpacity(0.3),
                      ),
                    ),
                    child: Text(
                      feature,
                      style: TextStyle(
                        color: AppColors.accent,
                        fontSize: 14,
                      ),
                    ),
                  )).toList(),
                ),
                const SizedBox(height: 24),

                // Agent Section
                _buildSectionHeader('Listed By'),
                const SizedBox(height: 12),
                Row(
                  children: [
                    CircleAvatar(
                      radius: 24,
                      backgroundColor: AppColors.darkGrey,
                      child: Text(
                        property.agentName.isNotEmpty ? property.agentName[0].toUpperCase() : '?',
                        style: const TextStyle(
                          color: AppColors.accent,
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            property.agentName,
                            style: const TextStyle(
                              color: AppColors.white,
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          Text(
                            property.agencyName,
                            style: TextStyle(
                              color: AppColors.white.withOpacity(0.7),
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 24),


              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Container(
      padding: const EdgeInsets.only(bottom: 8),
      decoration: BoxDecoration(
        border: Border(
          bottom: BorderSide(
            color: AppColors.accent.withOpacity(0.3),
            width: 2,
          ),
        ),
      ),
      child: Text(
        title,
        style: const TextStyle(
          color: AppColors.white,
          fontSize: 20,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  Widget _buildFeatureItem(IconData icon, String text, Color color) {
    return Column(
      children: [
        Icon(
          icon,
          color: color,
          size: 28,
        ),
        const SizedBox(height: 4),
        Text(
          text,
          style: const TextStyle(
            color: AppColors.white,
            fontSize: 14,
          ),
        ),
      ],
    );
  }

  Widget _buildCompactFeatureItem(IconData icon, String text, Color color) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(
          icon,
          color: color,
          size: 16,
        ),
        const SizedBox(width: 4),
        Text(
          text,
          style: const TextStyle(
            color: AppColors.white,
            fontSize: 12,
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
