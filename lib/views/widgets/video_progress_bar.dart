import 'package:flutter/material.dart';

class VideoProgressBar extends StatelessWidget {
  final Duration position;
  final Duration duration;
  final double height;
  final Color color;

  const VideoProgressBar({
    super.key,
    required this.position,
    required this.duration,
    this.height = 2.0,
    this.color = Colors.white,
  });

  @override
  Widget build(BuildContext context) {
    final progress = duration.inMilliseconds > 0
        ? position.inMilliseconds / duration.inMilliseconds
        : 0.0;

    return Container(
      width: MediaQuery.of(context).size.width,
      height: height,
      decoration: BoxDecoration(
        color: color.withOpacity(0.2),
      ),
      child: LayoutBuilder(
        builder: (context, constraints) {
          return Row(
            children: [
              Container(
                width: constraints.maxWidth * progress,
                height: height,
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.only(
                    topRight: Radius.circular(height / 2),
                    bottomRight: Radius.circular(height / 2),
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}
