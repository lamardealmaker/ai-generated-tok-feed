import 'dart:ui';
import 'package:flutter/material.dart';
import '../../constants.dart';

class VideoInteractionButton extends StatelessWidget {
  final IconData icon;
  final String count;
  final VoidCallback onPressed;
  final bool isSelected;
  final bool showBlur;

  const VideoInteractionButton({
    super.key,
    required this.icon,
    required this.count,
    required this.onPressed,
    this.isSelected = false,
    this.showBlur = true,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 44,
          height: 44,
          margin: const EdgeInsets.symmetric(vertical: 4),
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: showBlur ? Colors.black.withOpacity(0.2) : Colors.transparent,
          ),
          child: ClipOval(
            child: showBlur ? BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
              child: _buildButton(),
            ) : _buildButton(),
          ),
        ),
        Text(
          count,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 12,
            fontWeight: FontWeight.w500,
            letterSpacing: 0.2,
          ),
        ),
      ],
    );
  }

  Widget _buildButton() {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onPressed,
        customBorder: const CircleBorder(),
        child: Center(
          child: Icon(
            icon,
            color: isSelected ? AppColors.heart : Colors.white,
            size: 24,
          ),
        ),
      ),
    );
  }
}
