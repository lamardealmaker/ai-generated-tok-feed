import 'dart:ui';
import 'package:flutter/material.dart';
import '../../constants.dart';

class VideoInteractionButton extends StatelessWidget {
  final IconData icon;
  final String count;
  final VoidCallback onPressed;
  final bool isSelected;
  final bool showBlur;
  final Color? activeColor;

  const VideoInteractionButton({
    super.key,
    required this.icon,
    required this.count,
    required this.onPressed,
    this.isSelected = false,
    this.showBlur = true,
    this.activeColor,
  });

  @override
  Widget build(BuildContext context) {
    final color = isSelected ? (activeColor ?? AppColors.accent) : AppColors.textPrimary;

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 48,
          height: 48,
          margin: const EdgeInsets.symmetric(vertical: 4),
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: isSelected
                ? (activeColor ?? AppColors.accent).withOpacity(0.1)
                : (showBlur ? AppColors.darkGrey.withOpacity(0.85) : Colors.transparent),
            border: isSelected
                ? Border.all(
                    color: (activeColor ?? AppColors.accent).withOpacity(0.3),
                    width: 1,
                  )
                : null,
          ),
          child: ClipOval(
            child: showBlur
                ? BackdropFilter(
                    filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
                    child: _buildButton(color),
                  )
                : _buildButton(color),
          ),
        ),
        Text(
          count,
          style: TextStyle(
            color: color,
            fontSize: 12,
            fontWeight: isSelected ? FontWeight.w600 : FontWeight.w500,
            letterSpacing: 0.2,
            shadows: const [
              Shadow(
                offset: Offset(0, 1),
                blurRadius: 2,
                color: Color(0x40000000),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildButton(Color color) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onPressed,
        customBorder: const CircleBorder(),
        child: Center(
          child: Icon(
            icon,
            color: color,
            size: 24,
            shadows: const [
              Shadow(
                offset: Offset(0, 1),
                blurRadius: 2,
                color: Color(0x40000000),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
