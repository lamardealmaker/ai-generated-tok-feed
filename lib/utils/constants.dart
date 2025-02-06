import 'package:flutter/material.dart';

// App theme colors
class AppColors {
  static const Color primary = Color(0xFF121212);
  static const Color secondary = Color(0xFFE8E8E8);
  static const Color accent = Color(0xFFFF0050);
  
  // Common TikTok-like colors
  static const Color background = Color(0xFF000000);
  static const Color white = Color(0xFFFFFFFF);
  static const Color grey = Color(0xFF888888);
  static const Color lightGrey = Color(0xFFAAAAAA);
  static const Color darkGrey = Color(0xFF333333);
  
  // Action colors
  static const Color like = Color(0xFFFF4365);
  static const Color comment = Color(0xFFFFFFFF);
  static const Color share = Color(0xFFFFFFFF);
}

// Material design constants
class AppTheme {
  static const double spacing_xs = 4.0;
  static const double spacing_sm = 8.0;
  static const double spacing_md = 16.0;
  static const double spacing_lg = 24.0;
  static const double spacing_xl = 32.0;

  static const double fontSize_xs = 12.0;
  static const double fontSize_sm = 14.0;
  static const double fontSize_md = 16.0;
  static const double fontSize_lg = 20.0;
  static const double fontSize_xl = 24.0;

  static const double iconSize_sm = 24.0;
  static const double iconSize_md = 32.0;
  static const double iconSize_lg = 48.0;

  static const double borderRadius_sm = 4.0;
  static const double borderRadius_md = 8.0;
  static const double borderRadius_lg = 16.0;
  static const double borderRadius_xl = 24.0;
}

// App-specific constants
class AppConstants {
  static const String appName = "Real Estate Tok";
  static const Duration defaultAnimationDuration = Duration(milliseconds: 300);
  static const double bottomNavBarHeight = 50.0;
  static const double videoAspectRatio = 9 / 16;
}
