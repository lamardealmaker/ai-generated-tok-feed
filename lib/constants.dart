import 'package:flutter/material.dart';

// App theme colors
class AppColors {
  static const Color primary = Color(0xFF121212);
  static const Color accent = Color(0xFFFF0050);    // Back to pink/red
  static const Color secondary = Color(0xFFE8E8E8);
  static const Color background = Color(0xFF000000);
  static const Color darkGrey = Color(0xFF121212);
  static const Color grey = Color(0xFF757575);
  static const Color white = Color(0xFFFFFFFF);
  static const Color buttonText = Color(0xFFFFFFFF); // White text for buttons
}

// Material design constants
class AppTheme {
  // Font Sizes
  static const double fontSize_xs = 12.0;
  static const double fontSize_sm = 14.0;
  static const double fontSize_md = 16.0;
  static const double fontSize_lg = 18.0;
  static const double fontSize_xl = 24.0;

  // Spacing
  static const double spacing_xs = 4.0;
  static const double spacing_sm = 8.0;
  static const double spacing_md = 16.0;
  static const double spacing_lg = 24.0;
  static const double spacing_xl = 32.0;

  // Border Radius
  static const double borderRadius_sm = 4.0;
  static const double borderRadius_md = 8.0;
  static const double borderRadius_lg = 12.0;

  // Icon Sizes
  static const double iconSize_sm = 16.0;
  static const double iconSize_md = 24.0;
  static const double iconSize_lg = 32.0;

  // Button Styles
  static final ButtonStyle primaryButton = ElevatedButton.styleFrom(
    backgroundColor: AppColors.accent,
    foregroundColor: AppColors.buttonText, // White text
    padding: const EdgeInsets.symmetric(
      horizontal: spacing_lg,
      vertical: spacing_md,
    ),
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(borderRadius_md),
    ),
  );

  static final ButtonStyle secondaryButton = ElevatedButton.styleFrom(
    backgroundColor: AppColors.darkGrey,
    foregroundColor: AppColors.white, // White text
    padding: const EdgeInsets.symmetric(
      horizontal: spacing_lg,
      vertical: spacing_md,
    ),
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(borderRadius_md),
    ),
  );
}

// App-specific constants
class AppConstants {
  static const String appName = 'Real Estate Tok';
}
