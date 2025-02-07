import 'package:flutter/material.dart';

// App theme colors
class AppColors {
  static const Color primary = Color(0xFF121212);
  static const Color accent = Color(0xFF00FF91);    // Mint green - main brand color
  static const Color heart = Color(0xFFFF4D8D);     // Soft pink - for likes
  static const Color save = Color(0xFF7B61FF);      // Indigo - for bookmarks
  static const Color error = Color(0xFFFF0000);     // Keep red for errors only
  static const Color secondary = Color(0xFFE8E8E8);
  static const Color background = Color(0xFF000000);
  static const Color darkGrey = Color(0xFF121212);
  static const Color grey = Color(0xFF757575);
  static const Color white = Color(0xFFFFFFFF);
  static const Color black = Color(0xFF000000);
  static const Color buttonText = Color(0xFF000000); // Black text for buttons

  // Gradient colors for overlays
  static final Color overlayStart = Colors.black.withOpacity(0.8);
  static final Color overlayEnd = Colors.transparent;
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
    foregroundColor: AppColors.buttonText, // Black text
    padding: const EdgeInsets.symmetric(
      horizontal: spacing_lg,
      vertical: spacing_md,
    ),
    textStyle: const TextStyle(
      fontSize: fontSize_md,
      fontWeight: FontWeight.bold,
    ),
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(borderRadius_md),
    ),
  );

  static final ButtonStyle secondaryButton = ElevatedButton.styleFrom(
    backgroundColor: AppColors.secondary,
    foregroundColor: AppColors.buttonText, // Black text
    padding: const EdgeInsets.symmetric(
      horizontal: spacing_lg,
      vertical: spacing_md,
    ),
    textStyle: const TextStyle(
      fontSize: fontSize_md,
      fontWeight: FontWeight.bold,
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
