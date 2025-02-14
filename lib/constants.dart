import 'package:flutter/material.dart';

// App theme colors
class AppColors {
  // Primary Colors
  static const Color primary = Color(0xFF1E2A39);    // Deep navy
  static const Color accent = Color(0xFF64B6AC);     // Muted teal
  static const Color background = Color(0xFF121417); // Soft black
  static const Color secondary = Color(0xFF4A90E2);  // Ocean blue (same as save for consistency)
  
  // Interactive Colors
  static const Color heart = Color(0xFFE76F51);      // Coral
  static const Color save = Color(0xFF4A90E2);       // Ocean blue
  
  // Neutral Colors
  static const Color grey = Color(0xFF9BA0A8);       // Warm grey
  static const Color lightGrey = Color(0xFFE5E9F0);  // Light warm grey
  static const Color darkGrey = Color(0xFF2C3440);   // Dark warm grey
  
  // Text Colors
  static const Color textPrimary = Color(0xFFF5F7FA);// Off-white
  static const Color textSecondary = Color(0xFFB0B7C3);// Muted text
  static const Color white = Color(0xFFFFFFFF);      // Pure white for contrast
  static const Color black = Color(0xFF000000);      // Pure black for contrast
  
  // Status Colors
  static const Color success = Color(0xFF34C759);    // Apple green
  static const Color error = Color(0xFFFF3B30);      // Soft red
  
  // Button Colors
  static const Color buttonText = Color(0xFF1E2A39); // Deep navy for button text
  static const Color buttonBackground = Color(0xFF64B6AC); // Muted teal for buttons
  
  // Overlay Colors
  static const Color overlay = Color(0x80000000);    // 50% black
  static const Color glassOverlay = Color(0x0DFFFFFF);// Subtle white
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
