import 'package:flutter/material.dart';

class AvatarGenerator {
  static Color generateColor(String text) {
    // Use username to generate a consistent color
    int hash = text.hashCode;
    
    // Predefined vibrant colors that look good
    final List<Color> colors = [
      const Color(0xFF1abc9c), // Turquoise
      const Color(0xFF2ecc71), // Emerald
      const Color(0xFF3498db), // Peter River
      const Color(0xFF9b59b6), // Amethyst
      const Color(0xFFf1c40f), // Sun Flower
      const Color(0xFFe67e22), // Carrot
      const Color(0xFFe74c3c), // Alizarin
      const Color(0xFF34495e), // Wet Asphalt
    ];

    // Use hash to pick a color
    return colors[hash.abs() % colors.length];
  }

  static String generateInitials(String? username, String email) {
    if (username != null && username.isNotEmpty) {
      // If username has multiple parts (e.g., "john_doe"), use first letter of each part
      final parts = username.split(RegExp(r'[._\- ]'));
      if (parts.length > 1) {
        return (parts[0][0] + parts[1][0]).toUpperCase();
      }
      // If username is single word, use first two letters
      return username.substring(0, username.length >= 2 ? 2 : 1).toUpperCase();
    }

    // Fallback to email
    final emailParts = email.split('@')[0].split(RegExp(r'[._\- ]'));
    if (emailParts.length > 1) {
      return (emailParts[0][0] + emailParts[1][0]).toUpperCase();
    }
    return email.substring(0, email.length >= 2 ? 2 : 1).toUpperCase();
  }

  static Widget generateAvatar({
    required String? username,
    required String email,
    double radius = 20,
  }) {
    final initials = generateInitials(username, email);
    final color = generateColor(username ?? email);

    return CircleAvatar(
      radius: radius,
      backgroundColor: color,
      child: Text(
        initials,
        style: TextStyle(
          color: Colors.white,
          fontWeight: FontWeight.bold,
          fontSize: radius * 0.8,
        ),
      ),
    );
  }
}
