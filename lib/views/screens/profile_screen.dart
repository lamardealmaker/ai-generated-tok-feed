import 'dart:io';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:image_picker/image_picker.dart';
import '../../controllers/auth_controller.dart';
import '../../constants.dart';
import '../../services/user_service.dart';
import '../../utils/avatar_generator.dart';

class ProfileScreen extends StatelessWidget {
  ProfileScreen({super.key});

  final AuthController _authController = Get.find<AuthController>();
  final UserService _userService = UserService();
  final ImagePicker _picker = ImagePicker();

  Future<void> _pickImage() async {
    try {
      final XFile? image = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 512,
        maxHeight: 512,
        imageQuality: 75,
      );

      if (image != null && _authController.user != null) {
        final File imageFile = File(image.path);
        await _userService.uploadProfileImage(
          _authController.user!.uid,
          imageFile,
        );
        // Refresh user model to get new image URL
        await _authController.refreshUserModel();
      }
    } catch (e) {
      Get.snackbar(
        'Error',
        'Failed to upload image: $e',
        snackPosition: SnackPosition.BOTTOM,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text(
          'Profile',
          style: TextStyle(color: AppColors.white),
        ),
      ),
      body: Obx(() {
        final user = _authController.userModel;
        if (user == null) return const Center(child: CircularProgressIndicator());

        return SingleChildScrollView(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            children: [
              // Profile Image or Generated Avatar
              Center(
                child: Stack(
                  children: [
                    if (user.profileImageUrl != null)
                      CircleAvatar(
                        radius: 64,
                        backgroundImage: NetworkImage(user.profileImageUrl!),
                      )
                    else
                      AvatarGenerator.generateAvatar(
                        username: user.username,
                        email: user.email,
                        radius: 64,
                      ),
                    Positioned(
                      bottom: 0,
                      right: 0,
                      child: CircleAvatar(
                        backgroundColor: AppColors.accent,
                        radius: 20,
                        child: IconButton(
                          icon: const Icon(Icons.camera_alt, size: 20),
                          color: Colors.white,
                          onPressed: _pickImage,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),

              // User Info
              _buildInfoTile('Email', user.email),
              if (user.username != null)
                _buildInfoTile('Username', user.username!),
              if (user.fullName != null)
                _buildInfoTile('Full Name', user.fullName!),
              if (user.phoneNumber != null)
                _buildInfoTile('Phone', user.phoneNumber!),
              _buildInfoTile('Account Type', user.userType),
              _buildInfoTile(
                'Member Since',
                '${user.createdAt.day}/${user.createdAt.month}/${user.createdAt.year}',
              ),

              const SizedBox(height: 24),

              // Edit Profile Button
              ElevatedButton(
                style: AppTheme.primaryButton,
                onPressed: () {
                  Get.snackbar(
                    'Coming Soon',
                    'Edit profile feature will be available soon!',
                    snackPosition: SnackPosition.BOTTOM,
                  );
                },
                child: const Text(
                  'Edit Profile',
                  style: TextStyle(
                    fontSize: AppTheme.fontSize_md,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
        );
      }),
    );
  }

  Widget _buildInfoTile(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 100,
            child: Text(
              label,
              style: const TextStyle(
                color: AppColors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: const TextStyle(color: AppColors.white),
            ),
          ),
        ],
      ),
    );
  }
}
