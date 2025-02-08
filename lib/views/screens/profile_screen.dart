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
                    Container(
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: AppColors.accent,
                          width: 2,
                        ),
                      ),
                      child: user.profileImageUrl != null
                          ? CircleAvatar(
                              radius: 64,
                              backgroundImage: NetworkImage(user.profileImageUrl!),
                            )
                          : AvatarGenerator.generateAvatar(
                              username: user.username,
                              email: user.email,
                              radius: 64,
                            ),
                    ),
                    Positioned(
                      bottom: 0,
                      right: 0,
                      child: Container(
                        decoration: BoxDecoration(
                          color: AppColors.accent,
                          shape: BoxShape.circle,
                          border: Border.all(
                            color: AppColors.background,
                            width: 2,
                          ),
                        ),
                        child: IconButton(
                          icon: const Icon(Icons.camera_alt, size: 20),
                          color: AppColors.buttonText,
                          onPressed: _pickImage,
                          constraints: const BoxConstraints(
                            minWidth: 40,
                            minHeight: 40,
                          ),
                          padding: EdgeInsets.zero,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 24),

              // Username
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: AppColors.darkGrey,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text(
                          'Username',
                          style: TextStyle(
                            color: AppColors.grey,
                            fontSize: 14,
                          ),
                        ),
                        TextButton(
                          onPressed: () {
                            // Show username edit dialog
                            Get.dialog(
                              AlertDialog(
                                backgroundColor: AppColors.darkGrey,
                                title: const Text(
                                  'Edit Username',
                                  style: TextStyle(color: AppColors.white),
                                ),
                                content: TextField(
                                  controller: TextEditingController(text: user.username),
                                  style: const TextStyle(color: AppColors.white),
                                  decoration: const InputDecoration(
                                    hintText: 'Enter new username',
                                    hintStyle: TextStyle(color: AppColors.grey),
                                  ),
                                  onSubmitted: (value) async {
                                    if (value.isNotEmpty) {
                                      await _userService.updateUsername(user.uid, value);
                                      await _authController.refreshUserModel();
                                      Get.back();
                                    }
                                  },
                                ),
                                actions: [
                                  TextButton(
                                    onPressed: () => Get.back(),
                                    child: const Text('Cancel'),
                                  ),
                                ],
                              ),
                            );
                          },
                          child: const Text(
                            'Edit',
                            style: TextStyle(
                              color: AppColors.accent,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(
                      user.username ?? 'Set your username',
                      style: const TextStyle(
                        color: AppColors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 24),

              // Sign Out Button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: () => _authController.signOut(),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.darkGrey,
                    foregroundColor: AppColors.white,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: const Text(
                    'Sign Out',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ],
          ),
        );
      }),
    );
  }
}
