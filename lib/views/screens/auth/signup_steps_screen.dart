import 'dart:io';
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:image_picker/image_picker.dart';
import '../../../constants.dart';
import '../../../controllers/auth_controller.dart';
import '../../../services/user_service.dart';

class SignupStepsScreen extends StatefulWidget {
  final String email;
  final String password;

  const SignupStepsScreen({
    super.key,
    required this.email,
    required this.password,
  });

  @override
  State<SignupStepsScreen> createState() => _SignupStepsScreenState();
}

class _SignupStepsScreenState extends State<SignupStepsScreen> {
  final _usernameController = TextEditingController();
  final _authController = Get.find<AuthController>();
  final _userService = UserService();
  final _imagePicker = ImagePicker();
  
  int _currentStep = 0;
  File? _selectedImage;
  int _selectedDefaultAvatar = 0;
  bool _isUsernameAvailable = false;
  bool _isCheckingUsername = false;
  Timer? _debounceTimer;

  final List<String> _defaultAvatars = [
    'assets/avatars/avatar1.png',
    'assets/avatars/avatar2.png',
    'assets/avatars/avatar3.png',
    'assets/avatars/avatar4.png',
    'assets/avatars/avatar5.png',
  ];

  @override
  void dispose() {
    _usernameController.dispose();
    _debounceTimer?.cancel();
    super.dispose();
  }

  Future<void> _checkUsername(String username) async {
    if (username.length < 3) {
      setState(() => _isUsernameAvailable = false);
      return;
    }

    setState(() => _isCheckingUsername = true);

    // Debounce the username check
    _debounceTimer?.cancel();
    _debounceTimer = Timer(const Duration(milliseconds: 500), () async {
      try {
        final isAvailable = await _userService.isUsernameAvailable(username);
        if (mounted) {
          setState(() {
            _isUsernameAvailable = isAvailable;
            _isCheckingUsername = false;
          });
        }
      } catch (e) {
        if (mounted) {
          setState(() {
            _isUsernameAvailable = false;
            _isCheckingUsername = false;
          });
        }
      }
    });
  }

  Future<void> _pickImage() async {
    try {
      final XFile? image = await _imagePicker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 512,
        maxHeight: 512,
        imageQuality: 75,
      );

      if (image != null) {
        setState(() => _selectedImage = File(image.path));
      }
    } catch (e) {
      Get.snackbar(
        'Error',
        'Failed to pick image: $e',
        snackPosition: SnackPosition.BOTTOM,
      );
    }
  }

  Future<void> _completeSignup() async {
    try {
      final username = _usernameController.text.trim();
      
      if (!_isUsernameAvailable) {
        Get.snackbar(
          'Error',
          'Please choose a valid username',
          snackPosition: SnackPosition.BOTTOM,
        );
        return;
      }

      // Sign up with email and password
      final result = await _authController.signUp(
        email: widget.email,
        password: widget.password,
        username: username,
      );

      if (result == 'success' && _authController.user != null) {
        // Upload profile image if selected
        if (_selectedImage != null) {
          await _userService.uploadProfileImage(
            _authController.user!.uid,
            _selectedImage!,
          );
        } else if (_selectedDefaultAvatar > 0) {
          // Use default avatar
          await _userService.setDefaultAvatar(
            _authController.user!.uid,
            _selectedDefaultAvatar,
          );
        }

        Get.offAllNamed('/verify-email');
      } else {
        Get.snackbar(
          'Error',
          result,
          snackPosition: SnackPosition.BOTTOM,
        );
      }
    } catch (e) {
      Get.snackbar(
        'Error',
        e.toString(),
        snackPosition: SnackPosition.BOTTOM,
      );
    }
  }

  Widget _buildUsernameStep() {
    return Column(
      children: [
        TextField(
          controller: _usernameController,
          decoration: InputDecoration(
            hintText: 'Choose a username',
            prefixIcon: const Icon(Icons.person),
            suffixIcon: _isCheckingUsername
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : Icon(
                    _isUsernameAvailable ? Icons.check_circle : Icons.cancel,
                    color: _isUsernameAvailable ? AppColors.accent : AppColors.error,
                  ),
          ),
          onChanged: _checkUsername,
        ),
        const SizedBox(height: 16),
        Text(
          _isUsernameAvailable ? 'Username available' : 'Username taken',
          style: TextStyle(
            color: _isUsernameAvailable ? AppColors.accent : AppColors.error,
            fontSize: AppTheme.fontSize_sm,
          ),
        ),
      ],
    );
  }

  Widget _buildProfilePictureStep() {
    return Column(
      children: [
        if (_selectedImage != null)
          CircleAvatar(
            radius: 64,
            backgroundImage: FileImage(_selectedImage!),
          )
        else
          CircleAvatar(
            radius: 64,
            backgroundImage: _selectedDefaultAvatar > 0
                ? AssetImage(_defaultAvatars[_selectedDefaultAvatar - 1])
                : null,
            child: _selectedDefaultAvatar == 0
                ? const Icon(Icons.person, size: 64)
                : null,
          ),
        const SizedBox(height: 16),
        ElevatedButton.icon(
          icon: const Icon(Icons.photo_library),
          label: const Text('Choose from Gallery'),
          onPressed: _pickImage,
        ),
        const SizedBox(height: 24),
        const Text(
          'Or choose a default avatar:',
          style: TextStyle(color: AppColors.white),
        ),
        const SizedBox(height: 16),
        SizedBox(
          height: 100,
          child: ListView.builder(
            scrollDirection: Axis.horizontal,
            itemCount: _defaultAvatars.length,
            itemBuilder: (context, index) {
              return GestureDetector(
                onTap: () {
                  setState(() {
                    _selectedDefaultAvatar = index + 1;
                    _selectedImage = null;
                  });
                },
                child: Container(
                  margin: const EdgeInsets.symmetric(horizontal: 8),
                  decoration: BoxDecoration(
                    border: Border.all(
                      color: _selectedDefaultAvatar == index + 1
                          ? AppColors.accent
                          : Colors.transparent,
                      width: 2,
                    ),
                    shape: BoxShape.circle,
                  ),
                  child: CircleAvatar(
                    radius: 32,
                    backgroundImage: AssetImage(_defaultAvatars[index]),
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text('Complete Your Profile'),
      ),
      body: Stepper(
        currentStep: _currentStep,
        onStepContinue: () {
          if (_currentStep == 0 && !_isUsernameAvailable) {
            Get.snackbar(
              'Error',
              'Please choose a valid username',
              snackPosition: SnackPosition.BOTTOM,
            );
            return;
          }
          
          if (_currentStep < 1) {
            setState(() => _currentStep++);
          } else {
            _completeSignup();
          }
        },
        onStepCancel: () {
          if (_currentStep > 0) {
            setState(() => _currentStep--);
          }
        },
        steps: [
          Step(
            title: const Text('Choose Username'),
            content: _buildUsernameStep(),
            isActive: _currentStep >= 0,
          ),
          Step(
            title: const Text('Profile Picture'),
            content: _buildProfilePictureStep(),
            isActive: _currentStep >= 1,
          ),
        ],
      ),
    );
  }
}
