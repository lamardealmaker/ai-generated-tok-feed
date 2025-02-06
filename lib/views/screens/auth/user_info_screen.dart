import 'package:flutter/material.dart';
import '../../../constants.dart';

class UserInfoScreen extends StatefulWidget {
  const UserInfoScreen({super.key});

  @override
  State<UserInfoScreen> createState() => _UserInfoScreenState();
}

class _UserInfoScreenState extends State<UserInfoScreen> {
  final TextEditingController _usernameController = TextEditingController();
  final TextEditingController _bioController = TextEditingController();
  DateTime? _selectedDate;
  String? _profileImageUrl;

  @override
  void dispose() {
    _usernameController.dispose();
    _bioController.dispose();
    super.dispose();
  }

  Future<void> _selectDate() async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: DateTime.now().subtract(const Duration(days: 365 * 18)),
      firstDate: DateTime.now().subtract(const Duration(days: 365 * 100)),
      lastDate: DateTime.now(),
      builder: (context, child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: ColorScheme.dark(
              primary: AppColors.accent,
              onPrimary: AppColors.white,
              surface: AppColors.background,
              onSurface: AppColors.white,
            ),
          ),
          child: child!,
        );
      },
    );
    if (picked != null) {
      setState(() {
        _selectedDate = picked;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: AppColors.white),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(AppTheme.spacing_lg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Complete your profile',
                style: TextStyle(
                  color: AppColors.white,
                  fontSize: AppTheme.fontSize_xl,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: AppTheme.spacing_md),
              const Text(
                'Add your details to create your account',
                style: TextStyle(
                  color: AppColors.grey,
                  fontSize: AppTheme.fontSize_md,
                ),
              ),
              const SizedBox(height: AppTheme.spacing_xl * 2),
              // Profile Image
              Center(
                child: GestureDetector(
                  onTap: () {
                    // TODO: Implement image picker
                  },
                  child: Container(
                    width: 120,
                    height: 120,
                    decoration: BoxDecoration(
                      color: AppColors.darkGrey,
                      shape: BoxShape.circle,
                      image: _profileImageUrl != null
                          ? DecorationImage(
                              image: NetworkImage(_profileImageUrl!),
                              fit: BoxFit.cover,
                            )
                          : null,
                    ),
                    child: _profileImageUrl == null
                        ? const Icon(
                            Icons.camera_alt,
                            color: AppColors.grey,
                            size: AppTheme.iconSize_lg,
                          )
                        : null,
                  ),
                ),
              ),
              const SizedBox(height: AppTheme.spacing_xl),
              // Username field
              TextField(
                controller: _usernameController,
                style: const TextStyle(color: AppColors.white),
                decoration: InputDecoration(
                  hintText: 'Username',
                  hintStyle: const TextStyle(color: AppColors.grey),
                  filled: true,
                  fillColor: AppColors.darkGrey,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(AppTheme.borderRadius_md),
                    borderSide: BorderSide.none,
                  ),
                  prefixIcon: const Icon(Icons.person, color: AppColors.grey),
                ),
              ),
              const SizedBox(height: AppTheme.spacing_md),
              // Birthday field
              GestureDetector(
                onTap: _selectDate,
                child: Container(
                  padding: const EdgeInsets.all(AppTheme.spacing_md),
                  decoration: BoxDecoration(
                    color: AppColors.darkGrey,
                    borderRadius: BorderRadius.circular(AppTheme.borderRadius_md),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.cake, color: AppColors.grey),
                      const SizedBox(width: AppTheme.spacing_md),
                      Text(
                        _selectedDate != null
                            ? '${_selectedDate!.month}/${_selectedDate!.day}/${_selectedDate!.year}'
                            : 'Birthday',
                        style: TextStyle(
                          color: _selectedDate != null
                              ? AppColors.white
                              : AppColors.grey,
                          fontSize: AppTheme.fontSize_md,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: AppTheme.spacing_md),
              // Bio field
              TextField(
                controller: _bioController,
                style: const TextStyle(color: AppColors.white),
                maxLines: 3,
                maxLength: 150,
                decoration: InputDecoration(
                  hintText: 'Bio (optional)',
                  hintStyle: const TextStyle(color: AppColors.grey),
                  filled: true,
                  fillColor: AppColors.darkGrey,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(AppTheme.borderRadius_md),
                    borderSide: BorderSide.none,
                  ),
                  counterStyle: const TextStyle(color: AppColors.grey),
                ),
              ),
              const SizedBox(height: AppTheme.spacing_xl * 2),
              // Create account button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _usernameController.text.isNotEmpty && _selectedDate != null
                      ? () {
                          // TODO: Implement account creation
                        }
                      : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.accent,
                    disabledBackgroundColor: AppColors.accent.withOpacity(0.5),
                    padding: const EdgeInsets.symmetric(vertical: AppTheme.spacing_md),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(AppTheme.borderRadius_md),
                    ),
                  ),
                  child: const Text(
                    'Create Account',
                    style: TextStyle(
                      color: AppColors.white,
                      fontSize: AppTheme.fontSize_lg,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
