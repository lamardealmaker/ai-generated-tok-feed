import 'package:flutter/material.dart';
import '../../../constants.dart';
import 'verification_screen.dart';

class PhoneEmailLoginScreen extends StatefulWidget {
  const PhoneEmailLoginScreen({super.key});

  @override
  State<PhoneEmailLoginScreen> createState() => _PhoneEmailLoginScreenState();
}

class _PhoneEmailLoginScreenState extends State<PhoneEmailLoginScreen> {
  bool _isPhoneLogin = true;
  final TextEditingController _phoneController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();

  @override
  void dispose() {
    _phoneController.dispose();
    _emailController.dispose();
    super.dispose();
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
        child: Padding(
          padding: const EdgeInsets.all(AppTheme.spacing_lg),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Toggle buttons
              Container(
                decoration: BoxDecoration(
                  color: AppColors.darkGrey,
                  borderRadius: BorderRadius.circular(AppTheme.borderRadius_lg),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: _buildToggleButton(
                        text: 'Phone',
                        isSelected: _isPhoneLogin,
                        onTap: () => setState(() => _isPhoneLogin = true),
                      ),
                    ),
                    Expanded(
                      child: _buildToggleButton(
                        text: 'Email',
                        isSelected: !_isPhoneLogin,
                        onTap: () => setState(() => _isPhoneLogin = false),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: AppTheme.spacing_xl),
              // Input field
              if (_isPhoneLogin)
                TextField(
                  controller: _phoneController,
                  style: const TextStyle(color: AppColors.white),
                  keyboardType: TextInputType.phone,
                  decoration: InputDecoration(
                    hintText: 'Phone number',
                    hintStyle: const TextStyle(color: AppColors.grey),
                    filled: true,
                    fillColor: AppColors.darkGrey,
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(AppTheme.borderRadius_md),
                      borderSide: BorderSide.none,
                    ),
                    prefixIcon: const Icon(Icons.phone, color: AppColors.grey),
                  ),
                )
              else
                TextField(
                  controller: _emailController,
                  style: const TextStyle(color: AppColors.white),
                  keyboardType: TextInputType.emailAddress,
                  decoration: InputDecoration(
                    hintText: 'Email address',
                    hintStyle: const TextStyle(color: AppColors.grey),
                    filled: true,
                    fillColor: AppColors.darkGrey,
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(AppTheme.borderRadius_md),
                      borderSide: BorderSide.none,
                    ),
                    prefixIcon: const Icon(Icons.email, color: AppColors.grey),
                  ),
                ),
              const SizedBox(height: AppTheme.spacing_xl),
              // Next button
              ElevatedButton(
                onPressed: () {
                  final String contact = _isPhoneLogin
                      ? _phoneController.text
                      : _emailController.text;
                  if (contact.isNotEmpty) {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => VerificationScreen(
                          contactInfo: contact,
                          isEmail: !_isPhoneLogin,
                        ),
                      ),
                    );
                  }
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppColors.accent,
                  padding: const EdgeInsets.symmetric(vertical: AppTheme.spacing_md),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(AppTheme.borderRadius_md),
                  ),
                ),
                child: const Text(
                  'Next',
                  style: TextStyle(
                    color: AppColors.white,
                    fontSize: AppTheme.fontSize_lg,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildToggleButton({
    required String text,
    required bool isSelected,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: AppTheme.spacing_md),
        decoration: BoxDecoration(
          color: isSelected ? AppColors.accent : Colors.transparent,
          borderRadius: BorderRadius.circular(AppTheme.borderRadius_lg),
        ),
        child: Text(
          text,
          style: TextStyle(
            color: isSelected ? AppColors.white : AppColors.grey,
            fontSize: AppTheme.fontSize_md,
            fontWeight: FontWeight.w600,
          ),
          textAlign: TextAlign.center,
        ),
      ),
    );
  }
}
