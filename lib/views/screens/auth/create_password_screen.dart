import 'package:flutter/material.dart';
import '../../../constants.dart';

class CreatePasswordScreen extends StatefulWidget {
  const CreatePasswordScreen({super.key});

  @override
  State<CreatePasswordScreen> createState() => _CreatePasswordScreenState();
}

class _CreatePasswordScreenState extends State<CreatePasswordScreen> {
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _confirmPasswordController = TextEditingController();
  bool _isPasswordVisible = false;
  bool _isConfirmPasswordVisible = false;
  
  final List<String> _passwordRequirements = [
    '8-20 characters',
    'Letters, numbers, and special characters',
    'No spaces or special characters',
  ];

  Map<String, bool> _requirementsMet = {
    '8-20 characters': false,
    'Letters, numbers, and special characters': false,
    'No spaces or special characters': false,
  };

  @override
  void dispose() {
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    super.dispose();
  }

  void _checkPasswordStrength(String password) {
    setState(() {
      _requirementsMet['8-20 characters'] = 
          password.length >= 8 && password.length <= 20;
      
      _requirementsMet['Letters, numbers, and special characters'] = 
          RegExp(r'^(?=.*[A-Za-z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]').hasMatch(password);
      
      _requirementsMet['No spaces or special characters'] = 
          !password.contains(' ');
    });
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
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Create password',
                style: TextStyle(
                  color: AppColors.white,
                  fontSize: AppTheme.fontSize_xl,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: AppTheme.spacing_md),
              const Text(
                'Create a secure password for your account',
                style: TextStyle(
                  color: AppColors.grey,
                  fontSize: AppTheme.fontSize_md,
                ),
              ),
              const SizedBox(height: AppTheme.spacing_xl),
              // Password field
              TextField(
                controller: _passwordController,
                obscureText: !_isPasswordVisible,
                style: const TextStyle(color: AppColors.white),
                onChanged: _checkPasswordStrength,
                decoration: InputDecoration(
                  hintText: 'Password',
                  hintStyle: const TextStyle(color: AppColors.grey),
                  filled: true,
                  fillColor: AppColors.darkGrey,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(AppTheme.borderRadius_md),
                    borderSide: BorderSide.none,
                  ),
                  suffixIcon: IconButton(
                    icon: Icon(
                      _isPasswordVisible ? Icons.visibility_off : Icons.visibility,
                      color: AppColors.grey,
                    ),
                    onPressed: () {
                      setState(() {
                        _isPasswordVisible = !_isPasswordVisible;
                      });
                    },
                  ),
                ),
              ),
              const SizedBox(height: AppTheme.spacing_md),
              // Confirm password field
              TextField(
                controller: _confirmPasswordController,
                obscureText: !_isConfirmPasswordVisible,
                style: const TextStyle(color: AppColors.white),
                decoration: InputDecoration(
                  hintText: 'Confirm password',
                  hintStyle: const TextStyle(color: AppColors.grey),
                  filled: true,
                  fillColor: AppColors.darkGrey,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(AppTheme.borderRadius_md),
                    borderSide: BorderSide.none,
                  ),
                  suffixIcon: IconButton(
                    icon: Icon(
                      _isConfirmPasswordVisible ? Icons.visibility_off : Icons.visibility,
                      color: AppColors.grey,
                    ),
                    onPressed: () {
                      setState(() {
                        _isConfirmPasswordVisible = !_isConfirmPasswordVisible;
                      });
                    },
                  ),
                ),
              ),
              const SizedBox(height: AppTheme.spacing_xl),
              // Password requirements
              const Text(
                'Password must contain:',
                style: TextStyle(
                  color: AppColors.white,
                  fontSize: AppTheme.fontSize_md,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: AppTheme.spacing_md),
              ...List.generate(
                _passwordRequirements.length,
                (index) => Padding(
                  padding: const EdgeInsets.only(bottom: AppTheme.spacing_sm),
                  child: Row(
                    children: [
                      Icon(
                        _requirementsMet[_passwordRequirements[index]] ?? false
                            ? Icons.check_circle
                            : Icons.circle_outlined,
                        color: _requirementsMet[_passwordRequirements[index]] ?? false
                            ? AppColors.accent
                            : AppColors.grey,
                        size: AppTheme.iconSize_sm,
                      ),
                      const SizedBox(width: AppTheme.spacing_sm),
                      Text(
                        _passwordRequirements[index],
                        style: TextStyle(
                          color: _requirementsMet[_passwordRequirements[index]] ?? false
                              ? AppColors.white
                              : AppColors.grey,
                          fontSize: AppTheme.fontSize_sm,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const Spacer(),
              // Next button
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _requirementsMet.values.every((met) => met) &&
                          _passwordController.text == _confirmPasswordController.text
                      ? () {
                          // TODO: Navigate to user info screen
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
                    'Next',
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
